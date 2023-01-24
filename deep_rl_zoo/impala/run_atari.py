# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
From the paper "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures"
https://arxiv.org/abs/1802.01561.
"""
from absl import app
from absl import flags
from absl import logging
import multiprocessing
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import ImpalaActorCriticConvNet, ImpalaActorCriticNetworkInputs
from deep_rl_zoo.impala import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_integer('num_actors', 8, 'Number of actor processes to use, consider to use larger number like 32, 64.')
flags.DEFINE_bool('use_lstm', False, 'Use LSTM layer, default off.')
flags.DEFINE_bool('clip_grad', True, 'Clip gradients, default on.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')

flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('rmsprop_momentum', 0.0, 'RMSProp momentum.')
flags.DEFINE_float('rmsprop_eps', 0.01, 'RMSProp epsilon.')
flags.DEFINE_float('rmsprop_alpha', 0.99, 'RMSProp alpha.')

flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('entropy_coef', 0.025, 'Coefficient for the entropy loss.')
flags.DEFINE_float('baseline_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_integer('unroll_length', 80, 'How many agent time step to unroll for actor.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for learning, use larger batch size if possible.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_frames', int(1e6 / 4), 'Number of training frames (after frame skip) to run per iteration, per actor.'
)
flags.DEFINE_integer('num_eval_frames', int(1e5), 'Number of evaluation frames (after frame skip) to run per iteration.')
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_integer(
    'debug_screenshots_frequency',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', 'logs/impala_atari_results.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', '', 'Path for checkpoint directory.')


def main(argv):
    """Trains IMPALA agent on Atari."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs IMPALA agent on {runtime_device}')
    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member
    # Listen to signals to exit process.
    main_loop.handle_exit_signal()

    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Create environment.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name=FLAGS.environment_name,
            screen_height=FLAGS.environment_height,
            screen_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**32),
            noop_max=30,
            terminal_on_life_loss=True,
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', num_actions)
    logging.info('Observation spec: %s', state_dim)

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    # Create policy network for actors, learner will copy new weights to this network after batch learning
    actor_policy_network = ImpalaActorCriticConvNet(input_shape=state_dim, num_actions=num_actions, use_lstm=FLAGS.use_lstm)
    actor_policy_network.share_memory()

    # Create policy network for leaner to optimize
    policy_network = ImpalaActorCriticConvNet(input_shape=state_dim, num_actions=num_actions, use_lstm=FLAGS.use_lstm)
    policy_optimizer = torch.optim.RMSprop(
        policy_network.parameters(),
        lr=FLAGS.learning_rate,
        momentum=FLAGS.rmsprop_momentum,
        eps=FLAGS.rmsprop_eps,
        alpha=FLAGS.rmsprop_alpha,
    )

    # Test network output.
    pi_input = ImpalaActorCriticNetworkInputs(
        s_t=torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1=torch.zeros((1, 1)).long(),
        r_t=torch.zeros((1, 1)),
        done=torch.tensor(False)[None, ...],
        hidden_s=policy_network.get_initial_hidden_state(1),
    )

    network_output = policy_network(pi_input)
    assert network_output.pi_logits.shape == (1, 1, num_actions)
    assert network_output.baseline.shape == (1, 1)

    # Create queue shared between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors)

    # Use replay to overcome the problem when using small number of actors.
    replay = replay_lib.UniformReplay(
        capacity=100, structure=agent.TransitionStructure, random_state=random_state, time_major=True
    )

    # Create learner instance
    learner_agent = agent.Learner(
        policy_network=policy_network,
        actor_policy_network=actor_policy_network,
        policy_optimizer=policy_optimizer,
        replay=replay,
        discount=FLAGS.discount,
        unroll_length=FLAGS.unroll_length,
        batch_size=FLAGS.batch_size,
        entropy_coef=FLAGS.entropy_coef,
        baseline_coef=FLAGS.baseline_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create actor environments, runtime devices, and actor instances.
    actor_envs = [environment_builder() for _ in range(FLAGS.num_actors)]
    # TODO map to dedicated device if have multiple GPUs
    actor_devices = [runtime_device] * FLAGS.num_actors

    actors = [
        agent.Actor(
            rank=i,
            unroll_length=FLAGS.unroll_length,
            data_queue=data_queue,
            policy_network=actor_policy_network,
            device=actor_devices[i],
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.ImpalaGreedyActor(network=policy_network, device=runtime_device)

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='IMPALA', save_dir=FLAGS.checkpoint_dir)
    checkpoint.register_pair(('policy_network', policy_network))

    # Run parallel training N iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_frames=FLAGS.num_train_frames,
        num_eval_frames=FLAGS.num_eval_frames,
        network=policy_network,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_envs=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        tensorboard=FLAGS.tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_frequency=FLAGS.debug_screenshots_frequency,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)
