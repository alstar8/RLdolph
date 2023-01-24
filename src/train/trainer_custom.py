import pytorch_lightning as pl
import bitsandbytes as bnb
import torch
from src.rudolph.model.utils import get_attention_mask

from gym.spaces import Box
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deep_rl_zoo.networks.dqn import RainbowDqnMlpNet, RainbowDqnConvNet, R2d2DqnConvNet
from deep_rl_zoo import main_loop
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
import deep_rl_zoo.types as types_lib
import random
from moviepy.editor import ImageSequenceClip
import pandas as pd

from omegaconf import OmegaConf
from src.rudolph.model import get_rudolph_model
from src.inference.dataloader import DatasetRetriever
from src.inference.inference import run_inference
from torch.utils.data import DataLoader
#VQA
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import nltk
import wordtodigits as w2d
from data_for_inference.vqa_metrics import calc_meteor
import json
# Generation
import os
import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from data_for_inference.generation_metrics.fid import calculate_fid_given_paths
from data_for_inference.generation_metrics.inception import InceptionV3
from data_for_inference.generation_metrics.ruclip import calc_ruclip_metric
import json
# TextQA
import argparse
from data_for_inference.textqa_metrics.calculate_f1 import get_f1
import json
def read_json(file_name):
    with open(file_name) as f:
        result = json.load(f)
    return result

from torch.utils.data import DataLoader
from src.train.dataloader_custom import DatasetRetriever, fb_collate_fn, DEFAULT_SPC_TOKENS  ########### dataloader_custom
from src.train.utils import create_dataset
from rudalle import get_tokenizer, get_vae
from catalyst.data import BalanceClassSampler, DistributedSamplerWrapper
import pandas as pd
import webdataset as wds
import s3dataset
from PIL import Image
import io
import torchvision.transforms as T
from pytorch_lightning.trainer.supporters import CombinedLoader

from src.inference.inference_api import ruDolphApi
from tqdm import tqdm


class ObjectWrapper(DataLoader):
    def __init__(self, baseObject, llen):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.llen = llen
    def __len__(self):
        return self.llen
    
def my_split_by_worker(urls):
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[wi.id::wi.num_workers]
    
def my_node_splitter(urls):
    node_id = torch.distributed.get_rank()
    node_count = torch.distributed.get_world_size()
    
    wi = torch.utils.data.get_worker_info()
    if wi is None:
        return urls
    else:
        return urls[node_id::node_count]    

class RudolphLightning(pl.LightningModule):

    def __init__(self, vae, model, n_training_steps, conf, task_weights, model_params, model_freeze, scheduler_conf, bs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = vae
        self.model = freeze(model=model, **model_freeze)
        self.n_training_steps = n_training_steps
        self.conf = conf
        self.task_weights = task_weights
        self.model_params = model_params
        self.scheduler_conf = scheduler_conf
        self.bs = bs
        
        self.tokenizer = get_tokenizer()
        
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.model.get_param('image_tokens_per_dim')*8, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        
        dataset_textqa = create_dataset('textqa', **conf.data.textqa) if conf.tasks.textqa == True else []
        dataset_mathqa = create_dataset('mathqa', **conf.data.mathqa) if conf.tasks.mathqa == True else []
        dataset_generation = create_dataset('generation', **conf.data.generation) if conf.tasks.generation == True else []
        dataset_captioning = create_dataset('captioning', **conf.data.captioning) if conf.tasks.captioning == True else []
        dataset_vqa = create_dataset('vqa', **conf.data.vqa) if conf.tasks.vqa == True else []
        dataset_text_recogn = create_dataset('text_recognition', **conf.data.text_recognition) if conf.tasks.text_recognition == True else []

        dataset = [*dataset_textqa, *dataset_mathqa, *dataset_generation, *dataset_captioning, *dataset_vqa, *dataset_text_recogn]
        df = pd.DataFrame(dataset)  

        self.df_train = df[df['stage'] == 'train']
        self.df_val = df[df['stage'] == 'val']
        
        self.NUM_WORKERS = os.cpu_count()
        
        storage_options = {
            "anon": False,
            'key': 'officecds-user01',
            'secret':'jym0FuboUnR5VsPmCgYTGv1QQfglYZhPRWbEfS59',
            'client_kwargs': {
                'endpoint_url':'https://s3pd12.sbercloud.ru'
            }
        }
        s3dataset.init_webdataset(storage_options)
        
        self.eval_env_breakout = gym_env.create_atari_environment(
                env_name='Breakout',
                screen_height=84,
                screen_width=84,
                frame_skip=4,
                frame_stack=4,
                max_episode_steps=28000,
                seed=1,
                noop_max=30,
                terminal_on_life_loss=False,
                clip_reward=False,
            )
        
        self.eval_env_pong = gym_env.create_atari_environment(
                env_name='Pong',
                screen_height=84,
                screen_width=84,
                frame_skip=4,
                frame_stack=4,
                max_episode_steps=28000,
                seed=1,
                noop_max=30,
                terminal_on_life_loss=False,
                clip_reward=False,
            )
        
        print('############################################INIT DONE')
        
        self.tasks = [   '/home/jovyan/Aleksei_RL/fb_train_2022/vqa/vqa_val.json',
                         '/home/jovyan/Aleksei_RL/fb_train_2022/textqa/text_small_val.json']
        self.if_breakout = False
        self.if_pong = False
        
        
        
    def preprocess_s3_frame(self, item):
    
        left_special_token = '<LT_RLA>'
        right_special_token = '<RT_RLA>'

        lt = torch.zeros(self.conf.model.params.l_text_seq_length,dtype=torch.int32)
        lt[0] = 2
        lt[1] = DEFAULT_SPC_TOKENS[left_special_token]
        lt[2] = 3

        img = Image.open(io.BytesIO(item['data']))  
        img = self.image_transform(img)
        rt = torch.zeros(self.conf.model.params.r_text_seq_length,dtype=torch.int32)
        rt[0] = 2
        rt[1] = DEFAULT_SPC_TOKENS[right_special_token]
        rt[2] = DEFAULT_SPC_TOKENS['ATARI_{}'.format(item['action'])]
        rt[3] = 3

        return {
                'task_id': 'Atari_breakout',
                'left_text': lt,
                'image': img,
                'right_text': rt
                }    
        
    def forward(self, input_ids, attention_mask):
        logits, has_cache = self.model(input_ids, attention_mask, return_loss=False)
        return logits

    
    def atari_loss(self, bs, left_text, image, right_text, task):
        attention_mask = get_attention_mask(bs, self.model_params.l_text_seq_length, self.model_params.image_tokens_per_dim, self.model_params.r_text_seq_length, left_text.device)
        #attention_mask.shape (5,1,1280,1280)
        if image is None:
            image_seq_length = self.model_params.image_tokens_per_dim ** 2
            image_input_ids = torch.zeros((bs, image_seq_length), dtype=torch.int32).to(left_text.device)
        else:
            image_input_ids = self.vae.get_codebook_indices(image)
        if right_text is None:
            input_ids = torch.cat((left_text, image_input_ids), dim=1)
        else:
            input_ids = torch.cat((left_text, image_input_ids, right_text), dim=1)
        loss, loss_values = self.model.forward_atari(input_ids, attention_mask, lt_loss_weight=task.lt_loss_weight,
                                               img_loss_weight=task.img_loss_weight, rt_loss_weight=task.rt_loss_weight, 
                                               return_loss=True)
        return loss
  
    
    def get_loss(self, bs, left_text, image, right_text, task):
        attention_mask = get_attention_mask(bs, self.model_params.l_text_seq_length, self.model_params.image_tokens_per_dim, self.model_params.r_text_seq_length, left_text.device)
        if image is None:
            image_seq_length = self.model_params.image_tokens_per_dim ** 2
            image_input_ids = torch.zeros((bs, image_seq_length), dtype=torch.int32).to(left_text.device)
        else:
            image_input_ids = self.vae.get_codebook_indices(image)
        if right_text is None:
            input_ids = torch.cat((left_text, image_input_ids), dim=1)
        else:
            input_ids = torch.cat((left_text, image_input_ids, right_text), dim=1)
        loss, loss_values = self.model.forward(input_ids, attention_mask, lt_loss_weight=task.lt_loss_weight,
                                               img_loss_weight=task.img_loss_weight, rt_loss_weight=task.rt_loss_weight, 
                                               return_loss=True)
        return loss
    
    def rudolph_step(self, observation):
        left_special_token = '<LT_RLA>'
        right_special_token = '<RT_RLA>'

        lt = torch.zeros(self.model_params.l_text_seq_length,dtype=torch.int32)
        lt[0] = 2
        lt[1] = DEFAULT_SPC_TOKENS[left_special_token]
        lt[2] = 3
        rt = torch.zeros(2, dtype=torch.int32)
        rt[0] = 2
        rt[1] = DEFAULT_SPC_TOKENS[right_special_token]

        img = np.vstack((np.hstack((observation[0],observation[1])),np.hstack((observation[2],observation[3]))))
        img = Image.fromarray(img)
        img = self.image_transform(img)
        img = img.unsqueeze(0).to('cuda')
        image_input_ids_text = self.vae.get_codebook_indices(img, disable_gumbel_softmax=True)[0]

        attention_mask_text = get_attention_mask(1, self.model_params.l_text_seq_length,self.model_params.image_tokens_per_dim,2, 'cuda')
        #attention_mask_text[:,:,self.model_params.l_text_seq_length:-args.r_text_seq_length,:]*=0

        input_ids_text = torch.cat((lt.to('cuda').unsqueeze(0), image_input_ids_text.to('cuda').unsqueeze(0), rt.to('cuda').unsqueeze(0)), dim=1)

        with torch.no_grad():
            logits = self.model(input_ids_text, attention_mask_text)
        # Ниже код, в котором выполняется семплирование из выдаваемого моделью распределения
        #distribution = torch.softmax(logits[0][:, -1, DEFAULT_SPC_TOKENS['ATARI_0']:DEFAULT_SPC_TOKENS['ATARI_0']+4], 1)
        #a_t = torch.multinomial(distribution, 1).item()
        a_t = torch.argmax(logits[0][:, -1, DEFAULT_SPC_TOKENS['ATARI_0']:DEFAULT_SPC_TOKENS['ATARI_0']+4]).item()
        return a_t
    
    def take_fire_action(self,env):
        """Some games requires the agent to press 'FIRE' to start the game once loss a life."""
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        s_t, _, _, _ = env.step(1)
        return s_t

    def check_atari_env(self,env):
        """Check if is atari env and has fire action."""
        has_fire_action = False
        lives = 0
        try:
            lives = env.ale.lives()
            if env.unwrapped.get_action_meanings()[1] == 'FIRE':
                has_fire_action = True
        except AttributeError:
            pass

        return has_fire_action, lives
    
    def get_vqa_textqa_score(self):

        vocab = self.tokenizer.tokenizer.vocab()
        allowed_token_ids = []
        for i, token in enumerate(vocab):
            allowed_token_ids.append(i)
        ignore_ids = [self.tokenizer.eos_id, self.tokenizer.bos_id, self.tokenizer.unk_id, self.tokenizer.pad_id, -1, *list(DEFAULT_SPC_TOKENS.values())]        
        
        vocab_size = self.model.get_param('vocab_size')
        image_tokens_per_dim = self.model.get_param('image_tokens_per_dim')
        l_text_seq_length = self.model.get_param('l_text_seq_length')
        r_text_seq_length = self.model.get_param('r_text_seq_length')
        image_seq_length = self.model.get_param('image_seq_length')
        device = self.model.get_param('device')
        spc_id = -1
        api = ruDolphApi(self.model, self.tokenizer, self.vae, bs=10)
        
        for task in self.tasks:
            task_name = task.split('/')[5]
            dataset = self.df_val[self.df_val['task_id']==task_name].sample(n = 10)
            
            test_dataset = DatasetRetriever(
                            task_ids = dataset['task_id'].values,
                            left_texts = dataset['left_text'].values,
                            image_paths = dataset['image_path'].values,
                            right_texts = dataset['right_text'].values,
                            stage='val',
                            tokenizer=self.tokenizer,
                            model_params = self.conf.model.params)
            test_dataloader = DataLoader(
                        test_dataset,
                        batch_size=10,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False,
                        num_workers=1)
                
            for batch in tqdm(test_dataloader):
                task_id, left_text, images, right_text = batch
                left_text = batch['left_text'].to(device)
                images = batch['image'].to(device)
                right_text = batch['right_text'].to(device)
                image_tokens = self.vae.get_codebook_indices(images).to(device)    
                
            texts = api.generate_tokens(image_tokens, left_text, vocab_size, 
                                    top_k=32, top_p=0.8, temperature=1.0, template = '', 
                                    allowed_token_ids = allowed_token_ids, special_token='<RT_UNK>') 
            
            if task_name=='vqa':
                true_json = {str(ii):[{'type':'text','content':[i]}] for ii,i in enumerate(self.tokenizer.tokenizer.decode(right_text.cpu().numpy().tolist(), ignore_ids=ignore_ids))}
                pred_json = {str(ii):[{'type':'text','content':i}] for ii,i in enumerate(texts)}
                meteor = calc_meteor(true_json, pred_json)
                
            if task_name=='textqa':
                true_json = {str(ii):[{'type':'text','content':[i]}] for ii,i in enumerate(self.tokenizer.tokenizer.decode(right_text.cpu().numpy().tolist(), ignore_ids=ignore_ids))}
                pred_json = {str(ii):[{'type':'text','content':i}] for ii,i in enumerate(texts)}
                f1 = get_f1(true_json, pred_json)
                
        return meteor, f1        
                
                    
    
    def get_atari_reward(self,env):
        observation = env.reset()
        should_fire, lives = self.check_atari_env(env)
        if should_fire:
            observation = self.take_fire_action(env)
        sum_rewards = []

        observation = env.reset()
        should_fire, lives = self.check_atari_env(env)
        if should_fire:
            observation = self.take_fire_action(env)
        num_actions = env.action_space.n

        idd = 0; sum_reward = 0; frames = []
        while True:
            a_t = self.rudolph_step(observation)  
            observation, reward, done, info = env.step(a_t); idd+=1; first_step = False

            sum_reward+=reward
            # Take fire action after loss a life
            if should_fire and not done and lives != info['lives']:
                lives = info['lives']
                observation = self.take_fire_action(env)

            if done:
                sum_rewards.append(sum_reward)
                print('Done with steps: ',idd,' Sum_reward: ',sum_reward)
                return np.mean(sum_rewards)  
        return np.mean(sum_rewards)
    
    def test_model(self):
        
        if self.if_breakout: 
            BREAKOUT_REWARD = self.get_atari_reward(self.eval_env_breakout)
            print({'Breakout_reward': BREAKOUT_REWARD})
            self.log('Breakout_reward',BREAKOUT_REWARD, prog_bar=True, logger=True)
        
        if self.if_pong: 
            PONG_REWARD = self.get_atari_reward(self.eval_env_pong)
            print({'Pong_reward': PONG_REWARD})
            self.log('Pong_reward',PONG_REWARD, prog_bar=True, logger=True)
        
        VQA_SCORE, TEXTQA_SCORE = self.get_vqa_textqa_score()
        print({'Vqa_score': VQA_SCORE})
        self.log('Vqa_score',VQA_SCORE, prog_bar=True, logger=True)
        print({'Textqa_score': TEXTQA_SCORE})
        self.log('Textqa_score',TEXTQA_SCORE, prog_bar=True, logger=True)
    
    
    def model_step(self, batch, batch_idx, stage):
        
        losses = []
        
        if stage=='train':
            fbc_dataset = batch[0]
            atari_dataset1 = batch[1]
            
            left_text_atari = atari_dataset1[0]
            image_atari = atari_dataset1[1]
            right_text_atari = atari_dataset1[2]
            bs_tr = left_text_atari.shape[0]
            
            loss_atari = self.atari_loss(bs_tr, left_text_atari, image_atari, right_text_atari, self.task_weights.atari)
            self.log(f"{stage}_loss_atari", loss_atari, prog_bar=True, logger=True, batch_size=bs_tr)
            losses.append((self.task_weights.atari_loss_weight)*loss_atari)
            
            
            if batch_idx % 200 == 0:
                self.test_model()
                
        if stage=='valid':
            fbc_dataset = batch # [0] 
        

        ## text_qa
        if any(fbc_dataset[0][:,1]==16390):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16390).flatten()
            left_text_t = fbc_dataset[0][indices,:]
            right_text_t = fbc_dataset[2][indices,:]
            bs_text = left_text_t.shape[0]
            loss_text = self.get_loss(bs_text, left_text_t, None, right_text_t, self.task_weights.text)
            self.log(f"{stage}_loss_text", loss_text, prog_bar=True, logger=True, batch_size=bs_text)
            losses.append(self.task_weights.text_loss_weight*loss_text)
          
        ## math_qa
        if any(fbc_dataset[0][:,1]==16392):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16392).flatten()
            left_text_m = fbc_dataset[0][indices,:]
            right_text_m = fbc_dataset[2][indices,:]
            bs_math = left_text_m.shape[0]
            loss_math = self.get_loss(bs_math, left_text_m, None, right_text_m, self.task_weights.math)
            self.log(f"{stage}_loss_math", loss_math, prog_bar=True, logger=True, batch_size=bs_math)
            losses.append(self.task_weights.math_loss_weight*loss_math)
            
        ## captioning
        if any(fbc_dataset[0][:,1]==16396):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16396).flatten()
            left_text_c = fbc_dataset[0][indices,:]
            image_c = fbc_dataset[1][indices,:]
            right_text_c = fbc_dataset[2][indices,:]
            bs_c = left_text_c.shape[0]
            loss_с = self.get_loss(bs_c, left_text_c, image_c, right_text_c, self.task_weights.capt)
            self.log(f"{stage}_loss_с", loss_с, prog_bar=True, logger=True, batch_size=bs_c)
            losses.append(self.task_weights.capt_loss_weight*loss_с)
            
        ## generation
        if any(fbc_dataset[0][:,1]==16398):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16398).flatten()
            left_text_g = fbc_dataset[0][indices,:]
            image_g = fbc_dataset[1][indices,:]           
            bs_g = left_text_g.shape[0]
            loss_g = self.get_loss(bs_g, left_text_g, image_g, None, self.task_weights.gener)
            self.log(f"{stage}_loss_g", loss_g, prog_bar=True, logger=True, batch_size=bs_g)
            losses.append(self.task_weights.gener_loss_weight*loss_g)
            
        ## vqa
        if any(fbc_dataset[0][:,1]==16394):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16394).flatten()
            left_text_vqa = fbc_dataset[0][indices,:]
            image_vqa = fbc_dataset[1][indices,:]
            right_text_vqa = fbc_dataset[2][indices,:]
            bs_vqa = left_text_vqa.shape[0]
            loss_vqa = self.get_loss(bs_vqa, left_text_vqa, image_vqa, right_text_vqa, self.task_weights.vqa)
            self.log(f"{stage}_loss_vqa", loss_vqa, prog_bar=True, logger=True, batch_size=bs_vqa)
            losses.append(self.task_weights.vqa_loss_weight*loss_vqa)
        
        ## text recognition
        if any(fbc_dataset[0][:,1]==16399):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16399).flatten()
            left_text_tr = fbc_dataset[0][indices,:]
            image_tr = fbc_dataset[1][indices,:]
            right_text_tr = fbc_dataset[2][indices,:]
            bs_tr = left_text_tr.shape[0]
            loss_tr = self.get_loss(bs_tr, left_text_tr, image_tr, right_text_tr, self.task_weights.text_recog)
            self.log(f"{stage}_loss_tr", loss_tr, prog_bar=True, logger=True, batch_size=bs_tr)
            losses.append(self.task_weights.text_recog_loss_weight*loss_tr)
            
            
        ## UNKNOWN TASK
        if any(fbc_dataset[0][:,1]==16384):
            indices = torch.nonzero(fbc_dataset[0][:,1]==16384).flatten()
            left_text_tr = fbc_dataset[0][indices,:]
            image_tr = fbc_dataset[1][indices,:]
            right_text_tr = fbc_dataset[2][indices,:]
            bs_tr = left_text_tr.shape[0]
            loss_tr = self.get_loss(bs_tr, left_text_tr, image_tr, right_text_tr, self.task_weights.text_recog)
            self.log(f"{stage}_loss_tr", loss_tr, prog_bar=True, logger=True, batch_size=bs_tr)
            losses.append((self.task_weights.text_recog_loss_weight/4.)*loss_tr)    

        
        ## join loss
        loss = sum(losses)
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True, batch_size=self.bs)
        return {"loss": loss}
    
    
    def training_step(self, batch, batch_idx):
        print('TRAIN', len(batch),batch_idx)
        return self.model_step(batch, batch_idx, 'train')
        
    def validation_step(self, batch, batch_idx): #dataset_idx
        print('VAL', len(batch),batch_idx) #dataset_idx
        return self.model_step(batch, batch_idx, 'valid')

    def configure_optimizers(self):
        optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=self.scheduler_conf.max_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.scheduler_conf.max_lr, final_div_factor=500, 
            total_steps=self.n_training_steps 
        )
        
        scheduler = {"scheduler": lr_scheduler, "interval" : "step"}     
        return [optimizer], [scheduler] 
    

    def train_dataloader(self):
        train_dataset = DatasetRetriever(
            task_ids=self.df_train['task_id'].values,
            left_texts=self.df_train['left_text'].values,
            image_paths=self.df_train['image_path'].values,
            right_texts=self.df_train['right_text'].values,
            stage='train',
            tokenizer=self.tokenizer,
            model_params = self.conf.model.params)


        train_sampler = DistributedSamplerWrapper(
               sampler=BalanceClassSampler(labels=train_dataset.get_task_labels()),
                num_replicas=self.conf.trainer.pl_trainer.gpus,
                rank=0,
                shuffle=False)

        std_train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.conf.trainer.bs, #10
            sampler=train_sampler,
            pin_memory=False,
            drop_last=False,
            num_workers=self.NUM_WORKERS,
            collate_fn=fb_collate_fn)
  
        
        if torch.distributed.get_rank() == 0:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ INIT BREAKOUT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            game = 'Breakout'
            urls = ['s3://officecds-bucket01/datasets_v3/rl_atari_dataset/atari_{}/atari_{}_tr_{}.tar'.format(game.lower(),game.lower(),str(10000+i)[1:]) for i in range(1,500)]
            self.if_breakout = True
            
        if torch.distributed.get_rank() == 1:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ INIT PONG @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            game = 'Pong'
            urls = ['s3://officecds-bucket01/datasets_v3/rl_atari_dataset/atari_{}/atari_{}_tr_{}.tar'.format(game.lower(),game.lower(),str(10000+i)[1:]) for i in range(1,500)]
            self.if_pong = True
            
        s3_dataset_atari = wds.WebDataset(
            urls, 
            handler=wds.warn_and_continue, nodesplitter=lambda x: x).shuffle(10000)
        s3_dataset_atari = s3_dataset_atari.map(self.preprocess_s3_frame)
        s3_train_dataloader_atari = DataLoader(
            s3_dataset_atari,
            batch_size= self.conf.trainer.atari_bs, #24
            pin_memory=False,
            drop_last=False,
            num_workers=self.NUM_WORKERS,
            collate_fn=fb_collate_fn)
        ss3_train_dataloader_atari = ObjectWrapper(s3_train_dataloader_atari,30000)
        
        loaders = [std_train_dataloader, ss3_train_dataloader_atari]

        return loaders


    def val_dataloader(self):
        val_dataset = DatasetRetriever(
            task_ids=self.df_val['task_id'].values,
            left_texts=self.df_val['left_text'].values,
            image_paths=self.df_val['image_path'].values,
            right_texts=self.df_val['right_text'].values,
            stage='val',
            tokenizer=self.tokenizer,
            model_params = self.conf.model.params)


        val_sampler = DistributedSamplerWrapper(
                sampler=BalanceClassSampler(labels=val_dataset.get_task_labels()),
                num_replicas=self.conf.trainer.pl_trainer.gpus,
                rank=0,
                shuffle=False)

        std_val_dataloader = DataLoader(
            val_dataset,
            batch_size=2,
            sampler=val_sampler,
            pin_memory=False,
            drop_last=False,
            num_workers=self.NUM_WORKERS,
            collate_fn=fb_collate_fn)     
  
        loaders = [std_val_dataloader]
        
        return loaders
  

    
def freeze(
    model,
    freeze_emb=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_other=False,
):
    for name, p in model.module.named_parameters():
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model