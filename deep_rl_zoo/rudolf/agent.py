import deep_rl_zoo.types as types_lib
import torch
import numpy as np
import torch
from rudolph.model.utils import get_attention_mask
from PIL import Image

class RudolphAgent(types_lib.Agent):
  def __init__(self, model, checkpoint_path, api, args, spc_tokens):
    
    serialize_to = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=serialize_to)
    model.load_state_dict(checkpoint)
    self.model = model
    self.model.eval()
    self.spc_tokens = spc_tokens
    self.api = api
    self.args = args
    self.actions = 18
    self.agent_name='Rudolph'

  def step(self, timestep: types_lib.TimeStep) -> types_lib.Action:
    """Этот метод также нужен для совместимости"""
    #self._step_t += 1

    a_t = self.act(timestep)

    return a_t

  def reset(self):
    """Этот метод нужен лишь для совместимости и не используется"""
    pass # nothing to do

  def act(self, timestep: types_lib.TimeStep) -> types_lib.Action:
    '''Возвращает действие по наблюдению'''
    ids, mask = self.observation2token(timestep.observation)

    with torch.no_grad():
      logits = self.model(ids, mask)
    # Ниже код, в котором выполняется семплирование из выдаваемого моделью распределения
    distribution = torch.softmax(logits[0][:, -1, self.spc_tokens['ATARI_0']:self.spc_tokens['ATARI_0']+self.actions], 1)
    a_t = torch.multinomial(distribution, 1).item()
    return a_t

  def observation2token(self, observation):
    '''По наблюдению (батчу картинок) составляется последовательность токенов, используемая для предсказания'''
    left_special_token = '<LT_RLA>'
    right_special_token = '<RT_RLA>'

    lt = torch.zeros(self.args.l_text_seq_length,dtype=torch.int32)
    lt[0] = 2
    lt[1] = self.spc_tokens[left_special_token]
    lt[2] = 3
    rt = torch.zeros(2, dtype=torch.int32)
    rt[0] = 2
    rt[1] = self.spc_tokens[right_special_token]

    img = np.vstack((np.hstack((observation[0],observation[1])),np.hstack((observation[2],observation[3]))))

    img = Image.fromarray(img)
    img = self.api.image_transform(img)
    img = img.unsqueeze(0).to(self.api.device)
    image_input_ids_text = self.api.vae.get_codebook_indices(img, disable_gumbel_softmax=True)[0]
    
    image_seq_length = self.args.image_tokens_per_dim ** 2
    total_seq_length = self.args.l_text_seq_length + image_seq_length + 2

    attention_mask_text = get_attention_mask(1, self.args.l_text_seq_length,
                                          self.args.image_tokens_per_dim, 
                                          2, self.args.device)

    input_ids_text = torch.cat((lt.to(self.args.device).unsqueeze(0), image_input_ids_text.to(self.args.device).unsqueeze(0), rt.to(self.args.device).unsqueeze(0)), dim=1)

    return input_ids_text, attention_mask_text

  @property
  def statistics(self):
    """Этот метод нужен только для совместимости"""
    pass
