import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB_API_KEY'] = '77b33f530c461728a2fb12eeb694e04811d2d960'

sys.path.insert(0, '../COMMON_PATH/data_for_inference')

from rudalle import get_tokenizer, get_vae
from src.inference.utils import create_dataset
from src.inference.dataloader import DatasetRetriever
from src.inference.inference import run_inference
from torch.utils.data import DataLoader
import torch
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
import pandas as pd
import argparse 
import json
import os
import time
import random
import numpy as np
import shutil
import copy
from pytorch_lightning.loggers import WandbLogger
from catalyst.data import BalanceClassSampler, DistributedSamplerWrapper
from pytorch_lightning.callbacks import LearningRateMonitor

from torch.utils.data import DataLoader
from rudalle import get_tokenizer, get_vae

from src.rudolph.model import get_rudolph_model, ruDolphModel, FP16Module
from src.rudolph import utils

from src.train.dataloader_custom import DatasetRetriever, fb_collate_fn  ########### dataloader_custom
from src.train.trainer_custom import RudolphLightning                           ############## trainer
from src.train.utils import create_dataset
from src.train.checkpoint_model_custom import ModelCheckpoint
import pytorch_lightning as pl





def main():

    #torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/fusion_train.yaml',type=str, help='config path')
    args = parser.parse_args(''.split())
    conf_origin = OmegaConf.load(args.config)
    conf_origin.trainer.pl_trainer.gpus = 4
    conf_origin.trainer.pl_trainer.strategy = 'ddp'
    del conf_origin.trainer.pl_trainer.progress_bar_refresh_rate
    
    wandb_logger = WandbLogger(project=conf_origin.trainer.logger.model_name)
    
    conf = copy.deepcopy(conf_origin)
    for task in list(conf.data.keys()):
        for k,v in conf.data[task].items():
            conf.data[task][k] = conf.data[task][k][:13] + 'Aleksei_RL' + conf.data[task][k][21:]
  
    conf.model.rudolph.name = '350M' # 1.3B 350M
    conf.model.rudolph_weight = '/home/jovyan/Aleksei_RL/inference/model/light_last.pt'    
    conf.model.params.num_layers = 24
    conf.model.params.hidden_size = 1024
    conf.model.params.num_attention_heads = 16
    conf.model.params.l_text_seq_length = 64
    conf.model.params.image_tokens_per_dim = 16
    conf.model.params.r_text_seq_length = 64
    conf.model.params.kernel_size = 7
    conf.model.params.last_kernel_size = 9
    conf.model.params.vocab_size = 16448
    conf.model.params.text_special_tokens = 384
    conf.model.params.image_special_tokens = 384  


    #conf.model.rudolph_weight = '/home/jovyan/Aleksei_RL/inference/model/last.pt'     #light_last

    vae = get_vae(**conf.model.vae)
    model = get_rudolph_model(**conf.model.rudolph, **conf.model.params)
    checkpoint = torch.load(conf.model.rudolph_weight, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(**conf.trainer.checkpoints)
    
    trainer = pl.Trainer(
              logger=wandb_logger,
              callbacks=[lr_monitor, checkpoint_callback],
              **conf.trainer.pl_trainer)
    
    total_training_steps = 50000 * conf.trainer.pl_trainer.max_epochs #* 4 #int(len(train_dataloader) #/  conf.trainer.pl_trainer.accumulate_grad_batches)

    rudolph_light = RudolphLightning(vae=vae, 
                                     model=model, 
                                     n_training_steps=total_training_steps, 
                                     conf=conf,
                                     task_weights=conf.trainer.task_weights,
                                     model_params=conf.model.params,
                                     model_freeze = conf.model.freeze,
                                     scheduler_conf=conf.trainer.scheduler,
                                     bs = conf.trainer.bs)
    
    trainer.fit(rudolph_light)  # , train_dataloader, val_dataloader
    



if __name__ == '__main__':
    main()
