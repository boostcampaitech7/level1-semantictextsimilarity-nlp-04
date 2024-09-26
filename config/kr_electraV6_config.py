import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig

"""
snunlp/KR-ELECTRA-discriminator(모델) + AugmentationV6(데이터) Config 
"""

krelectraV6_data_config = DataConfig(dataloader=BaseDataloader,
                                model_name='snunlp/KR-ELECTRA-discriminator',
                                dataset=Dataset,
                                batch_size=16,#32
                                shuffle=True,
                                train_path='/data/ephemeral/home/gayeon2/data/train_augmentV6.csv',
                                dev_path='/data/ephemeral/home/gayeon2/data/dev_spellcheck.csv',
                                test_path='/data/ephemeral/home/gayeon2/data/dev_spellcheck.csv',
                                predict_path='/data/ephemeral/home/gayeon2/data/test_spellcheck.csv'
                                )

krelectraV6_model_config = ModelConfig(model=BaseModel,
                                 model_name='snunlp/KR-ELECTRA-discriminator',
                                 loss_func=LossfunctionWrap(loss=torch.nn.L1Loss),
                                 optimizer=OptimizerWrap(optimizer=torch.optim.Adam,
                                                         hyperparmeter={'lr':2*1e-5}),
                                 scheduler=SchedulerWrap(scheduler=CosineAnnealingWarmRestarts, 
                                                         hyperparmeter={'T_0':7, 'T_mult':1, 'eta_min':7*1e-6})
                                )

krelectraV6_trainer_config = TrainerConfig(seed=42,
                                     epoch=4, # 11,
                                     save_path='/data/ephemeral/home/gayeon2/model/snunlp-KR-ELECTRA-discriminator-V6(10).pt',
                                     precision=32, # 32
                                     callbacks= None,
                                     strategy='auto') 


krelectraV6_inference_config = InferenceConfig(predict_path='/data/ephemeral/home/gayeon2/result/snunlp-KR-ELECTRA-discriminator-V6.csv',
                                                model_path = '/data/ephemeral/home/gayeon2/model/snunlp-KR-ELECTRA-discriminator-V6.pt',
                                                submission_path='/data/ephemeral/home/gayeon2/data/sample_submission.csv',
                                                ensemble_weight = 0.8869
                                                )