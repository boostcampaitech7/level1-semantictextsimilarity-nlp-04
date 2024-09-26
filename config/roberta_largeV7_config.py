import torch
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig


"""
klue/roberta_large(모델) + AugmentationV7(데이터) Config 
"""

roberta_largeV7_data_config = DataConfig(dataloader=BaseDataloader,
                                model_name='klue/roberta-large',
                                dataset=Dataset,
                                batch_size=16,
                                shuffle=True,
                                train_path='/data/ephemeral/home/gayeon2/data/train_augmentV7.csv',
                                dev_path='/data/ephemeral/home/gayeon2/data/dev_spellcheck.csv',
                                test_path='/data/ephemeral/home/gayeon2/data/dev_spellcheck.csv',
                                predict_path='/data/ephemeral/home/gayeon2/data/test_spellcheck.csv'
                                )

roberta_largeV7_model_config = ModelConfig(model=BaseModel,
                                 model_name='klue/roberta-large',
                                 loss_func=LossfunctionWrap(loss=torch.nn.L1Loss),
                                 optimizer=OptimizerWrap(optimizer=torch.optim.AdamW,
                                                         hyperparmeter={'lr':1e-5}),
                                 scheduler=SchedulerWrap(scheduler=get_linear_schedule_with_warmup, 
                                                         hyperparmeter={'num_warmup_steps':0, 'num_training_steps':33488//(16*30)})  # (train_data_length// (batch_size * max_epoch)
                                )

roberta_largeV7_trainer_config = TrainerConfig(seed=0,
                                     epoch=5, # 5,
                                     save_path='/data/ephemeral/home/gayeon2/model/snunlp-KR-ELECTRA-discriminator-V7.pt',
                                     precision=32, # 32
                                     callbacks= None,
                                     strategy='auto') 


roberta_largeV7_inference_config = InferenceConfig(predict_path='/data/ephemeral/home/gayeon2/result/klue-roberta-largeV7.csv',
                                                    model_path = '/data/ephemeral/home/gayeon2/model/klue-roberta-largeV7.pt',  # bin 파일을 설정한 경로에 위치시켜줘야 함.
                                                    submission_path='/data/ephemeral/home/gayeon2/data/sample_submission.csv',
                                                    ensemble_weight =  0.9175
                                                    )