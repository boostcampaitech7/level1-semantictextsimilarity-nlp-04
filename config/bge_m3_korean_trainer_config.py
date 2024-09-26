import torch
from dataloader import BaseDataloader
from dataset import Dataset
from model import BaseModel
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig

"""
upskyybge-m3-korean(모델) + AugmentationV1(데이터) Config 
"""

bge_m3_korean_data_config = DataConfig(dataloader=BaseDataloader,
                                model_name='upskyy/bge-m3-korean',
                                dataset=Dataset,
                                batch_size=4,
                                shuffle=True,
                                train_path='/data/ephemeral/home/gayeon2/data/train_augmentV2.csv',  # 절대 경로로 변경
                                dev_path='/data/ephemeral/home/gayeon2/data/dev.csv',                  # 절대 경로로 변경
                                test_path='/data/ephemeral/home/gayeon2/data/dev.csv',                 # 절대 경로로 변경
                                predict_path='/data/ephemeral/home/gayeon2/data/test.csv'              # 절대 경로로 변경
                                )

bge_m3_korean_model_config = ModelConfig(model=BaseModel,
                                 model_name='upskyy/bge-m3-korean',
                                 loss_func=LossfunctionWrap(loss=torch.nn.L1Loss),
                                 optimizer=OptimizerWrap(optimizer=torch.optim.Adam,
                                                         hyperparmeter={'lr':1e-5}),
                                 scheduler=None
                                )
#epoch=3,batch=4, 0.7002584338188171,v1
#epoch=1,batch=4, 0.815945565700531,v1
bge_m3_korean_trainer_config = TrainerConfig(seed=0,
                                     epoch=1,  # 15,
                                     save_path='/data/ephemeral/home/gayeon2/model/upskyy-bge-m3-korean.pt',  # 절대 경로로 변경
                                     precision=32,  # precision 32 맞는지 확인
                                     callbacks=None,
                                     strategy='auto')  

bge_m3_korean_inference_config = InferenceConfig(predict_path='/data/ephemeral/home/gayeon2/result/upskyy-bge-m3-korean-V1.csv',  # 절대 경로로 변경
                                         model_path='/data/ephemeral/home/gayeon2/model/upskyy-bge-m3-korean.pt',  # 절대 경로로 변경
                                         submission_path='/data/ephemeral/home/gayeon2/data/sample_submission.csv',  # 절대 경로로 변경
                                         ensemble_weight=0.9166
                                         )
