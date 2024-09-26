import pandas as pd
import torch
import wandb
import pytorch_lightning as pl
from wrappers.config_wrapper import TrainerConfig, DataConfig, ModelConfig, InferenceConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

class Learner(pl.LightningModule):
    """
    모델 훈련 및 저장을 관리하는 클래스.
    """
    def __init__(self, model_config: ModelConfig,data_config: DataConfig, 
                 train_config: TrainerConfig, inference_config: InferenceConfig):
        
        super().__init__()
        self.DATACONFIG = data_config
        self.MODELCONFIG = model_config
        self.TRAINCONFIG = train_config
        self.INFERCONFIG = inference_config

        # Dataloader 객체 선언 및 인스턴스 생성.
        dataloader_object = self.DATACONFIG.dataloader
        self.dataloader = dataloader_object(model_name=self.DATACONFIG.model_name,
                                            dataset=self.DATACONFIG.dataset,
                                            batch_size=self.DATACONFIG.batch_size,
                                            shuffle=self.DATACONFIG.shuffle,
                                            train_path=self.DATACONFIG.train_path,
                                            dev_path=self.DATACONFIG.dev_path,
                                            test_path=self.DATACONFIG.test_path,
                                            predict_path=self.DATACONFIG.predict_path)

        # Model 객체 선언 및 인스턴스 생성
        self.model_object = self.MODELCONFIG.model
        self.model = self.model_object(model_name=self.MODELCONFIG.model_name,
                                       loss_func=self.MODELCONFIG.loss_func,
                                       optimizer=self.MODELCONFIG.optimizer,
                                       scheduler=self.MODELCONFIG.scheduler)
        
        # EarlyStopping 설정
        early_stop_callback = EarlyStopping(
              monitor='val_pearson',  # val_pearson을 모니터
            min_delta=0.001,
            patience=2,
            verbose=True,
            mode='max'  # val_pearson이 최대화되는 방향
        )



        # WandB Logger 설정
        wandb.init(project='01', name='01')
        logger = WandbLogger(log_model='all')  # 모델 로그를 wandb에 기록
        
        # Trainer 인스턴스 생성
        self.trainer = pl.Trainer(accelerator="gpu", 
                                   devices=1,
                                   max_epochs=self.TRAINCONFIG.epoch,
                                   precision=self.TRAINCONFIG.precision,
                                   #callbacks=[early_stop_callback] + self.TRAINCONFIG.callbacks,
                                   callbacks= self.TRAINCONFIG.callbacks,
                                   logger=logger,
                                   log_every_n_steps=1)

    def configure_optimizers(self):
        return self.MODELCONFIG.optimizer

    def run_and_save(self):
        # Train & Test with dev set
        self.trainer.fit(model=self.model, datamodule=self.dataloader)
        self.trainer.test(model=self.model, datamodule=self.dataloader)
        
        # 학습이 완료된 모델을 저장.
        if self.MODELCONFIG.model_name != 'klue/roberta-large':
            torch.save(self.model.state_dict(), self.TRAINCONFIG.save_path)  # 모델의 state_dict 저장

    def predict(self):
        # 모델 로드 및 예측
        model = self.MODELCONFIG.model(
            model_name=self.MODELCONFIG.model_name,
            loss_func=self.MODELCONFIG.loss_func,
            optimizer=self.MODELCONFIG.optimizer,
            scheduler=self.MODELCONFIG.scheduler
        )
        model.load_state_dict(torch.load(self.INFERCONFIG.model_path))  # .pt 파일 로드

        # 예측 수행
        predictions = self.trainer.predict(model=model, datamodule=self.dataloader)
        predictions = list(max(0, min(round(float(i), 1), 5)) for i in torch.cat(predictions))
        
        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv(self.INFERCONFIG.submission_path)
        output['target'] = predictions
        output.to_csv(self.INFERCONFIG.predict_path, index=False)


