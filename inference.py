import argparse
import yaml
import pandas as pd
import os
from tqdm.auto import tqdm
import json

import torch
#import transformers
#import pandas as pd

import pytorch_lightning as pl
#import wandb
##############################
from utils import data_pipeline, utils



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inference', type=str, help="Choose between 'inference' or 'ensemble' mode")
    args = parser.parse_args()

    assert args.mode in ['inference', 'ensemble'], "--mode should be 'inference' or 'ensemble'"

    # baseline_config 설정 불러오기
    with open('./config/config.yaml', encoding='utf-8') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # inference
    if args.mode == 'inference':
        # 저장된 폴더 이름
        exp_name = CFG['inference']['model_file_name']['name_1']

        # dataloader / model 설정
        dataloader = data_pipeline.Dataloader(CFG)

        model = torch.load(f'./experiments/{exp_name}/model.pt')

        # trainer 인스턴스 생성
        trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=CFG['train']['max_epoch'], log_every_n_steps=1)

        # Inference part
        predictions = trainer.predict(model=model, datamodule=dataloader)
        ## datamodule에서 predict_dataloader 호출

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv(CFG['data']['submission_path'])
        output['target'] = predictions
        output.to_csv(f'./output/output_({exp_name}).csv', index=False)
    
    # ensemble
    else:
        CFG = CFG['inference']

        result_path_list, score_list = utils.score_path(CFG)
        
        postprocessing_list = CFG['postprocessing_list']

        utils.ensemble(result_path_list = result_path_list, 
        score_list = score_list, 
        postprocessing_list = postprocessing_list,
        save_path = './output/ensemble/' + CFG['save_name'] + '.csv')