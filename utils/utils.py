import os
from datetime import datetime
from typing import List

import wandb
import pandas as pd
import re

# 모델 저장 폴더
def create_experiment_folder(CFG, base_path="./experiments"):
    # 현재 시간 기록
    current_time = datetime.now().strftime("%m%d_%H%M")

    # admin 값을 가져와서 폴더 이름에 추가
    # user_name = CFG["user_name"]
    model_name = CFG["model"]["model_name"]

    # 월일_시간분_user_name 형식으로 폴더 이름 생성
    experiment_folder_name = f"{model_name}_{current_time}"
    experiment_folder_name = re.sub('/', '_', experiment_folder_name)

    # experiments 경로에 해당 폴더 생성
    experiment_path = os.path.join(base_path, experiment_folder_name)
    os.makedirs(experiment_path, exist_ok=True)

    return experiment_path


# wandB 초기화 및 저장 폴더 생성
def wandb_init(CFG, base_path="./wandb"):
    # 현재 시각
    current_time = datetime.now().strftime("%m%d_%H%M")

    # user_name = CFG['user_name']
    wandb_name = f"{CFG['model']['model_name']}_{current_time}"

    # wandb_folder_name = re.sub('/', '_', wandb_folder_name)

    # # wandB 저장 폴더 생성
    # wandb_path = os.path.join(base_path, wandb_folder_name)
    # os.makedirs(wandb_path, exist_ok=True)

    # # wandB 저장 폴더 설정
    # os.environ["WANDB_DIR"] = wandb_path

    # WandB 초기화
    wandb.init(project="wandb_logs", name=wandb_name, config={
        "learning_rate" : CFG["train"]["learning_rate"],
        "batch_size" : CFG["train"]["batch_size"],
        "epoch" : CFG["train"]["max_epoch"]
    })  # 프로젝트 이름과 실험 이름 설정


# 후처리
def postprocessing(df:pd.DataFrame) -> pd.DataFrame:
    """
    후처리 함수 : 예측값에 단순 -0.1 뺄셈
    """
    df['target'] = df['target'].apply(lambda x : max(0, x-0.1))
    return df

# score_list, path_list
def score_path(CFG):
    CFG = CFG["inference"]

    path_list = []
    for i in range(1, len(CFG['model_path'])+1):
        path_list.append(CFG['model_path'][f'path_{i}'])
    
    score_list = []
    for i in range(1, len(CFG['model_weight'])+1):
        score_list.append(CFG['model_weight'][f'weight_{i}'])

    return path_list, score_list

# 앙상블
def ensemble(result_path_list:List[pd.DataFrame], score_list:List[float], 
                postprocessing_list:List[bool], save_path='./output/ensemble/ensemble.csv') -> pd.DataFrame:
    """
    점수 가중 평균 Ensemble
    """
    df_submission, weight_sum = None, 0 
    for i, (path, weight, pp) in enumerate(zip(result_path_list, score_list, postprocessing_list)):
        df_now = pd.read_csv(path)
        # 후처리를 진행
        if pp:
            df_now = postprocessing(df_now)
        
        # i == 0에서 제출 파일 생성 / 점수 가중 합
        if i == 0:
            df_submission = pd.read_csv(path)
            df_submission['target'] = weight * df_now['target']
        else:
            df_submission['target'] += weight * df_now['target']
        
        weight_sum += weight
    
    # 점수 가중 평균
    df_submission['target'] /= weight_sum
    df_submission.to_csv(save_path)