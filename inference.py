import argparse
from learner import Learner
from utils import seed_everything
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from config.kykim_config import kykim_data_config, kykim_model_config, kykim_trainer_config, kykim_inference_config
from config.kr_electraV2_config import krelectraV2_data_config, krelectraV2_model_config, krelectraV2_trainer_config, krelectraV2_inference_config
from config.kr_electraV1_config import krelectraV1_data_config, krelectraV1_model_config, krelectraV1_trainer_config, krelectraV1_inference_config
from config.roberta_large_config import roberta_large_data_config, roberta_large_model_config, roberta_large_trainer_config, roberta_large_inference_config
from config.roberta_large_nnp_config import roberta_large_nnp_data_config, roberta_large_nnp_model_config, roberta_large_nnp_trainer_config, roberta_large_nnp_inference_config
from config.bge_m3_korean_trainer_config import bge_m3_korean_data_config,bge_m3_korean_inference_config,bge_m3_korean_model_config,bge_m3_korean_trainer_config
from config.kr_electraV4_config import krelectraV4_data_config,krelectraV4_inference_config,krelectraV4_model_config,krelectraV4_trainer_config
from config.kr_electraV5_config import krelectraV5_data_config,krelectraV5_inference_config,krelectraV5_model_config,krelectraV5_trainer_config
from config.kr_electraV6_config import krelectraV6_data_config,krelectraV6_inference_config,krelectraV6_model_config,krelectraV6_trainer_config
from utils import ensemble



if __name__ == '__main__':

    # 터미널 실행시 --inference=True (default)는 학습한 모델을 입력받아 추론을, False는 앙상블을 진행합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='inference', type=str)
    args = parser.parse_args()

    assert args.mode in ['inference', 'ensemble'], "--mode should be 'inference' or 'ensemble'"
    # 추론
    if args.mode == 'inference':
        
        # # (1) kykim/electra-kor-base(model) + AugmentationV2(data)모델 추론
        # seed_everything(SEED=kykim_trainer_config.seed)
        # kykim_learner = Learner(model_config=kykim_model_config,
        #                         data_config=kykim_data_config,
        #                         train_config=kykim_trainer_config,
        #                         inference_config=kykim_inference_config)

        # kykim_learner.predict()

        # (2) KR-ELECTRA-discriminator(model) + AugmentationV2(data)모델 추론
        # seed_everything(SEED=krelectraV2_trainer_config.seed)
        # krelectraV2_learner = Learner(model_config=krelectraV2_model_config,
        #                             data_config=krelectraV2_data_config,
        #                             train_config=krelectraV2_trainer_config,
        #                             inference_config=krelectraV2_inference_config)

        # krelectraV2_learner.predict()

        # # # (3) KR-ELECTRA-discriminator(model) + AugmentationV1(data)모델 추론
        # seed_everything(SEED=krelectraV1_trainer_config.seed)
        # krelectraV1_learner = Learner(model_config=krelectraV1_model_config,
        #                             data_config=krelectraV1_data_config,
        #                             train_config=krelectraV1_trainer_config,
        #                             inference_config=krelectraV1_inference_config)

        # krelectraV1_learner.predict()

        # # #(4) klue/roberta-large(model) + AugmentationV2(data)모델 추론
        # seed_everything(SEED=roberta_large_trainer_config.seed)
        # roberta_large_learner = Learner(model_config=roberta_large_model_config,
        #                             data_config=roberta_large_data_config,
        #                             train_config=roberta_large_trainer_config,
        #                             inference_config=roberta_large_inference_config)

        # roberta_large_learner.predict()

        # # (5) klue/roberta-large(model) + AugmentationV2 + nnp(data)모델 추론
        # seed_everything(SEED=roberta_large_nnp_trainer_config.seed)
        # roberta_large_nnp_learner = Learner(model_config=roberta_large_nnp_model_config,
        #                             data_config=roberta_large_nnp_data_config,
        #                             train_config=roberta_large_nnp_trainer_config,
        #                             inference_config=roberta_large_nnp_inference_config)

        # roberta_large_nnp_learner.predict()
        # # (6) upskyy/bge-m3-korean(model) + Augmentation(data) 모델 학습 및 저장
        # seed_everything(SEED=bge_m3_korean_trainer_config.seed)  # SEED 설정
        # bge_m3_korean_learner = Learner(model_config=bge_m3_korean_model_config,
        #                                 data_config=bge_m3_korean_data_config,
        #                                 train_config=bge_m3_korean_trainer_config,
        #                                 inference_config=bge_m3_korean_inference_config)

        # bge_m3_korean_learner.predict()
        
       # (7) KR-ELECTRA-discriminator(model) + AugmentationV4(data)모델 추론
        seed_everything(SEED=krelectraV4_trainer_config.seed)
        krelectraV4_learner = Learner(model_config=krelectraV4_model_config,
                                    data_config=krelectraV4_data_config,
                                    train_config=krelectraV4_trainer_config,
                                    inference_config=krelectraV4_inference_config)

        krelectraV4_learner.predict()
        #(0.8869208693504333)
        # (8) KR-ELECTRA-discriminator(model) + AugmentationV5(data)모델 추론
        # seed_everything(SEED=krelectraV5_trainer_config.seed)
        # krelectraV5_learner = Learner(model_config=krelectraV5_model_config,
        #                             data_config=krelectraV5_data_config,
        #                             train_config=krelectraV5_trainer_config,
        #                             inference_config=krelectraV5_inference_config)

        # krelectraV6_learner.predict()
        seed_everything(SEED=krelectraV6_trainer_config.seed)
        krelectraV6_learner = Learner(model_config=krelectraV6_model_config,
                                    data_config=krelectraV6_data_config,
                                    train_config=krelectraV6_trainer_config,
                                    inference_config=krelectraV6_inference_config)

        krelectraV6_learner.predict()


    # Ensemble
    elif args.mode == 'ensemble':
        # result_path_list = [kykim_inference_config.predict_path, 
        #                     krelectraV2_inference_config.predict_path, 
        #                     krelectraV1_inference_config.predict_path,
        #                     roberta_large_inference_config.predict_path,
        #                     roberta_large_nnp_inference_config.predict_path]
        
        # score_list = [kykim_inference_config.ensemble_weight, 
        #               krelectraV2_inference_config.ensemble_weight, 
        #               krelectraV1_inference_config.ensemble_weight,
        #               roberta_large_inference_config.ensemble_weight,
        #               roberta_large_nnp_inference_config.ensemble_weight]
        
        # postprocessing_list = [True, False, False, False, False]
        result_path_list = [kykim_inference_config.predict_path, 
                            krelectraV4_inference_config.predict_path ,
                            krelectraV2_inference_config.predict_path
                            
        ]
        
        score_list = [kykim_inference_config.ensemble_weight, 
                      krelectraV4_inference_config.ensemble_weight,
                       krelectraV2_inference_config.ensemble_weight
                      ]
        
        postprocessing_list = [ True,False, False]
        # 모델 앙상블
        ensemble(result_path_list = result_path_list, 
                 score_list = score_list, 
                 postprocessing_list = postprocessing_list,
                 save_path = '/data/ephemeral/home/gayeon2/result/ensemle(snunlp-v4,snunlp-v2,kykim).csv')
        
        
