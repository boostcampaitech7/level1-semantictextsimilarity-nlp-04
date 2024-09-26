from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from learner import Learner
from utils import seed_everything
from config.kykim_config import kykim_data_config, kykim_model_config, kykim_trainer_config, kykim_inference_config
from config.kr_electraV2_config import krelectraV2_data_config, krelectraV2_model_config, krelectraV2_trainer_config, krelectraV2_inference_config
from config.kr_electraV1_config import krelectraV1_data_config, krelectraV1_model_config, krelectraV1_trainer_config, krelectraV1_inference_config
from config.roberta_large_config import roberta_large_data_config, roberta_large_model_config, roberta_large_trainer_config, roberta_large_inference_config
from config.roberta_largeV7_config import roberta_largeV7_data_config,roberta_largeV7_model_config,roberta_largeV7_inference_config,roberta_largeV7_trainer_config
from config.roberta_large_nnp_config import roberta_large_nnp_data_config, roberta_large_nnp_model_config, roberta_large_nnp_trainer_config, roberta_large_nnp_inference_config
from config.bge_m3_korean_trainer_config import bge_m3_korean_data_config,bge_m3_korean_inference_config,bge_m3_korean_model_config,bge_m3_korean_trainer_config
from config.kr_electraV4_config import krelectraV4_data_config,krelectraV4_inference_config,krelectraV4_model_config,krelectraV4_trainer_config
from config.kr_electraV5_config import krelectraV5_data_config,krelectraV5_inference_config,krelectraV5_model_config,krelectraV5_trainer_config
from config.kr_electraV6_config import krelectraV6_data_config,krelectraV6_inference_config,krelectraV6_model_config,krelectraV6_trainer_config
from config.kr_electraV7_config import krelectraV7_data_config,krelectraV7_inference_config,krelectraV7_model_config,krelectraV7_trainer_config


if __name__ == '__main__':
    # # #완료(0.9243184328079224)
    # # # (1) kykim/electra-kor-base(model) + AugmentationV2(data) 학습 및 저장 및 추론
    # seed_everything(SEED=kykim_trainer_config.seed)
    # kykim_learner = Learner(model_config=kykim_model_config,
    #                         data_config=kykim_data_config,
    #                         train_config=kykim_trainer_config,
    #                         inference_config=kykim_inference_config)

    # kykim_learner.run_and_save()

    # # #완료(0.9244502782821655)
    # #(2) KR-ELECTRA-discriminator(model) + AugmentationV2(data)모델 학습 및 저장
    # seed_everything(SEED=krelectraV2_trainer_config.seed)
    # krelectraV2_learner = Learner(model_config=krelectraV2_model_config,
    #                               data_config=krelectraV2_data_config,
    #                               train_config=krelectraV2_trainer_config,
    #                               inference_config=krelectraV2_inference_config)

    # # krelectraV2_learner.run_and_save()
    # # #완료( 0.9228395223617554)
    # # #(3) KR-ELECTRA-discriminator(model) + AugmentationV1(data) 모델 학습 및 저장
    # seed_everything(SEED=krelectraV1_trainer_config.seed)
    # krelectraV1_learner = Learner(model_config=krelectraV1_model_config,
    #                               data_config=krelectraV1_data_config,
    #                               train_config=krelectraV1_trainer_config,
    #                              inference_config=krelectraV1_inference_config)

    # krelectraV1_learner.run_and_save()
    # #완료( 0.9175017476081848)
    # #(4) klue/roberta-large(model) + AugmentationV2(data) 모델 학습 및 저장
    seed_everything(SEED=roberta_large_trainer_config.seed)
    roberta_large_learner = Learner(model_config=roberta_large_model_config,
                                  data_config=roberta_large_data_config,
                                  train_config=roberta_large_trainer_config,
                                  inference_config=roberta_large_inference_config)

    roberta_large_learner.run_and_save()
    # # #이 코드 대신 change.py 사용
    # # # # roberta_large : .ckpt/ (체크포인트 디렉토리)을 .pt (.pt 모델 파일)로 변환
    # # # convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir = roberta_large_trainer_config.save_path[:-3] + 'ckpt/', 
    # # #                                            output_file = roberta_large_inference_config.model_path)

    # #완료( 0.9101384878158569
    # # (5) klue/roberta-large(model) + AugmentationV3(data) 모델 학습 및 저장
    # seed_everything(SEED=roberta_large_nnp_trainer_config.seed)
    # roberta_large_nnp_learner = Learner(model_config=roberta_large_nnp_model_config,
    #                               data_config=roberta_large_nnp_data_config,
    #                               train_config=roberta_large_nnp_trainer_config,
    #                               inference_config=roberta_large_nnp_inference_config)

    # roberta_large_nnp_learner.run_and_save()
    #이 코드 대신 change.py 사용
    # roberta_large : .ckpt/ (체크포인트 디렉토리)을 .pt (.pt 모델 파일)로 변환
    # convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir = roberta_large_nnp_trainer_config.save_path[:-3] + 'ckpt/', 
    #                                            output_file = roberta_large_nnp_inference_config.model_path)

     # (6) upskyy/bge-m3-korean(model) + Augmentation(data) 모델 학습 및 저장
    # seed_everything(SEED=bge_m3_korean_trainer_config.seed)  # SEED 설정
    # bge_m3_korean_learner = Learner(model_config=bge_m3_korean_model_config,
    #                                 data_config=bge_m3_korean_data_config,
    #                                 train_config=bge_m3_korean_trainer_config,
    #                                 inference_config=bge_m3_korean_inference_config)

    # # #완료( 0.9213051795959473)
    # #(7) KR-ELECTRA-discriminator(model) + AugmentationV4(data)모델 학습 및 저장
    # seed_everything(SEED=krelectraV4_trainer_config.seed)
    # krelectraV4_learner = Learner(model_config=krelectraV4_model_config,
    #                               data_config=krelectraV4_data_config,
    #                               train_config=krelectraV4_trainer_config,
    #                               inference_config=krelectraV4_inference_config)
    # krelectraV4_learner.run_and_save()

     #(8) KR-ELECTRA-discriminator(model) + AugmentationV5(data)모델 학습 및 저장
    # seed_everything(SEED=krelectraV5_trainer_config.seed)
    # krelectraV5_learner = Learner(model_config=krelectraV5_model_config,
    #                               data_config=krelectraV5_data_config,
    #                               train_config=krelectraV5_trainer_config,
    #                               inference_config=krelectraV5_inference_config)
    # krelectraV5_learner.run_and_save()
    #(9) KR-ELECTRA-discriminator(model) + AugmentationV6(data)모델 학습 및 저장
    # seed_everything(SEED=krelectraV6_trainer_config.seed)
    # krelectraV6_learner = Learner(model_config=krelectraV6_model_config,
    #                               data_config=krelectraV6_data_config,
    #                               train_config=krelectraV6_trainer_config,
    #                               inference_config=krelectraV6_inference_config)
    # krelectraV6_learner.run_and_save()
    #0.9218734502792358
    #(10) KR-ELECTRA-discriminator(model) + AugmentationV7(data)모델 학습 및 저장
    # seed_everything(SEED=krelectraV7_trainer_config.seed)
    # krelectraV7_learner = Learner(model_config=krelectraV7_model_config,
    #                               data_config=krelectraV7_data_config,
    #                               train_config=krelectraV7_trainer_config,
    #                               inference_config=krelectraV7_inference_config)
    # krelectraV7_learner.run_and_save()
    # #(11) klue/roberta-large(model) + AugmentationV7(data) 모델 학습 및 저장
    # seed_everything(SEED=roberta_largeV7_trainer_config.seed)
    # roberta_largeV7_learner = Learner(model_config=roberta_largeV7_model_config,
    #                               data_config=roberta_largeV7_data_config,
    #                               train_config=roberta_largeV7_trainer_config,
    #                               inference_config=roberta_largeV7_inference_config)
    # roberta_largeV7_learner.run_and_save()