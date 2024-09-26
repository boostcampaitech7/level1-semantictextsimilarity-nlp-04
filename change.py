import torch
from model import BaseModel  # BaseModel을 import 하세요
from wrappers.train_wrapper import LossfunctionWrap, OptimizerWrap, SchedulerWrap
##ckpt 파일 pt로 변경하는 코드
# 체크포인트 파일 경로
ckpt_path = '/data/ephemeral/home/gayeon2/model/klue-roberta-large.ckpt'
# 변환할 .pt 파일 경로
pt_path = '/data/ephemeral/home/gayeon2/model/klue-roberta-large.pt'

# 필요한 인수들 정의
model_name = 'klue/roberta-large'
loss_func = LossfunctionWrap(loss=torch.nn.L1Loss)  # 예시로 L1Loss 사용
optimizer = OptimizerWrap(optimizer=torch.optim.AdamW, hyperparmeter={'lr': 1e-5})
scheduler = SchedulerWrap(scheduler=None, hyperparmeter={})  # 스케줄러가 필요 없다면 None으로

# 모델 인스턴스 생성
model = BaseModel(model_name=model_name, loss_func=loss_func, optimizer=optimizer, scheduler=scheduler)

# 체크포인트 로드
checkpoint = torch.load(ckpt_path)

# 모델의 state_dict 로드
model.load_state_dict(checkpoint['state_dict'])

# .pt 파일로 저장
torch.save(model.state_dict(), pt_path)

print(f"모델이 {pt_path}로 변환되었습니다.")
