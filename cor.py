import pandas as pd
from scipy.stats import pearsonr

# CSV 파일 경로 목록
file_paths = [
    # '/data/ephemeral/home/gayeon2/result/klue-roberta-large-nnp.csv',
    # '/data/ephemeral/home/gayeon2/result/klue-roberta-large.csv',
    # '/data/ephemeral/home/gayeon2/result/kykim-electra-kor-base.csv',
    # '/data/ephemeral/home/gayeon2/result/snunlp-KR-ELECTRA-discriminator-V1.csv',
    # '/data/ephemeral/home/gayeon2/result/snunlp-KR-ELECTRA-discriminator-V2.csv',
    # '/data/ephemeral/home/gayeon2/result/upskyy-bge-m3-korean-V1.csv'
    '/data/ephemeral/home/gayeon2/result/ensemle.csv',
    '/data/ephemeral/home/gayeon2/result/ensemle2.csv'
]

# 예측 결과를 담을 데이터프레임 생성
predictions = {}

# 각 CSV 파일에서 예측 결과를 읽어오기
for path in file_paths:
    df = pd.read_csv(path)
    # 예측 결과가 있는 열 이름을 확인하고, 예를 들어 'prediction'이라고 가정
    predictions[path] = df['target']

# 피어슨 상관계수 계산을 위한 빈 데이터프레임 생성
correlation_matrix = pd.DataFrame(index=file_paths, columns=file_paths)

# 피어슨 상관계수 계산
for i in range(len(file_paths)):
    for j in range(len(file_paths)):
        if i == j:
            correlation_matrix.iloc[i, j] = 1.0  # 자기 자신과의 상관계수는 1
        else:
            corr, _ = pearsonr(predictions[file_paths[i]], predictions[file_paths[j]])
            correlation_matrix.iloc[i, j] = corr

# 결과를 CSV 파일로 저장
correlation_matrix.to_csv('/data/ephemeral/home/gayeon2/result/correlation_matrix.csv')
# 결과 출력
print(correlation_matrix)
