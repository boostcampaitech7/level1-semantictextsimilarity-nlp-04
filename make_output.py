import pandas as pd
#ouput 제출 형식으로 변경(반올림)
# CSV 파일 읽기
df = pd.read_csv('/data/ephemeral/home/gayeon2/result/ensemle2.csv')

# 첫 번째 열과 세 번째 열 삭제
df = df.drop(columns=['Unnamed: 0', 'prediction'])

# 'target' 열 반올림
#df['target'] = df['target'].round(1)
# 'target' 열 소수점 첫째 자리에서 자르기
df['target'] = df['target'].apply(lambda x: int(x * 10) / 10)
# 수정된 DataFrame을 새 CSV 파일로 저장
df.to_csv('/data/ephemeral/home/gayeon2/result/ensemle_rounded(cut?????????)2.csv', index=False)
