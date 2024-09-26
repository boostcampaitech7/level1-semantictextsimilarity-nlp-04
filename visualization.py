import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Matplotlib and Seaborn are installed successfully!")

# CSV 파일 읽기
data = pd.read_csv('/data/ephemeral/home/gayeon2/result/upskyy-bge-m3-korean-V1.csv', header=None, names=['id', 'prediction', 'target'])

# 'target' 열의 데이터 확인
print(data['target'].unique())
print(data['target'].dtype)

# 문자열을 숫자로 변환
data['target'] = pd.to_numeric(data['target'], errors='coerce')

# NaN 및 무한대 값 제거
data = data.dropna(subset=['target'])
data = data[~np.isinf(data['target'])]  # 무한대 값 제거

# 구간 설정
bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]  # 필요에 따라 조정
labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']
data['target_bin'] = pd.cut(data['target'], bins=bins, labels=labels, right=False)

# 각 구간의 개수 세기
distribution = data['target_bin'].value_counts().sort_index()

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x=distribution.index, y=distribution.values, palette='viridis', hue=distribution.index, legend=False)  # hue 설정
plt.title('Target Distribution')
plt.xlabel('Target Bins')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
