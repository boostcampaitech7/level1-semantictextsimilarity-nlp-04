# Idea
### (A,B) 간의 유사도는 (B,A) 간의 유사도와 같을 것이다.  
1. 학습 시 (A,B)를 그대로 넣는 대신  (A,B) 또는 (B,A) 형태로 랜덤하게 섞어서 넣어준다.  
2. (A,B)를 복제하고 순서를 바꾼 (B,A) 데이터셋을 추가한다.  

### 쌩뚱맞은 문장과 반대 의미의 문장은 각각 유사도를 몇점 줄 것인가?
- 둘 다 0점을 주게 되어있다면 0점에 가까운 데이터들의 비율을 엉뚱한 문장 50%, 반대 의미의 문장 50%로 맞춰야 하지 않을까?  

# Benchmark Dataset

## Simialrity and Paraphrase Task
문장 페어 분류 문제 (Sentence Pair Classification Tasks, Pairwise Text Classification)   
문서쌍 단위의 유사도를 예측  

- STS-B (The Sematic Textual Similarity Benchmark)
원시데이터 출처: 뉴스 헤드라인, 비디오와 이미지 캡션, NLI 데이터  
annotatiobn: 유사도 1-5스코어 주석   

- MRPC(The Microsoft Resarch Paraphrase Corpus)
원시데이터 출처: 온라인 뉴스  
annotation: 문서쌍이 의미적으로 유사한지 주석  

- QQP(The Quora Question Pairs)
원시데이터 출처:  Quora 질의 응답 웹사이트 커뮤니티  
annotatiobn: 질문 쌍이 의미적으로 동일한지 주석  

# Model

## MT-DNN
- https://y-rok.github.io/nlp/2019/05/20/mt-dnn.html  

## XLNet
- https://tech.scatterlab.co.kr/xlnet-review/  

### 참고자료
https://paperswithcode.com/task/semantic-textual-similarity  