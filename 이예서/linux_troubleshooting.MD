### Conda 세팅 관련
- root에서는 conda를 바로 사용할 수 없고 init 혹은 source ~/.bashrc 해줘야함
- 그러나 user를 생성하고 거기에 각각 세팅해두면 바로 해당 유저로 접근했을 때 자동으로 conda activate됨

1. conda config --append envs_dirs /data/ephemeral/home/
2. 유저 생성: useradd yeseo
3. 유저 홈 변경: usermod -d /data/ephemeral/home/yeseo /data/ephemeral/home/
4. 유저 기본 셀 bash: sudo chsh -s /bin/bash yeseo
5. conda 패스 설정: export PATH="/opt/conda/bin:$PATH"
6. conda 세팅   
  conda init  
  conda create --name yeseo python=3.10  
7. 실행시 자동으로 activate 설정하기  
  echo "conda activate yeseo" >> ~/.bashrc  
  source ~/.bashrc  
8. 이후 해당 폴더 모든 권한 열어주기 sudo chmod -R 777 /data/ephemeral/home/yeseo/
