from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload  # 추가된 부분
import os.path
import pickle

# Google Drive API 인증
SCOPES = ['https://www.googleapis.com/auth/drive.file']
creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

# Google Drive 서비스 구축
service = build('drive', 'v3', credentials=creds)

# 파일 업로드
file_metadata = {'name': 'klue-roberta-large.pt'}
media = MediaFileUpload('/data/ephemeral/home/gayeon2/model/klue-roberta-large.pt', mimetype='application/octet-stream')
file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
print('File ID: %s' % file.get('id'))
