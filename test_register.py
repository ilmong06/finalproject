# test_register.py
import requests
import os

# secrets.properties 파일에서 FLASK_BASE_URL 읽기
def get_base_url():
    with open("secrets.properties", "r") as f:
        for line in f:
            if line.startswith("FLASK_BASE_URL"):
                return line.strip().split("=")[1].rstrip("/")  # 끝 슬래시 제거
    raise ValueError("FLASK_BASE_URL not found in secrets.properties")

BASE_URL = get_base_url()

# 화자 등록 요청 함수
def send_file_to_register(file_path):
    url = f"{BASE_URL}/register"
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
        response = requests.post(url, files=files)
        print(f"[화자등록] {file_path} -> {response.status_code}")
        try:
            print(response.json())
        except Exception:
            print("⚠ JSON 파싱 실패. 응답 본문:\n", response.text)

# 키워드 등록 요청 함수
def send_file_to_keyword(file_path, keyword):
    url = f"{BASE_URL}/register_keyword"
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
        data = {'keyword': keyword}
        response = requests.post(url, files=files, data=data)
        print(f"[키워드등록] {file_path} -> {response.status_code}")
        try:
            print(response.json())
        except Exception:
            print("⚠ JSON 파싱 실패. 응답 본문:\n", response.text)

# 테스트용 경로 설정
speaker_files = [
    "test_audio/speaker1.wav",
    "test_audio/speaker2.wav",
    "test_audio/speaker3.wav",
    "test_audio/speaker4.wav"
]

keyword_files = [
    "test_audio/keyword1.wav",
    "test_audio/keyword2.wav",
    "test_audio/keyword3.wav",
    "test_audio/keyword4.wav",
    "test_audio/keyword5.wav",
    "test_audio/keyword6.wav"
]

keyword_text = "안녕하세요"

# 화자 등록 테스트
for file in speaker_files:
    send_file_to_register(file)

# 키워드 등록 테스트
for file in keyword_files:
    send_file_to_keyword(file, keyword_text)
