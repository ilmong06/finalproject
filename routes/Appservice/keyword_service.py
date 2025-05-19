import os
import json
import numpy as np
from Mysqldb import models
from datetime import datetime
from DataControl import validator

# 모델 저장 경로
KEYWORD_VECTOR_FILE = 'model/registered_keyword_vectors.json'

# 🔵 키워드 등록
def register_keyword(uuid, keyword_text, files):
    """
    사용자가 입력한 키워드와 음성 파일들을 등록
    - uuid : 사용자 고유 ID
    - keyword_text : 키워드 텍스트
    - files : 키워드 음성 녹음 파일들 (6개)
    """
    # 1. 키워드 DB 저장
    now = datetime.now()
    models.insert_keyword(uuid, keyword_text, now)

    # 2. 키워드 벡터 평균 저장 (임시로 random 벡터 생성)
    # (원래는 모델로부터 embedding 받아야 함)
    keyword_vectors = []

    for file in files:
        # 실제로는 여기에 모델 인퍼런스 추가해야 함
        dummy_vector = np.random.rand(512)  # 512차원 dummy
        keyword_vectors.append(dummy_vector)

    keyword_avg_vector = np.mean(keyword_vectors, axis=0)

    # 3. 파일에 저장
    if os.path.exists(KEYWORD_VECTOR_FILE):
        with open(KEYWORD_VECTOR_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[keyword_text] = keyword_avg_vector.tolist()

    with open(KEYWORD_VECTOR_FILE, 'w') as f:
        json.dump(data, f)
