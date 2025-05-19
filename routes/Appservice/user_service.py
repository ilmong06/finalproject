from Mysqldb import models
from DataControl import validator
from datetime import datetime

# 🔵 사용자 등록 함수
def register_user(uuid, name, phnum):
    """
    새로운 사용자 등록
    - uuid : 사용자 고유 ID
    - name : 사용자 이름
    - phnum : 사용자 전화번호
    """
    # 1. 전화번호 형식 검사
    if not validator.is_valid_phnum(phnum):
        raise ValueError("전화번호는 11자리 숫자여야 합니다.")
    
    # 2. 전화번호 중복 검사
    if validator.is_duplicate_phnum(phnum):
        raise ValueError("이미 등록된 전화번호입니다.")

    # 3. DB에 사용자 정보 저장
    now = datetime.now()
    models.insert_user(uuid, name, phnum, now)

    return {"message": "사용자 등록 성공"}

# 🔵 사용자 정보 조회 함수 
def get_user_info(uuid):
    """
    사용자 UUID로 정보 조회
    """
    user = models.get_user_by_uuid(uuid)
    if not user:
        raise ValueError("해당 UUID의 사용자가 없습니다.")
    
    return {
        "uuid": user[0],
        "name": user[1],
        "phnum": user[2],
        "reg_date": user[3]
    }

