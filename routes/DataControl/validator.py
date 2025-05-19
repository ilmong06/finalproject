import re
from Mysqldb import models

# 🔵 전화번호 형식 검사
def is_valid_phnum(phnum):
    """
    전화번호가 11자리 숫자인지 확인
    ex) 01012345678
    """
    pattern = re.compile(r'^\d{11}$')
    return bool(pattern.match(phnum))

# 🔵 전화번호 중복 체크
def is_duplicate_phnum(phnum):
    """
    DB에 같은 전화번호가 이미 존재하는지 확인
    """
    existing_user = models.get_user_by_phnum(phnum)
    return existing_user is not None

# 🔵 키워드 중복 체크 (기존 있던 것 유지)
def is_keyword_exist(uuid, keyword_text):
    """
    같은 사용자가 같은 키워드를 이미 등록했는지 확인
    """
    existing_keyword = models.get_keyword_by_text(uuid, keyword_text)
    return existing_keyword is not None
