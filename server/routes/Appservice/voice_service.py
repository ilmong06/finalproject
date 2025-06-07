from Mysqldb.models import get_connection
import os
from datetime import datetime

# ✅ 음성 파일 저장 후 DB에 경로 등록
def save_voice_file(user_uuid, file_path):
    from datetime import datetime
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            # 🔍 선택된 키워드 ID 불러오기
            cursor.execute("SELECT selected_keyword FROM userinfo WHERE uuid = %s", (user_uuid,))
            result = cursor.fetchone()
            if not result or not result["selected_keyword"]:
                raise Exception("선택된 키워드가 없습니다.")
            keyword_id = result["selected_keyword"]

            # 🔍 해당 키워드에 대해 현재 몇 번째 등록인지 확인
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM voice WHERE uuid = %s AND keyword_id = %s",
                (user_uuid, keyword_id)
            )
            count_result = cursor.fetchone()
            voice_index = count_result["cnt"] + 1  # 1부터 시작

            if voice_index > 4:
                raise Exception("이미 4개의 음성이 등록되었습니다.")

            # ✅ voice 테이블에 insert
            sql = """
                INSERT INTO voice (uuid, keyword_id, voice_index, voice_path, reg_date)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (user_uuid, keyword_id, voice_index, file_path, datetime.now()))
        connection.commit()
    finally:
        connection.close()

