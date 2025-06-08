from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import traceback
from routes.Appservice.voice_service import save_voice_file
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from Mysqldb import models
voice_bp = Blueprint('voice', __name__)

UPLOAD_FOLDER = "uploads/voice"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@voice_bp.route('/register_voice', methods=['POST'])
def register_voice():
    file = request.files.get('file')
    uuid = request.form.get('uuid')
    index = request.form.get('index')
    selected_keyword = request.form.get('selected_keyword')

    if not file or not uuid or not index or not selected_keyword:
        return jsonify({'error': '필수 값 누락'}), 400

    save_dir = os.path.join('uploads', 'voice')
    os.makedirs(save_dir, exist_ok=True)

    filename = f"speaker{index}.wav"
    save_path = os.path.join(save_dir, filename)
    file.save(save_path)

    # DB 처리
    from Mysqldb.models import get_connection
    from datetime import datetime
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # keyword_id 찾기
        cursor.execute(
            "SELECT id FROM keyword WHERE uuid = %s AND keywd_text = %s",
            (uuid, selected_keyword)
        )
        keyword_row = cursor.fetchone()
        if keyword_row is None:
            return jsonify({'error': '해당 키워드가 존재하지 않음'}), 404
        keyword_id = keyword_row["id"]

        # voice 테이블에 INSERT
        cursor.execute("""
            INSERT INTO voice (keyword_id, uuid, voice_index, voice_path, reg_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (keyword_id, uuid, index, save_path, datetime.now()))
        conn.commit()
    except Exception as e:
        print("DB 오류:", e)
        traceback.print_exc()
        return jsonify({'error': 'DB 저장 실패', 'detail': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    # ✅ voice DB 저장 후 자동 학습 요청 (index == 4일 때만)
    try:
        if index == "4":  # 마지막 녹음일 경우에만 학습 요청
            import requests
            res = requests.post(
                "http://192.168.219.105:5000/train_from_voice_db",
                data={"uuid": uuid}
            )
            print("[INFO] 학습 요청 응답 코드:", res.status_code)
            try:
                print("[INFO] 응답 내용:", res.json())
            except Exception:
                print("[WARN] 응답이 JSON 형식이 아님:", res.text)
    except Exception as train_err:
        print("[WARN] 학습 요청 실패:", train_err)

    return jsonify({'message': '파일 저장 및 DB 저장 완료', 'filename': filename}), 200


@voice_bp.route('/set_selected_keyword', methods=['POST'])
def set_selected_keyword():
    from Mysqldb.models import get_connection  # 수정된 위치

    data = request.get_json()
    uuid = data.get('uuid')
    keyword_text = data.get('keyword_text')

    if not uuid or not keyword_text:
        return jsonify({"error": "UUID 또는 키워드가 누락됨"}), 400

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # 🔍 해당 사용자의 keyword_text에 해당하는 키워드 ID 찾기
        cursor.execute("SELECT id FROM keyword WHERE uuid = %s AND keywd_text = %s", (uuid, keyword_text))
        result = cursor.fetchone()

        if result is None:
            return jsonify({"error": "해당 키워드가 존재하지 않음"}), 404

        keyword_id = result["id"]

        # ✅ userinfo 테이블의 selected_keyword 필드 업데이트
        cursor.execute("UPDATE userinfo SET selected_keyword = %s WHERE uuid = %s", (keyword_id, uuid))
        conn.commit()
        return jsonify({"message": "선택된 키워드가 저장되었습니다."}), 200

    except Exception as e:
        print("DB 오류:", e)
        traceback.print_exc()  # 🔍 정확한 에러 로그 확인 가능
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()


