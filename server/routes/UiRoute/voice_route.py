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
    if 'file' not in request.files or 'uuid' not in request.form:
        return jsonify({'error': '파일 또는 UUID 누락'}), 400

    file = request.files['file']
    user_uuid = request.form['uuid']

    if file.filename == '':
        return jsonify({'error': '파일명이 없습니다'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, f"{user_uuid}_{uuid.uuid4().hex}.wav")
    file.save(save_path)

    # ✅ 서비스 로직 수행
    save_voice_file(user_uuid, save_path)

    return jsonify({'message': '음성 등록 성공', 'path': save_path}), 200

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


