from flask import Blueprint, request, jsonify
from flask_cors import CORS
import pymysql
import os
import requests

main_bp = Blueprint('main', __name__)
CORS(main_bp)


# ✅ DB 연결 함수
def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",
        database="endproject",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )


# ✅ /upload_voice 라우트
@main_bp.route("/upload_voice", methods=["POST"])
def upload_voice():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 포함되어야 합니다."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일명이 비어 있습니다."}), 400

    # 저장 경로 설정
    save_dir = "compare"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "compare.wav")
    file.save(save_path)

    # ✅ STT 서버로 compare.wav 전송
    try:
        stt_url = "http://192.168.219.231:5000/stt"
        with open(save_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(stt_url, files=files)

        if response.status_code != 200:
            return jsonify({"error": "STT 요청 실패", "status": response.status_code}), 500

        stt_result = response.json()
        return jsonify({
            "message": "compare.wav 저장 완료 및 STT 요청 성공",
            "stt_result": stt_result
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
