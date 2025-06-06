from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from routes.Appservice.voice_service import save_voice_file


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
