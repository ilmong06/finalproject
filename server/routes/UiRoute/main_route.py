# main_route.py

from flask import Blueprint, request, jsonify
import os

main_bp = Blueprint('main', __name__)

@main_bp.route('/upload_voice', methods=['POST'])
def upload_voice():
    if 'file' not in request.files:
        return jsonify({"error": "파일 없음"}), 400

    file = request.files['file']

    os.makedirs("uploads/voice", exist_ok=True)  # 디렉토리 없으면 생성
    save_path = os.path.join("uploads", "voice", "compare.wav")
    file.save(save_path)

    return jsonify({"message": "compare.wav 저장 완료"}), 200
