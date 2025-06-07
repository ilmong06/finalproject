from flask import Blueprint, request, jsonify
import os
import requests
import traceback

main_bp = Blueprint('main', __name__)

@main_bp.route('/upload_voice', methods=['POST'])
def upload_voice():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "파일 없음"}), 400

        file = request.files['file']

        os.makedirs("uploads/voice", exist_ok=True)
        save_path = os.path.join("uploads", "voice", "compare.wav")
        file.save(save_path)

        # 🔽 compare.wav 저장 후 /stt로 전송
        stt_url = "http://192.168.219.105:5000/stt"
        with open(save_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(stt_url, files=files)
            stt_result = response.json()

        return jsonify({
            "message": "compare.wav 저장 완료 및 STT 요청 성공",
            "stt_result": stt_result
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
