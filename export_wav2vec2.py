from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import os
import uuid

app = Flask(__name__)
CORS(app)

# ✅ 영어 모델
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# ✅ 한국어 모델
processor_ko = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model_ko = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

@app.route("/stt", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # ✅ 언어 파라미터 받기
    lang = request.form.get("lang", "ko")  # 기본 영어

    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    file.save(temp_filename)

    try:
        audio_input, sample_rate = librosa.load(temp_filename, sr=16000)
        input_values = None
        transcription = ""

        if lang == "ko":
            print("🌐 한국어 인식 중...")
            input_values = processor_ko(audio_input, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model_ko(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor_ko.batch_decode(predicted_ids)[0]
        else:
            print("🌐 영어 인식 중...")
            input_values = processor_en(audio_input, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model_en(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor_en.batch_decode(predicted_ids)[0]

        print("📣 인식된 문장:", transcription)

    finally:
        os.remove(temp_filename)

    return jsonify({"text": transcription})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
