import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import SpeakerRecognition
from pydub import AudioSegment
import torchaudio
import torch
import uuid

app = Flask(__name__)
CORS(app)

# ✅ 영어 STT 모델
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# ✅ 한국어 STT 모델
processor_ko = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model_ko = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# ✅ 화자 임베딩 모델 (x-vector)
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=None  # → 자동 캐시 경로 사용
)

@app.route("/stt", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    lang = request.form.get("lang", "ko")  # 기본값: 한국어

    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    file.save(temp_filename)

    try:
        # ✅ WAV 재인코딩 (torchaudio 오류 방지)
        audio = AudioSegment.from_file(temp_filename)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format="wav")

        # ✅ torchaudio 로딩
        waveform, sample_rate = torchaudio.load(temp_filename)

        # ✅ Wav2Vec2 STT
        transcription = ""
        if lang == "ko":
            print("🌐 한국어 인식 중...")
            input_values = processor_ko(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model_ko(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor_ko.batch_decode(predicted_ids)[0]
        else:
            print("🌐 영어 인식 중...")
            input_values = processor_en(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model_en(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor_en.batch_decode(predicted_ids)[0]

        print("📣 인식된 문장:", transcription)

        # ✅ 화자 특징 추출 (x-vector)
        print("🧬 화자 특징 추출 중...")
        speaker_embedding = speaker_model.encode_batch(waveform)
        speaker_vector = speaker_embedding.squeeze().tolist()

    except Exception as e:
        print("❌ 오류 발생:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

    return jsonify({
        "text": transcription,
        "speaker_vector": speaker_vector
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
