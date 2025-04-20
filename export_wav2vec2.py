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
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# ✅ 영어 STT 모델
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# ✅ 한국어 STT 모델
processor_ko = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model_ko = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# ✅ 화자 임벤딩 모델 (x-vector)
speaker_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=None
)

TEMP_VECTORS_FILE = "registered_vectors.json"
FINAL_VECTOR_FILE = "registered_speaker.json"

@app.route("/register", methods=["POST"])
def register_speaker():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    temp_filename = f"register_{uuid.uuid4().hex}.wav"
    file.save(temp_filename)

    try:
        audio = AudioSegment.from_file(temp_filename).set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format="wav")
        waveform, _ = torchaudio.load(temp_filename)

        embedding = speaker_model.encode_batch(waveform).squeeze().numpy()
        norm_vector = embedding / np.linalg.norm(embedding)

        if os.path.exists(TEMP_VECTORS_FILE):
            with open(TEMP_VECTORS_FILE, "r") as f:
                vectors = json.load(f)
        else:
            vectors = []

        vectors.append(norm_vector.tolist())
        with open(TEMP_VECTORS_FILE, "w") as f:
            json.dump(vectors, f)

        print(f"🧾 현재 등록된 화자 벡터 수: {len(vectors)}")

        if len(vectors) == 4:
            mean_vector = np.mean(np.array(vectors), axis=0)
            final_vector = mean_vector / np.linalg.norm(mean_vector)
            with open(FINAL_VECTOR_FILE, "w") as f:
                json.dump(final_vector.tolist(), f)
            os.remove(TEMP_VECTORS_FILE)
            print("✅ 화자 체인 확정 완료")
            print("✅ 평균 벡터:", final_vector.tolist())
            return jsonify({"message": "화자 등록 완료 (4/4)"})
        else:
            return jsonify({"message": f"등록 {len(vectors)}/4 완료"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

@app.route("/stt", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    lang = request.form.get("lang", "ko")
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    file.save(temp_filename)

    try:
        audio = AudioSegment.from_file(temp_filename).set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format="wav")
        waveform, sample_rate = torchaudio.load(temp_filename)

        print("화자 특징 추출 중...")
        speaker_embedding = speaker_model.encode_batch(waveform)
        speaker_vector = speaker_embedding.squeeze().numpy()
        norm_vector = speaker_vector / np.linalg.norm(speaker_vector)

        if not os.path.exists(FINAL_VECTOR_FILE):
            return jsonify({"error": "등록된 화자가 없습니다"}), 403

        with open(FINAL_VECTOR_FILE, "r") as f:
            registered_vector = np.array(json.load(f))

        similarity = np.dot(norm_vector, registered_vector)
        print(f"화자 유사도: {similarity:.4f}")
        if similarity < 0.7:
            return jsonify({"error": "화자 인증 실패"}), 403

        transcription = ""
        if lang == "ko":
            print("한국어 인식 중...")
            input_values = processor_ko(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model_ko(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor_ko.batch_decode(predicted_ids)[0]
        else:
            print("영어 인식 중...")
            input_values = processor_en(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model_en(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor_en.batch_decode(predicted_ids)[0]

        print("인식된 문장:", transcription)

        return jsonify({
            "text": transcription,
            "speaker_vector": norm_vector.tolist()
        })

    except Exception as e:
        print("오류 발생:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)