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
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# ✅ STT 모델
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor_ko = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model_ko = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

# ✅ 화자 임베딩 모델
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# ✅ 저장 파일
TEMP_VECTORS_FILE = "registered_vectors.json"
FINAL_VECTOR_FILE = "registered_speaker.json"
KEYWORD_VECTOR_FILE = "registered_keyword_vectors.json"

# ✅ 화자 등록
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

        vectors = []
        if os.path.exists(TEMP_VECTORS_FILE):
            with open(TEMP_VECTORS_FILE, "r") as f:
                vectors = json.load(f)
        vectors.append(norm_vector.tolist())
        with open(TEMP_VECTORS_FILE, "w") as f:
            json.dump(vectors, f)

        print(f"🧾 현재 등록된 화자 벡터 수: {len(vectors)}")

        if len(vectors) == 4:
            vectors_np = np.array(vectors)
            mean_vector = np.mean(vectors_np, axis=0)
            final_vector = mean_vector / np.linalg.norm(mean_vector)
            with open(FINAL_VECTOR_FILE, "w") as f:
                json.dump(final_vector.tolist(), f)
            os.remove(TEMP_VECTORS_FILE)
            print("✅ 화자 체인 확정 완료")
            print("✅ 평균 벡터:", final_vector.tolist())

            # 시각화
            plt.figure(figsize=(14, 6))
            for i, vec in enumerate(vectors_np):
                plt.plot(vec, label=f"Registered Vector {i+1}", linestyle='--', alpha=0.6)
            plt.plot(final_vector, label="Mean Speaker Vector", linewidth=2.5)
            plt.title("📊 화자 벡터 평균 처리 결과")
            plt.xlabel("차원 인덱스")
            plt.ylabel("값")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            return jsonify({"message": "화자 등록 완료 (4/4)"})
        else:
            return jsonify({"message": f"등록 {len(vectors)}/4 완료"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

# ✅ 키워드 등록
@app.route("/register_keyword", methods=["POST"])
def register_keyword():
    keyword = request.form.get("keyword")
    if not keyword:
        return jsonify({"error": "키워드가 없습니다."}), 400

    if "file" not in request.files:
        return jsonify({"error": "음성 파일이 없습니다."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    temp_filename = f"keyword_{uuid.uuid4().hex}.wav"
    file.save(temp_filename)

    try:
        audio = AudioSegment.from_file(temp_filename).set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format="wav")
        waveform, _ = torchaudio.load(temp_filename)
        embedding = speaker_model.encode_batch(waveform).squeeze().numpy()
        norm_vector = embedding / np.linalg.norm(embedding)

        if os.path.exists(KEYWORD_VECTOR_FILE):
            with open(KEYWORD_VECTOR_FILE, "r") as f:
                data = json.load(f)
        else:
            data = {}

        if keyword not in data:
            data[keyword] = []

        data[keyword].append(norm_vector.tolist())

        if len(data[keyword]) == 6:
            mean_vector = np.mean(np.array(data[keyword]), axis=0)
            final_vector = mean_vector / np.linalg.norm(mean_vector)
            data[keyword].append(final_vector.tolist())  # 마지막 인덱스 = 평균 벡터
            print(f"✅ 키워드 '{keyword}' 등록 완료 (6/6)")

        with open(KEYWORD_VECTOR_FILE, "w") as f:
            json.dump(data, f)

        return jsonify({"message": f"키워드 '{keyword}' 등록 {len(data[keyword])}/6 완료"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

# ✅ STT + 화자 + 키워드 인증
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

        # 🔐 화자 인증
        if not os.path.exists(FINAL_VECTOR_FILE):
            return jsonify({"error": "등록된 화자가 없습니다"}), 403
        with open(FINAL_VECTOR_FILE, "r") as f:
            registered_vector = np.array(json.load(f))
        speaker_similarity = np.dot(norm_vector, registered_vector)
        print(f"화자 유사도: {speaker_similarity:.4f}")
        if speaker_similarity < 0.7:
            return jsonify({"error": "화자 인증 실패"}), 403

        # 🎯 키워드 인증
        if not os.path.exists(KEYWORD_VECTOR_FILE):
            return jsonify({"error": "등록된 키워드가 없습니다"}), 403
        with open(KEYWORD_VECTOR_FILE, "r") as f:
            keyword_data = json.load(f)

        triggered_keyword = None
        for keyword, vectors in keyword_data.items():
            if len(vectors) < 7:
                continue
            keyword_vector = np.array(vectors[-1])  # 평균 벡터
            keyword_similarity = np.dot(norm_vector, keyword_vector)
            print(f"🔍 키워드 '{keyword}' 유사도: {keyword_similarity:.4f}")
            if keyword_similarity > 0.8:
                triggered_keyword = keyword
                break

        if not triggered_keyword:
            return jsonify({"error": "키워드 인증 실패"}), 403

        # ✅ STT 수행
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

        print(f"📣 인식된 문장: {transcription}")

        return jsonify({
            "text": transcription,
            "triggered_keyword": triggered_keyword,
            "speaker_vector": norm_vector.tolist()
        })

    except Exception as e:
        print("오류 발생:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

# ✅ 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
