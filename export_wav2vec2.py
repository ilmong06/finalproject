import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import SpeakerRecognition
from pydub import AudioSegment
from pathlib import Path
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import torch
import uuid
import json
import traceback
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from torchaudio.models.conformer import Conformer

app = Flask(__name__)
CORS(app)
def segment_waveform(waveform, sample_rate=16000, segment_ms=1000):
    segment_samples = int(sample_rate * segment_ms / 1000)
    segments = []
    for i in range(0, waveform.shape[1], segment_samples):
        chunk = waveform[:, i:i+segment_samples]
        if chunk.shape[1] == segment_samples:
            energy = chunk.pow(2).mean().item()
            if energy > 1e-5:  # 무음 제거
                segments.append(chunk)
    return segments
# ✅ Conformer 기반 인코더 정의
# 클래스 정의만 먼저
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, encoder_dim=144, num_layers=4):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=input_dim
        )
        self.encoder = Conformer(
            input_dim=input_dim,
            num_heads=4,
            ffn_dim=encoder_dim * 4,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,
            dropout=0.1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, waveform):
        mel = self.mel_transform(waveform)      # [1, n_mels, T]
        mel = mel.transpose(1, 2)               # [1, T, n_mels]
        lengths = torch.full(size=(mel.shape[0],), fill_value=mel.shape[1], dtype=torch.long)
        x, _ = self.encoder(mel,lengths)
        x = x.transpose(1, 2)                   # [1, d_model, T]
        x = self.pool(x).squeeze(-1)            # [1, d_model]
        return x

# 클래스 바깥에서 인스턴스 생성 및 모델 로딩
conformer_encoder = ConformerEncoder()
# ✅ 모델이 있을 경우에만 로드
if os.path.exists("fewshot_model.pt"):
    state = torch.load("fewshot_model.pt", map_location="cpu")
    conformer_encoder.load_state_dict(state["model"])
    conformer_encoder.eval()
    print("✅ fewshot_model.pt 로드 완료")
else:
    print("⚠️ fewshot_model.pt 없음 → 키워드 등록만 가능, 학습 후 재시작 필요")

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
        waveform, sr = torchaudio.load(temp_filename)
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
    if not keyword or "file" not in request.files:
        return jsonify({"error": "키워드 또는 파일 없음"}), 400

    save_dir = Path("data/custom") / keyword
    save_dir.mkdir(parents=True, exist_ok=True)

    existing = list(save_dir.glob("*.wav"))
    index = len(existing) + 1
    save_path = save_dir / f"record_{index}.wav"

    try:
        file = request.files["file"]
        file.save(str(save_path))

        # 전처리: wav → mono 16kHz
        audio = AudioSegment.from_file(str(save_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(save_path), format="wav")

        if index == 6:
            vectors = []
            for i in range(1, 7):
                wav_path = save_dir / f"record_{i}.wav"
                waveform, sr = torchaudio.load(wav_path)
                segments = segment_waveform(waveform)
                for seg in segments:
                    with torch.no_grad():
                        emb = conformer_encoder(seg).squeeze().numpy()
                    emb = emb / np.linalg.norm(emb)
                    vectors.append(emb)

            print("✅ vectors 개수:", len(vectors))  # ✅ if 블록 안으로 넣기
            print("✅ vectors[0] shape:", np.array(vectors[0]).shape)

            # 🔹 저장
            label_map_path = "label_map.json"
            if os.path.exists(label_map_path):
                with open(label_map_path, "r") as f:
                    label_map = json.load(f)
            else:
                label_map = {}
            label_map[keyword] = [vec.tolist() for vec in vectors]

            with open(label_map_path, "w") as f:
                json.dump(label_map, f)

            print("🚀 키워드 6개 등록 완료 → 학습 시작")  # ✅ 로그
            subprocess.run(["python", "train_fewshot.py"])  # ✅ 학습 실행
            return jsonify({"message": f"{keyword} 키워드 등록 완료 ✅"}), 200

        else:
            return jsonify({"message": f"{keyword} 키워드 {index}/6 녹음 저장 완료"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
        waveform, sr = torchaudio.load(temp_filename)

        print("화자 특징 추출 중...")
        speaker_embedding = speaker_model.encode_batch(waveform)
        speaker_vector = speaker_embedding.squeeze().numpy()
        norm_vector = speaker_vector / np.linalg.norm(speaker_vector)

        if not os.path.exists(FINAL_VECTOR_FILE):
            return jsonify({"error": "등록된 화자가 없습니다"}), 403
        with open(FINAL_VECTOR_FILE, "r") as f:
            registered_vector = np.array(json.load(f))
        speaker_similarity = np.dot(norm_vector, registered_vector)
        print(f"화자 유사도: {speaker_similarity:.4f}")
        if speaker_similarity < 0.7:
            return jsonify({"error": "화자 인증 실패"}), 403

        # 🔍 키워드 인증 (segment 기반)
        segments = segment_waveform(waveform)
        segment_embeddings = []
        for seg in segments:
            with torch.no_grad():
                emb = conformer_encoder(seg).squeeze().numpy()
            emb = emb / np.linalg.norm(emb)
            segment_embeddings.append(emb)

        triggered_keyword = None
        best_score = -1

        with open("label_map.json", "r") as f:
            label_map = json.load(f)

        print("\n📊 키워드 유사도:")
        for keyword, vec_list in label_map.items():
            for emb in segment_embeddings:
                for ref_vec in vec_list:
                    ref_vec = np.array(ref_vec).reshape(-1)
                    emb = np.array(emb).reshape(-1)
                    score = np.dot(emb, ref_vec)
                    if isinstance(score, np.ndarray):
                        score = score.item()
                    print(f"🔸 '{keyword}' ↔ 세그먼트 유사도 : {score:.4f}")
                    if score > best_score:
                        best_score = score
                        triggered_keyword = keyword





        # 인증 실패 시 중단
        if best_score < 0.7:
            return jsonify({
                "error": "키워드 인증 실패 (segment 기반)",
                "triggered_keyword": triggered_keyword,
                "similarity": round(best_score, 4)
            }), 403


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
            "similarity": round(best_score, 4),
            "speaker_vector": norm_vector.tolist()
        })

    except Exception as e:
           traceback.print_exc()  # 🔍 콘솔 전체 에러 출력
           return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

# ✅ 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
