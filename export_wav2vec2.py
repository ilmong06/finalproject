import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.pretrained import SpeakerRecognition
from pydub import AudioSegment
from pathlib import Path
import torchaudio
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
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

    raw_path = save_dir / f"raw_{index}.wav"
    fixed_path = save_dir / f"record_{index}.wav"

    try:
        file = request.files["file"]
        file.save(str(raw_path))

        audio = AudioSegment.from_file(str(raw_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(fixed_path), format="wav")
        os.remove(str(raw_path))

        if index == 6:
            os.system("python train_fewshot.py")
            return jsonify({"message": f"{keyword} 키워드 6개 녹음 완료 → 모델 학습 시작됨"}), 200
        else:
            return jsonify({"message": f"{keyword} 키워드 {index}/6 녹음 저장 완료"}), 200

    except Exception as e:
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
        waveform, _ = torchaudio.load(temp_filename)

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

        from sklearn.metrics.pairwise import cosine_similarity
        import torch.nn as nn
        import torch.nn.functional as F

        class ResKeywordNet(nn.Module):
            def __init__(self):
                super().__init__()
                base_model = resnet18(pretrained=False)
                base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.features = nn.Sequential(*list(base_model.children())[:-1])
                self.fc = nn.Linear(512, 128)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

        def extract_mel(filepath, target_length=232):
            waveform, sr = torchaudio.load(filepath)
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=1024,
                hop_length=160,
                n_mels=80
            )(waveform)
            if mel.shape[-1] < target_length:
                pad = target_length - mel.shape[-1]
                mel = F.pad(mel, (0, pad))
            elif mel.shape[-1] > target_length:
                mel = mel[:, :, :target_length]
            return mel.unsqueeze(0)

        # ✅ 키워드 유사도 판단
        if not os.path.exists("fewshot_model.pt") or not os.path.exists("label_map.json"):
            return jsonify({"error": "Few-shot 모델 또는 라벨맵 없음"}), 403

        mel = extract_mel(temp_filename)
        model = ResKeywordNet()
        checkpoint = torch.load("fewshot_model.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

        with open("label_map.json", "r", encoding="utf-8") as f:
            label_map = json.load(f)

        with torch.no_grad():
            emb = model(mel).numpy()

        triggered_keyword = None
        best_score = 0
        print("\n📊 키워드 유사도 (Cosine Similarity):")
        for keyword, vec in label_map.items():
            score = cosine_similarity(emb, [vec])[0][0]
            print(f"🔸 '{keyword}' ↔ 입력 음성 : {score:.4f}")
            if score > best_score:
                best_score = score
                triggered_keyword = keyword

        if best_score < 0.7:
            return jsonify({
                "error": "키워드 인증 실패 (Few-shot)",
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
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

# ✅ 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
