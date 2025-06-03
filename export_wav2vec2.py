import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from flask import Flask, request, jsonify
from flask_cors import CORS
from speechbrain.pretrained import SpeakerRecognition
from vosk import Model as VoskModel, KaldiRecognizer
from pydub import AudioSegment
from pathlib import Path
import torchaudio
import torch.nn as nn
import torch
import uuid
import json
import traceback
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import wave
from g2pk import G2p
g2p = G2p()
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_stt_key.json"
from google.cloud import speech

# ✅ MatchboxNet 인코더 임포트
from train_matchboxnet_protonet import MatchboxNetEncoder

# ✅ Flask 설정
app = Flask(__name__)
CORS(app)

# ✅ Vosk 모델 로드
vosk_model = VoskModel("model")

def segment_waveform(waveform, sample_rate=16000, segment_ms=250  ):
    segment_samples = int(sample_rate * segment_ms / 1000  )
    segments = []
    for i in range(0, waveform.shape[1], segment_samples):
        chunk = waveform[:, i:i+segment_samples]
        if chunk.shape[1] == segment_samples:
            energy = chunk.pow(2).mean().item()
            if energy > 1e-8:
                segments.append(chunk)
    return segments

# ✅ MatchboxNet 모델 로드
matchbox_model = MatchboxNetEncoder()
if os.path.exists("matchbox_model.pt"):
    state = torch.load("matchbox_model.pt", map_location="cpu")
    matchbox_model.load_state_dict(state["model"])
    matchbox_model.eval()
    print("✅ matchbox_model.pt 로드 완료")
else:
    print("⚠️ matchbox_model.pt 없음 → 키워드 등록만 가능")

# ✅ 화자 인식 모델
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
TEMP_VECTORS_FILE = "registered_vectors.json"
FINAL_VECTOR_FILE = "registered_speaker.json"

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

        if len(vectors) == 4:
            vectors_np = np.array(vectors)
            mean_vector = np.mean(vectors_np, axis=0)
            final_vector = mean_vector / np.linalg.norm(mean_vector)
            with open(FINAL_VECTOR_FILE, "w") as f:
                json.dump(final_vector.tolist(), f)
            os.remove(TEMP_VECTORS_FILE)
            return jsonify({"message": "화자 등록 완료 (4/4)"})
        else:
            return jsonify({"message": f"등록 {len(vectors)}/4 완료"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)

@app.route("/register_keyword", methods=["POST"])
def register_keyword():
    raw_keyword = request.form.get("keyword")
    keyword = g2p(raw_keyword).replace(" ", "")
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
        audio = AudioSegment.from_file(str(save_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(save_path), format="wav")

        print(f"[DEBUG] ▶️ record_{index}.wav 저장 완료")

        if index == 6:
            vectors = []
            for i in range(1, 7):
                wav_path = save_dir / f"record_{i}.wav"
                waveform, sr = torchaudio.load(wav_path)
                segments = segment_waveform(waveform)

                print(f"[DEBUG] 🔍 record_{i}.wav 세그먼트 수: {len(segments)}")

                if len(segments) == 0:
                    print(f"[WARN] ❌ 세그먼트 없음 → {wav_path}는 무시됨")
                    continue

                for seg in segments:
                    with torch.no_grad():
                        seg = seg.unsqueeze(0)  # [1, 1, T]
                        emb = matchbox_model(seg).squeeze().mean(dim=-1).numpy()
                    emb = emb / np.linalg.norm(emb)  # ✅ 정규화
                    vectors.append(np.array(emb).flatten().tolist())  # ✅ list[float] 형태로 저장

            print(f"[DEBUG] ✅ 생성된 벡터 수: {len(vectors)}")

            if len(vectors) == 0:
                return jsonify({"error": "❌ 키워드 벡터 생성 실패: 세그먼트 없음"}), 500

            # ✅ label_map.json에 저장
            label_map_path = "label_map.json"
            if os.path.exists(label_map_path):
                with open(label_map_path, "r") as f:
                    label_map = json.load(f)
            else:
                label_map = {}

            # 🔁 최종 저장
            label_map[keyword] = [v if isinstance(v, list) else v.tolist() for v in vectors]

            with open(label_map_path, "w") as f:
                json.dump(label_map, f, indent=2)

            print(f"[DEBUG] 💾 label_map.json 저장 완료: 키워드={keyword}, 벡터={len(vectors)}")

            subprocess.run(["python", "train_fewshot.py"])
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

    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    file.save(temp_filename)

    try:
        audio = AudioSegment.from_file(temp_filename).set_frame_rate(16000).set_channels(1)
        audio.export(temp_filename, format="wav")
        waveform, sr = torchaudio.load(temp_filename)

        # 🔐 화자 인증
        speaker_embedding = speaker_model.encode_batch(waveform)
        speaker_vector = speaker_embedding.squeeze().numpy()
        norm_vector = speaker_vector / np.linalg.norm(speaker_vector)

        if not os.path.exists(FINAL_VECTOR_FILE):
            return jsonify({"error": "등록된 화자가 없습니다"}), 403
        with open(FINAL_VECTOR_FILE, "r") as f:
            registered_vector = np.array(json.load(f)).flatten()

        sim_sp = float(np.dot(norm_vector, registered_vector))
        print(f"[DEBUG] 🔐 화자 유사도: {sim_sp:.4f}")

        # ✅ Google STT로 먼저 음성 인식
        try:
            client = speech.SpeechClient()
            with open(temp_filename, "rb") as audio_file:
                content = audio_file.read()
            audio_g = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR",
            )
            response = client.recognize(config=config, audio=audio_g)
            transcript = " ".join(result.alternatives[0].transcript for result in response.results)
            print(f"[DEBUG] 🗣️ Google STT 결과: {transcript}")
        except Exception as stt_err:
            transcript = ""
            print(f"[ERROR] Google STT 실패: {stt_err}")

        # ✅ KoG2P로 텍스트를 발음 단위로 변환
        phonetic_transcript = g2p(transcript).replace(" ", "")
        print(f"[DEBUG] 🔤 변환된 발음: {phonetic_transcript}")

        # ✅ label_map.json 기반 키워드 유사도 계산
        if not os.path.exists("label_map.json"):
            return jsonify({"error": "등록된 키워드 없음"}), 403

        with open("label_map.json", "r") as f:
            label_map = json.load(f)

        best_keyword = None
        best_score = -1
        segment, sr = torchaudio.load(temp_filename)
        segments = segment_waveform(segment)

        segment_embeddings = []
        for seg in segments:
            with torch.no_grad():
                seg = seg.unsqueeze(0)
                emb = matchbox_model(seg).squeeze().mean(dim=-1).numpy()
            emb = emb / np.linalg.norm(emb)
            segment_embeddings.append(emb)

        for keyword, vec_list in label_map.items():
            phonetic_keyword = g2p(keyword).replace(" ", "")
            if phonetic_keyword in phonetic_transcript:
                # 가장 높은 유사도를 best_score로
                for emb in segment_embeddings:
                    for ref_vec in vec_list:
                        ref_vec = np.array(ref_vec)
                        if emb.shape != ref_vec.shape:
                            continue
                        score = float(np.dot(emb, ref_vec))
                        if score > best_score:
                            best_score = score
                            best_keyword = keyword

        sim_kw = best_score
        print(f"[DEBUG] 🔍 키워드 유사도: {sim_kw:.4f}")

        if sim_kw < 0.7:
            return jsonify({
                "error": "키워드 인증 실패",
                "triggered_keyword": best_keyword,
                "similarity": round(sim_kw, 4),
                "text": transcript
            }), 403

        return jsonify({
            "text": transcript.strip(),
            "speaker_similarity": round(sim_sp, 4),
            "triggered_keyword": best_keyword,
            "keyword_similarity": round(sim_kw, 4),
            "s_total": round(sim_sp + sim_kw, 4)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print("[WARN] 파일 삭제 실패:", e)

# ✅ 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
