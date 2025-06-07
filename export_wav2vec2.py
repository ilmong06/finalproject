import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
import pymysql
from datetime import datetime
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

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="",  # 비밀번호 없는 경우
        database="endproject",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
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
    uuid_value = request.form.get("uuid")
    order_value = request.form.get("order")

    if not raw_keyword or not uuid_value or not order_value:
        return jsonify({"error": "키워드, UUID, 순번 누락"}), 400

    # ✅ 키워드 텍스트 파일에 저장 (keywords.txt)
    keyword_file = "keywords.txt"
    keywords = []
    if os.path.exists(keyword_file):
        with open(keyword_file, "r", encoding="utf-8") as f:
            keywords = f.read().splitlines()

    if raw_keyword not in keywords:
        keywords.append(raw_keyword)
        keywords.append(g2p(raw_keyword).replace(" ", ""))  # 발음형도 같이 저장
        with open(keyword_file, "w", encoding="utf-8") as f:
            f.write("\n".join(keywords))

    # ✅ MySQL DB에 저장
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO keyword (uuid, keywd_text, keywd_order, add_date)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (uuid_value, raw_keyword, int(order_value), datetime.now()))
        conn.commit()
    except Exception as e:
        return jsonify({"error": f"MySQL 저장 실패: {str(e)}"}), 500
    finally:
        conn.close()

    return jsonify({"message": f"{raw_keyword} 키워드 등록 완료 ✅"}), 200

@app.route("/finalize_speaker_registration", methods=["POST"])
def finalize_speaker_registration():
    uuid = request.form.get("uuid")
    if not uuid:
        return jsonify({"error": "UUID 누락"}), 400

    try:
        conn = get_connection()
        with conn.cursor(dictionary=True) as cursor:
            # ✅ 현재 선택된 키워드 ID 가져오기
            cursor.execute("SELECT selected_keyword FROM userinfo WHERE uuid = %s", (uuid,))
            row = cursor.fetchone()
            if not row or not row["selected_keyword"]:
                return jsonify({"error": "선택된 키워드 없음"}), 400
            keyword_id = row["selected_keyword"]

            # ✅ voice 테이블에서 해당 사용자 + 키워드 조합의 4개 음성 경로 가져오기
            cursor.execute("""
                SELECT voice_path FROM voice
                WHERE uuid = %s AND keyword_id = %s
                ORDER BY voice_index ASC
            """, (uuid, keyword_id))
            rows = cursor.fetchall()

            if len(rows) < 4:
                return jsonify({"error": "녹음 파일이 4개 미만입니다"}), 400

            paths = [r["voice_path"] for r in rows]

    except Exception as e:
        return jsonify({"error": f"MySQL 오류: {str(e)}"}), 500
    finally:
        conn.close()

    # ✅ 음성 벡터 추출
    try:
        from train_matchboxnet_protonet import MatchboxNetEncoder  # 필요시 수정
        model = MatchboxNetEncoder()
        model.load_state_dict(torch.load("matchbox_model.pt", map_location="cpu")["model"])
        model.eval()

        vectors = []
        for path in paths:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform[0:1, :]  # mono 처리
            with torch.no_grad():
                emb = model(waveform.unsqueeze(0))  # [1, 1, T] → [1, 80]
                vectors.append(emb.squeeze().numpy())

        final_vector = np.mean(np.stack(vectors), axis=0)

        # ✅ 저장
        os.makedirs("vector_store", exist_ok=True)
        final_path = os.path.join("vector_store", "registered_speaker.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump({uuid: final_vector.tolist()}, f, indent=2)

        return jsonify({"message": "화자 벡터 등록 완료 ✅"}), 200

    except Exception as e:
        return jsonify({"error": f"벡터 처리 실패: {str(e)}"}), 500


# ✅ STT + 화자 + 키워드 텍스트 인증
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

        # ✅ Google STT로 텍스트 추출
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

        # ✅ 발음 변환
        phonetic_transcript = g2p(transcript).replace(" ", "")
        print(f"[DEBUG] 🔤 변환된 발음: {phonetic_transcript}")

        # ✅ 키워드 포함 여부만으로 판별
        keyword_file = "keywords.txt"
        if not os.path.exists(keyword_file):
            return jsonify({"error": "등록된 키워드 없음"}), 403

        with open(keyword_file, "r", encoding="utf-8") as f:
            keyword_list = f.read().splitlines()

        for keyword in keyword_list:
            original_keyword = keyword
            g2p_keyword = g2p(keyword).replace(" ", "")

            if (original_keyword in transcript or
                g2p_keyword in transcript.replace(" ", "") or
                original_keyword in phonetic_transcript or
                g2p_keyword in phonetic_transcript):
                matched_keyword = original_keyword
                sim_kw = 1.0
                break


        print(f"[DEBUG] 🔍 키워드 유사도: {sim_kw:.4f}")


        if not matched_keyword:
            return jsonify({
                "error": "키워드 인증 실패",
                "triggered_keyword": None,
                "similarity": sim_kw,
                "text": transcript
            }), 403

        return jsonify({
            "text": transcript.strip(),
            "speaker_similarity": round(sim_sp, 4),
            "triggered_keyword": matched_keyword,
            "triggered_keyword_g2p": g2p(matched_keyword).replace(" ", ""),
            "keyword_similarity": sim_kw,
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
