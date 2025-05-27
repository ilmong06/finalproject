import os
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"

from flask import Flask, request, jsonify
from flask_cors import CORS
from train_matchboxnet_protonet import MatchboxNetEncoder
from pydub import AudioSegment
from pathlib import Path
import torchaudio
import torch
import uuid
import json
import numpy as np
from sklearn.cluster import KMeans
from vosk import Model as VoskModel, KaldiRecognizer
import wave
import subprocess

app = Flask(__name__)
CORS(app)

if not os.path.exists("matchbox_model.pt"):
    subprocess.run(["python", "train_matchboxnet_protonet.py"])

vosk_model = VoskModel("model")

speaker_model = MatchboxNetEncoder()
speaker_model.load_state_dict(torch.load("matchbox_model.pt", map_location="cpu")["model"])
speaker_model.eval()

TEMP_VECTORS_FILE = "registered_vectors.json"
FINAL_VECTOR_FILE = "registered_speaker.json"
KEYWORD_VECTOR_FILE = "registered_keyword_vectors.json"
PROTO_LABEL_MAP_FILE = "proto_label_map.json"


def preprocess_waveform(path):
    audio = AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)
    audio.export(path, format="wav")
    waveform, _ = torchaudio.load(path)  # [1, T] expected

    print("[DEBUG] waveform shape before:", waveform.shape, flush=True)

    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, 1, T]
    elif waveform.dim() == 3 and waveform.shape[:2] == (1, 1):
        pass  # already [1, 1, T]
    else:
        raise RuntimeError(f"[ERROR] Unexpected waveform shape: {waveform.shape}")

    print("[DEBUG] waveform shape after:", waveform.shape, flush=True)
    return waveform



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
        waveform = preprocess_waveform(temp_filename)
        with torch.no_grad():
            embed = speaker_model(waveform)  # [1, 1, T]
            weights = torch.softmax(torch.norm(embed, dim=1), dim=-1)
            speaker_vec = torch.sum(embed * weights.unsqueeze(1), dim=-1).squeeze().numpy()
        speaker_vec = speaker_vec / np.linalg.norm(speaker_vec)
        vectors = []
        if os.path.exists(TEMP_VECTORS_FILE):
            with open(TEMP_VECTORS_FILE, "r") as f:
                vectors = json.load(f)
        vectors.append(speaker_vec.tolist())
        with open(TEMP_VECTORS_FILE, "w") as f:
            json.dump(vectors, f)
        if len(vectors) == 4:
            avg = np.mean(np.array(vectors), axis=0)
            avg = avg / np.linalg.norm(avg)
            with open(FINAL_VECTOR_FILE, "w") as f:
                json.dump(avg.tolist(), f)
            os.remove(TEMP_VECTORS_FILE)
            return jsonify({"message": "화자 등록 완료 (4/4)"})
        return jsonify({"message": f"등록 {len(vectors)}/4 완료"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_filename)


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
        audio = AudioSegment.from_file(str(raw_path)).set_frame_rate(16000).set_channels(1)
        audio.export(str(fixed_path), format="wav")
        os.remove(str(raw_path))
        waveform = preprocess_waveform(str(fixed_path))
        with torch.no_grad():
            embed = speaker_model(waveform)  # [1, 1, T]
            weights = torch.softmax(torch.norm(embed, dim=1), dim=-1)
            vec = torch.sum(embed * weights.unsqueeze(1), dim=-1).squeeze().numpy()
        vec = vec / np.linalg.norm(vec)
        all_vectors = {}
        if os.path.exists(KEYWORD_VECTOR_FILE):
            with open(KEYWORD_VECTOR_FILE, "r") as f:
                all_vectors = json.load(f)
        all_vectors.setdefault(keyword, []).append(vec.tolist())
        with open(KEYWORD_VECTOR_FILE, "w") as f:
            json.dump(all_vectors, f, indent=2)
        if len(all_vectors[keyword]) == 6:
            subprocess.run(["python", "train_matchboxnet_protonet.py"])
        return jsonify({"message": f"{keyword} 키워드 {index}/6 저장 완료"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(raw_path):
            os.remove(str(raw_path))


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
        waveform = preprocess_waveform(temp_filename)
        with torch.no_grad():
            raw_embed = speaker_model(waveform)
            time_weights = torch.softmax(torch.norm(raw_embed, dim=1), dim=-1)
            weighted_embed = torch.sum(raw_embed * time_weights.unsqueeze(1), dim=-1).squeeze().numpy()
        speaker_vec = weighted_embed / np.linalg.norm(weighted_embed)

        if not os.path.exists(FINAL_VECTOR_FILE):
            return jsonify({"error": "화자 벡터 없음"}), 403
        with open(FINAL_VECTOR_FILE, "r") as f:
            ref_vec = np.array(json.load(f))
        sim_sp = float(np.dot(speaker_vec, ref_vec))

        if not os.path.exists(KEYWORD_VECTOR_FILE):
            return jsonify({"error": "키워드 벡터 없음"}), 403
        with open(KEYWORD_VECTOR_FILE, "r") as f:
            label_map = json.load(f)

        best_keyword = None
        best_score = 0
        proto_label_map = {}
        for keyword, vecs in label_map.items():
            vecs_np = []
            for v in vecs:
                v_np = np.array(v)
                if v_np.ndim == 1:
                    vecs_np.append(v_np / np.linalg.norm(v_np))
                else:
                    energy = np.linalg.norm(v_np, axis=-1)
                    weights = np.exp(energy) / np.sum(np.exp(energy))
                    weighted = np.sum(v_np * weights[..., None], axis=0)
                    weighted /= np.linalg.norm(weighted)
                    vecs_np.append(weighted)
            vecs_np = np.array(vecs_np)
            vecs_np = vecs_np / np.linalg.norm(vecs_np, axis=1, keepdims=True)
            n_proto = min(3, len(vecs_np))
            kmeans = KMeans(n_clusters=n_proto, random_state=0).fit(vecs_np)
            prototypes = kmeans.cluster_centers_
            proto_label_map[keyword] = prototypes.tolist()
            scores = np.dot(prototypes, speaker_vec)
            score = float(np.max(scores))
            if score > best_score:
                best_score = score
                best_keyword = keyword
        with open(PROTO_LABEL_MAP_FILE, "w") as f:
            json.dump(proto_label_map, f, indent=2)

        sim_kw = best_score
        total_score = sim_sp + sim_kw
        if total_score < 1.5:
            return jsonify({
                "error": "인증 실패",
                "sim_sp": round(sim_sp, 4),
                "sim_kw": round(sim_kw, 4)
            }), 403

        # ✅ wave.open은 with 블록으로 열기
        transcript = ""
        with wave.open(temp_filename, "rb") as wf:
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(json.loads(rec.Result()))
            results.append(json.loads(rec.FinalResult()))
            transcript = " ".join([r.get("text", "") for r in results if r.get("text")])

        return jsonify({
            "text": transcript,
            "speaker_similarity": round(sim_sp, 4),
            "keyword": best_keyword,
            "keyword_similarity": round(sim_kw, 4),
            "s_total": round(total_score, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                print("[WARN] 파일 삭제 실패:", e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
