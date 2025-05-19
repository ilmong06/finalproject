import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# 🔵 모델 로딩
processor_ko = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
model_ko = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

SPEAKER_VECTOR_FILE = 'model/registered_speaker.json'
KEYWORD_VECTOR_FILE = 'model/registered_keyword_vectors.json'

# 🔵 STT 및 인증 처리
def process_stt(file, lang):
    """
    음성파일을 받아서 화자 인증 + 키워드 인증 + STT 결과 반환
    """
    # 1. 화자 인증
    speaker_passed = check_speaker(file)
    if not speaker_passed:
        return {"error": "화자 인증 실패"}

    # 2. 키워드 인증
    keyword_passed = check_keyword(file)
    if not keyword_passed:
        return {"error": "키워드 인증 실패"}

    # 3. STT 변환
    stt_text = stt(file, lang)
    return {"result": stt_text}

# 🔵 화자 인증
def check_speaker(file):
    if not os.path.exists(SPEAKER_VECTOR_FILE):
        return False

    with open(SPEAKER_VECTOR_FILE, 'r') as f:
        registered = json.load(f)

    # 실제로는 음성에서 speaker vector 추출해야 함
    dummy_vector = np.random.rand(512)

    for uuid, registered_vector in registered.items():
        dist = cosine(dummy_vector, np.array(registered_vector))
        if dist < 0.5:  # threshold
            return True

    return False

# 🔵 키워드 인증
def check_keyword(file):
    if not os.path.exists(KEYWORD_VECTOR_FILE):
        return False

    with open(KEYWORD_VECTOR_FILE, 'r') as f:
        registered = json.load(f)

    # 실제로는 음성에서 keyword vector 추출해야 함
    dummy_vector = np.random.rand(512)

    for keyword, registered_vector in registered.items():
        dist = cosine(dummy_vector, np.array(registered_vector))
        if dist < 0.5:  # threshold
            return True

    return False

# 🔵 STT
def stt(file, lang):
    waveform, sample_rate = torchaudio.load(file)
    input_values = processor_ko(waveform.squeeze(), return_tensors="pt", sampling_rate=sample_rate).input_values
    logits = model_ko(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_ko.batch_decode(predicted_ids)[0]
    return transcription
