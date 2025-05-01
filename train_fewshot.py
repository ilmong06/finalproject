import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
print("🔁 Conformer + PrototypicalNet 기반 Few-shot KWS 학습 시작...")

# ✅ segment 분할 함수
def segment_waveform(waveform, sample_rate=16000, segment_ms=1000):
    segment_samples = int(sample_rate * segment_ms / 1000)
    segments = []
    for i in range(0, waveform.shape[1], segment_samples):
        chunk = waveform[:, i:i+segment_samples]
        if chunk.shape[1] == segment_samples:
            energy = chunk.pow(2).mean().item()
            if energy > 1e-5:
                segments.append(chunk)
    return segments

# ✅ Conformer Encoder 정의
from torchaudio.models.conformer import Conformer

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
        mel = self.mel_transform(waveform)
        mel = mel.transpose(1, 2)
        lengths = torch.full((mel.shape[0],), mel.shape[1], dtype=torch.long)
        x, _ = self.encoder(mel, lengths)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

# ✅ 학습용: gradient 추적 O
def get_segment_avg_train(waveform, model):
    segments = segment_waveform(waveform)
    embs = []
    for seg in segments:
        if seg.ndim == 1:
            seg = seg.unsqueeze(0)  # [1, T]
        emb = model(seg).squeeze()
        emb = emb / emb.norm()
        embs.append(emb)
    if not embs:
        return torch.zeros(144, requires_grad=True)
    return torch.stack(embs).mean(dim=0)

# ✅ 평가용: gradient 추적 X
def get_segment_avg_eval(waveform, model):
    segments = segment_waveform(waveform)
    embs = []
    for seg in segments:
        if seg.ndim == 1:
            seg = seg.unsqueeze(0)  # [1, T]
        with torch.no_grad():
            emb = model(seg).squeeze()
            emb = emb / emb.norm()
            embs.append(emb)
    if not embs:
        return torch.zeros(144)
    return torch.stack(embs).mean(dim=0)

# ✅ 데이터 로딩
def load_data(base_path="data/custom"):
    X, y, label_map = [], [], {}
    for idx, keyword_dir in enumerate(sorted(Path(base_path).glob("*"))):
        if keyword_dir.is_dir():
            label_map[idx] = keyword_dir.name
            for file in keyword_dir.glob("*.wav"):
                waveform, sr = torchaudio.load(file)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                X.append(waveform)
                y.append(idx)
    return X, torch.tensor(y), label_map

# ✅ 학습
X, y, label_map = load_data()
model = ConformerEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"📁 키워드 {len(label_map)}개, 총 {len(y)}개 샘플")

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    embeddings = [get_segment_avg_train(x, model) for x in X]
    embeddings = torch.stack(embeddings)
    loss = 0
    for cls in torch.unique(y):
        idxs = (y == cls).nonzero(as_tuple=False).squeeze()
        if idxs.ndim == 0 or idxs.numel() < 2:
            continue
        support = embeddings[idxs[:-1]]
        query = embeddings[idxs[-1]].unsqueeze(0)
        proto = support.mean(dim=0, keepdim=True)
        dist = F.cosine_similarity(query, proto)
        loss += 1 - dist.mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ✅ 벡터 저장
model.eval()
keyword_embeddings = {}
with torch.no_grad():
    for cls in torch.unique(y):
        keyword = label_map[cls.item()]
        idxs = (y == cls).nonzero(as_tuple=False).squeeze()
        reps = torch.stack([get_segment_avg_eval(X[i], model) for i in idxs])
        proto = reps.mean(dim=0)
        keyword_embeddings[keyword] = (proto / proto.norm()).tolist()

torch.save({"model": model.state_dict()}, "fewshot_model.pt")
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(keyword_embeddings, f)
print("✅ 모델 및 키워드 벡터 저장 완료 → fewshot_model.pt, label_map.json")

# ✅ 시각화 (선택적으로 저장)
vecs = torch.stack([get_segment_avg_eval(x, model) for x in X])
labels = [label_map[i.item()] for i in y]
proj = TSNE(n_components=2, perplexity=5, init='random').fit_transform(vecs.detach().numpy())
plt.figure(figsize=(8, 6))
for label in set(labels):
    inds = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(proj[inds, 0], proj[inds, 1], label=label)
plt.legend()
plt.title("Keyword Embedding (Conformer + t-SNE)")
plt.savefig("tsne_result.png")  # 또는 plt.show() 원할 경우 사용
