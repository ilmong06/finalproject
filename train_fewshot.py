import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

print("🔁 사용자 정의 키워드 기반 Few-shot KWS 학습 시작...")

# ✅ ResNet 기반 키워드 모델
from torchvision.models import resnet18
class ResKeywordNet(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet18(pretrained=False)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # fc 제거
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.features(x)      # [B, 512, 1, 1]
        x = x.view(x.size(0), -1) # [B, 512]
        return self.fc(x)

# ✅ Mel-Spectrogram 변환
def extract_mel(filepath, target_length=232):
    waveform, sr = torchaudio.load(filepath)
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80,
        n_fft=1024,
        hop_length=160
    )(waveform)
    if mel.shape[-1] < target_length:
        mel = F.pad(mel, (0, target_length - mel.shape[-1]))
    elif mel.shape[-1] > target_length:
        mel = mel[:, :, :target_length]
    return mel.unsqueeze(0)  # [1, 1, n_mels, T]

# ✅ 데이터 로딩
def load_data(base_path="data/custom"):
    X, y, label_map, file_paths = [], [], {}, {}
    for idx, keyword_dir in enumerate(sorted(Path(base_path).glob("*"))):
        if keyword_dir.is_dir():
            keyword = keyword_dir.name
            label_map[idx] = keyword
            file_paths[keyword] = []
            for file in keyword_dir.glob("*.wav"):
                mel = extract_mel(file)  # [1, 1, n_mels, T]
                X.append(mel)
                y.append(idx)
                file_paths[keyword].append(file)
    return torch.cat(X), torch.tensor(y), label_map, file_paths

# ✅ 학습 준비
X, y, label_map, file_paths = load_data()
print(f"📁 키워드 {len(label_map)}개, 총 {len(y)}개 샘플")

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ResKeywordNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ✅ 학습 루프
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

# ✅ 키워드 평균 임베딩 저장
model.eval()
keyword_embeddings = {}
with torch.no_grad():
    for keyword, paths in file_paths.items():
        vecs = []
        for path in paths:
            mel = extract_mel(path)
            emb = model(mel).squeeze()
            vecs.append(emb)
        mean_vec = torch.stack(vecs).mean(dim=0)
        keyword_embeddings[keyword] = (mean_vec / mean_vec.norm()).tolist()

# ✅ 저장
torch.save({"model": model.state_dict()}, "fewshot_model.pt")
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(keyword_embeddings, f)

print("✅ 모델 및 키워드 벡터 저장 완료 → fewshot_model.pt, label_map.json")
