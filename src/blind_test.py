import gzip
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

BASE = "/home/gulsum/Masaüstü/Dizgen_project"

def read_sequences(fastq_path, max_reads=20):
    sequences = []
    opener = gzip.open if fastq_path.endswith('.gz') else open
    with opener(fastq_path, 'rt') as f:
        while len(sequences) < max_reads:
            header = f.readline().strip()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()
            f.readline()
            if 'N' not in seq and len(seq) >= 250:
                sequences.append((header[1:].split()[0], seq))
    return sequences

def dna_to_fcgr(sequence, k=4):
    matrix_size = 2 ** k
    matrix = np.zeros((matrix_size, matrix_size), dtype=float)
    coords = {'A': (0,0), 'T': (1,1), 'G': (1,0), 'C': (0,1)}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k].upper()
        if any(base not in coords for base in kmer):
            continue
        x, y = 0, 0
        for base in kmer:
            cx, cy = coords[base]
            x = x // 2 + cx * (matrix_size // 2)
            y = y // 2 + cy * (matrix_size // 2)
        matrix[y][x] += 1
    if matrix.max() > 0:
        matrix = matrix / matrix.max()
    return matrix

def seq_to_image(sequence, size=64):
    r = dna_to_fcgr(sequence, k=3)
    g = dna_to_fcgr(sequence, k=4)
    b = dna_to_fcgr(sequence, k=4)
    r = np.array(Image.fromarray((r*255).astype(np.uint8)).resize((size,size)))
    g = np.array(Image.fromarray((g*255).astype(np.uint8)).resize((size,size)))
    b = np.array(Image.fromarray((b*255).astype(np.uint8)).resize((size,size)))
    return Image.fromarray(np.stack([r,g,b], axis=2).astype(np.uint8))

class HybridModel(nn.Module):
    def __init__(self, num_bacterium_classes, num_resistance_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4))
        )
        self.cnn_fc = nn.Linear(128*4*4, 256)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0)
        self.fusion = nn.Sequential(
            nn.Linear(256 + self.vit.num_features, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU()
        )
        self.bacterium_head = nn.Linear(256, num_bacterium_classes)
        self.resistance_head = nn.Linear(256, num_resistance_classes)

    def forward(self, x):
        c = self.cnn(x).view(x.size(0), -1)
        c = self.cnn_fc(c)
        v = self.vit(nn.functional.interpolate(x, (224,224), mode='bilinear', align_corners=False))
        f = self.fusion(torch.cat([c, v], dim=1))
        return self.bacterium_head(f), self.resistance_head(f)

# Modeli yükle
df = pd.read_csv(f"{BASE}/data/3_labels/amr_labels.csv")
le_b = LabelEncoder().fit(df['bacterium'])
le_r = LabelEncoder().fit(df['resistance_class'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(len(le_b.classes_), len(le_r.classes_))
model.load_state_dict(torch.load(f"{BASE}/models/dizgen_hybrid_v1.pth", map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Kör test - ERR772063_2 (modelin hiç görmediği veri)
print("=" * 50)
print("KÖR TEST BAŞLIYOR")
print("Veri: ERR772063_2.fastq.gz")
print("=" * 50)

sequences = read_sequences(f"{BASE}/data/1_raw/ERR772063_2.fastq.gz", max_reads=20)
print(f"Test dizisi: {len(sequences)}\n")

for seq_id, seq in sequences:
    img = seq_to_image(seq)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        b_out, r_out = model(tensor)
    b_pred = le_b.inverse_transform([torch.argmax(b_out).item()])[0]
    r_pred = le_r.inverse_transform([torch.argmax(r_out).item()])[0]
    print(f"Dizi: {seq_id}")
    print(f"  Bakteri : {b_pred}")
    print(f"  Direnç  : {r_pred}")
    print()

print("Kör test tamamlandı!")