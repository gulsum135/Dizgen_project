import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FCGRDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        df = pd.read_csv(label_file)
        self.le_bacterium = LabelEncoder()
        self.le_resistance = LabelEncoder()
        df['bacterium_enc'] = self.le_bacterium.fit_transform(df['bacterium'])
        df['resistance_enc'] = self.le_resistance.fit_transform(df['resistance_class'])
        
        self.samples = []
        for _, row in df.iterrows():
            gene_id = row['gene_id']
            for fname in os.listdir(image_dir):
                if fname.startswith(gene_id):
                    self.samples.append({
                        'path': os.path.join(image_dir, fname),
                        'bacterium': row['bacterium_enc'],
                        'resistance': row['resistance_enc']
                    })
                    break
        
        print(f"Yüklenen örnek: {len(self.samples)}")
        print(f"Bakteri sınıfı sayısı: {len(self.le_bacterium.classes_)}")
        print(f"Direnç sınıfı sayısı: {len(self.le_resistance.classes_)}")
        
        self.num_bacterium = len(self.le_bacterium.classes_)
        self.num_resistance = len(self.le_resistance.classes_)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, sample['bacterium'], sample['resistance']

class HybridModel(nn.Module):
    def __init__(self, num_bacterium_classes, num_resistance_classes):
        super(HybridModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.cnn_fc = nn.Linear(128 * 4 * 4, 256)
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        vit_out = self.vit.num_features
        self.fusion = nn.Sequential(
            nn.Linear(256 + vit_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.bacterium_head = nn.Linear(256, num_bacterium_classes)
        self.resistance_head = nn.Linear(256, num_resistance_classes)
    
    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = self.cnn_fc(cnn_out)
        x_vit = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        vit_out = self.vit(x_vit)
        combined = torch.cat([cnn_out, vit_out], dim=1)
        fused = self.fusion(combined)
        return self.bacterium_head(fused), self.resistance_head(fused)

def train():
    image_dir = "/home/gulsum/Masaüstü/Dizgen_project/data/4_fcgr_images/protein_images"
    label_file = "/home/gulsum/Masaüstü/Dizgen_project/data/3_labels/amr_protein_labels.csv"
    model_save = "/home/gulsum/Masaüstü/Dizgen_project/models/dizgen_protein_v1.pth"
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    dataset = FCGRDataset(image_dir, label_file, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = HybridModel(dataset.num_bacterium, dataset.num_resistance)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\nEğitim başlıyor...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for images, bact_labels, res_labels in dataloader:
            images = images.to(device)
            bact_labels = bact_labels.to(device)
            res_labels = res_labels.to(device)
            optimizer.zero_grad()
            bact_out, res_out = model(images)
            loss = criterion(bact_out, bact_labels) + criterion(res_out, res_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/30 | Loss: {total_loss/len(dataloader):.4f}")
    
    torch.save(model.state_dict(), model_save)
    print(f"Model kaydedildi: {model_save}")

if __name__ == "__main__":
    print("Protein model eğitimi başlıyor...")
    train()