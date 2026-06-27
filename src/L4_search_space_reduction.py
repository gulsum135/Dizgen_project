import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Bakteri → BLAST alt küme eşleşmesi
BACTERIA_TO_SUBSET = {
    'Escherichia coli': 'enterobacteriaceae',
    'Klebsiella pneumoniae': 'enterobacteriaceae',
    'Salmonella enterica': 'enterobacteriaceae',
    'Mycobacterium tuberculosis': 'mycobacterium',
    'Mycobacteroides abscessus': 'mycobacterium',
    'Staphylococcus aureus': 'staphylococcus',
    'Streptococcus pneumoniae': 'streptococcus',
    'Neisseria gonorrhoeae': 'neisseria',
    'Acinetobacter': 'acinetobacter',
    'Pseudomonas': 'pseudomonas'
}

def get_blast_subset(predicted_bacterium):
    for key, subset in BACTERIA_TO_SUBSET.items():
        if key.lower() in predicted_bacterium.lower():
            return subset
    return 'all'

def predict_from_image(image_path, model, transform, device, le_bacterium, le_resistance):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        bact_out, res_out = model(image)
    
    bact_pred = torch.argmax(bact_out, dim=1).item()
    res_pred = torch.argmax(res_out, dim=1).item()
    
    bacterium = le_bacterium.inverse_transform([bact_pred])[0]
    resistance = le_resistance.inverse_transform([res_pred])[0]
    blast_subset = get_blast_subset(bacterium)
    
    return bacterium, resistance, blast_subset

def reduce_search_space(image_path, model_path, label_file, model_class, device='cpu'):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    df = pd.read_csv(label_file)
    le_bacterium = LabelEncoder()
    le_resistance = LabelEncoder()
    le_bacterium.fit(df['bacterium'])
    le_resistance.fit(df['resistance_class'])
    
    num_bact = len(le_bacterium.classes_)
    num_res = len(le_resistance.classes_)
    
    model = model_class(num_bact, num_res)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    bacterium, resistance, blast_subset = predict_from_image(
        image_path, model, transform, device, le_bacterium, le_resistance
    )
    
    print(f"Tahmin edilen bakteri: {bacterium}")
    print(f"Tahmin edilen direnç: {resistance}")
    print(f"BLAST alt kümesi: {blast_subset}")
    
    return bacterium, resistance, blast_subset

if __name__ == "__main__":
    print("Search Space Reduction test ediliyor...")
    print("Bu modül main.py tarafından çağrılacak.")