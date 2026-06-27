import os
import sys
import time
import gzip
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.append('/home/gulsum/Masaüstü/Dizgen_project/src')

from L2_fcgr_converter import fcgr_to_rgb
from L4_search_space_reduction import reduce_search_space, get_blast_subset
from L5_blast_integration import run_blast, parse_blast_results

BASE = "/home/gulsum/Masaüstü/Dizgen_project"

def fastq_to_fasta(fastq_path, output_fasta, max_reads=100):
    """FASTQ'dan temiz dizileri FASTA'ya çevirir"""
    count = 0
    opener = gzip.open if fastq_path.endswith('.gz') else open
    
    with opener(fastq_path, 'rt') as f_in, open(output_fasta, 'w') as f_out:
        while count < max_reads:
            header = f_in.readline().strip()
            if not header:
                break
            sequence = f_in.readline().strip()
            f_in.readline()
            f_in.readline()
            
            if 'N' not in sequence and len(sequence) >= 250:
                seq_id = header[1:].split()[0]
                f_out.write(f">{seq_id}\n{sequence}\n")
                count += 1
    
    print(f"{count} dizi FASTA'ya çevrildi")
    return output_fasta

def sequence_to_fcgr_image(sequence, image_size=64):
    """DNA dizisini FCGR görüntüsüne çevirir"""
    rgb = fcgr_to_rgb(sequence, image_size)
    return Image.fromarray(rgb.astype(np.uint8))

def run_pipeline(input_fastq):
    print("=" * 50)
    print("DizGen Hibrit Pipeline Başlatılıyor")
    print("=" * 50)
    
    # 1. FASTQ → FASTA
    print("\n[1/5] FASTQ işleniyor...")
    temp_fasta = f"{BASE}/data/temp_query.fasta"
    fastq_to_fasta(input_fastq, temp_fasta, max_reads=100)
    
    # 2. FASTA dizilerini oku
    print("\n[2/5] Diziler okunuyor...")
    sequences = {}
    current_id = None
    with open(temp_fasta, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
            elif current_id:
                sequences[current_id] = line
    print(f"Toplam dizi: {len(sequences)}")
    
    # 3. FCGR görüntüleri oluştur ve model tahmini yap
    print("\n[3/5] FCGR + Model tahmini...")
    
    from L3_hybrid_classifier import HybridModel
    
    label_file = f"{BASE}/data/3_labels/amr_labels.csv"
    model_path = f"{BASE}/models/dizgen_hybrid_v1.pth"
    
    df = pd.read_csv(label_file)
    le_bacterium = LabelEncoder()
    le_resistance = LabelEncoder()
    le_bacterium.fit(df['bacterium'])
    le_resistance.fit(df['resistance_class'])
    
    model = HybridModel(len(le_bacterium.classes_), len(le_resistance.classes_))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    results = []
    for seq_id, sequence in list(sequences.items())[:10]:
        img = sequence_to_fcgr_image(sequence)
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            bact_out, res_out = model(img_tensor)
        
        bact_pred = le_bacterium.inverse_transform([torch.argmax(bact_out).item()])[0]
        res_pred = le_resistance.inverse_transform([torch.argmax(res_out).item()])[0]
        blast_subset = get_blast_subset(bact_pred)
        
        results.append({
            'seq_id': seq_id,
            'bacterium': bact_pred,
            'resistance': res_pred,
            'blast_subset': blast_subset
        })
        print(f"  {seq_id[:30]}... → {bact_pred} | {res_pred} | Alt küme: {blast_subset}")
    
    # 4. BLAST doğrulama
    print("\n[4/5] BLAST doğrulama...")
    blast_start = time.time()
    blast_output = f"{BASE}/data/blast_results.txt"
    
    most_common_subset = max(set([r['blast_subset'] for r in results]), 
                             key=[r['blast_subset'] for r in results].count)
    
    result_file, blast_time = run_blast(temp_fasta, most_common_subset, blast_output)
    
    # 5. Sonuçları raporla
    print("\n[5/5] Sonuçlar:")
    print("=" * 50)
    for r in results:
        print(f"Dizi: {r['seq_id'][:40]}")
        print(f"  Bakteri: {r['bacterium']}")
        print(f"  Direnç:  {r['resistance']}")
        print(f"  BLAST alt kümesi: {r['blast_subset']}")
        print()
    
    print(f"BLAST süresi: {blast_time:.2f} saniye")
    print(f"İşlenen dizi: {len(results)}")
    print("Pipeline tamamlandı!")

if __name__ == "__main__":
    input_fastq = "/home/gulsum/Masaüstü/Dizgen_project/data/1_raw/ERR772063_1.fastq.gz"
    run_pipeline(input_fastq)