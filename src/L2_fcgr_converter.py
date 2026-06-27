import numpy as np
from PIL import Image
import os

def dna_to_fcgr(sequence, k=4, image_size=64):
    matrix_size = 2 ** k
    matrix = np.zeros((matrix_size, matrix_size), dtype=float)
    coords = {'A': (0, 0), 'T': (1, 1), 'G': (1, 0), 'C': (0, 1)}
    
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

def fcgr_to_rgb(sequence, image_size=64):
    r_channel = dna_to_fcgr(sequence, k=3, image_size=image_size)
    g_channel = dna_to_fcgr(sequence, k=4, image_size=image_size)
    b_channel = dna_to_fcgr(sequence, k=4, image_size=image_size)
    
    size = image_size
    r = np.array(Image.fromarray((r_channel * 255).astype(np.uint8)).resize((size, size)))
    g = np.array(Image.fromarray((g_channel * 255).astype(np.uint8)).resize((size, size)))
    b = np.array(Image.fromarray((b_channel * 255).astype(np.uint8)).resize((size, size)))
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def process_fasta(input_fasta, output_dir, image_size=64):
    os.makedirs(output_dir, exist_ok=True)
    sequences = {}
    current_id = None
    
    with open(input_fasta, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]
            elif current_id:
                sequences[current_id] = line
    
    print(f"Toplam dizi: {len(sequences)}")
    
    for seq_id, sequence in sequences.items():
        print(f"İşleniyor: {seq_id}")
        rgb = fcgr_to_rgb(sequence, image_size)
        safe_id = seq_id.replace('|', '_').replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f"{safe_id}.png")
        img = Image.fromarray(rgb.astype(np.uint8))
        img.save(output_path)
        print(f"Kaydedildi: {safe_id}.png")
    
    print(f"Tamamlandı! {len(sequences)} görüntü oluşturuldu.")

if __name__ == "__main__":
    print("🚀 DizGen Tasarım Hattı: Layer 2 — FCGR Dönüşümü Başlatıldı...")
    
    # 1. Yeni klasör yapımıza göre giriş ve çıkış yollarını tanımlıyoruz
    input_fasta = "/home/gulsum/Masaüstü/Dizgen_project/data/2_processed/cleaned_amr_rRNA.fasta"
    output_dir = "/home/gulsum/Masaüstü/Dizgen_project/data/4_fcgr_images/rRNA_images"
    
    # 2. Dönüşüm fonksiyonunu yeni yollarla tetikliyoruz
    process_fasta(input_fasta, output_dir, image_size=64)
    
    print(f"✅ İşlem tamamlandı! Üretilen 64x64 boyutundaki RGB resimler şuraya kaydedildi:\n👉 {output_dir}")


if __name__ == "__main__":
    print("Script başladı...")
    input_fasta = "/home/gulsum/Masaüstü/Dizgen_project/data/2_processed/cleaned_amr_protein.fasta"
    output_dir = "/home/gulsum/Masaüstü/Dizgen_project/data/4_fcgr_images/protein_images"
    process_fasta(input_fasta, output_dir, image_size=64)



