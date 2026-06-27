import re
import os

# Antibiyotik kısaltma eşleşmeleri
antibiotic_map = {
    'AMK': 'Aminoglycoside', 'KAN': 'Aminoglycoside', 'NEO': 'Aminoglycoside',
    'STR': 'Aminoglycoside', 'G418': 'Aminoglycoside',
    'SPT': 'Spectinomycin', 'EDN': 'Aminoglycoside',
    'LZD': 'Oxazolidinone', 'TET': 'Tetracycline',
    'CLI': 'Lincosamide', 'ERY': 'Macrolide'
}

def parse_header(header):
    # Bakteri adını çıkar
    bacterium_match = re.search(r'\[(.+?)\]', header)
    bacterium = bacterium_match.group(1) if bacterium_match else 'Unknown'
    bacterium = bacterium.split(' str.')[0].split(' subsp.')[0].strip()
    
    # ARO kodunu çıkar
    aro_match = re.search(r'ARO:(\d+)', header)
    aro = aro_match.group(1) if aro_match else 'Unknown'
    
    # Kısa ismi çıkar
    short_match = re.search(r'ARO:\d+\|(\S+)', header)
    short_name = short_match.group(1) if short_match else 'Unknown'
    
    # Direnç sınıfını bul
    resistance = 'Unknown'
    for key, value in antibiotic_map.items():
        if key in short_name.upper():
            resistance = value
            break
    
    return aro, short_name, bacterium, resistance

def process_card_rrna(input_fasta, output_fasta, output_labels):
    sequences = {}
    current_id = None
    current_meta = None
    
    with open(input_fasta, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                aro, short_name, bacterium, resistance = parse_header(line)
                current_id = f"ARO_{aro}|{short_name}|{bacterium}"
                current_meta = (aro, short_name, bacterium, resistance)
            elif current_id:
                sequences[current_id] = (line, current_meta)
    
    with open(output_fasta, 'w') as f_fasta, open(output_labels, 'w') as f_labels:
        f_labels.write("gene_id,gene_name,bacterium,resistance_class\n")
        for seq_id, (sequence, meta) in sequences.items():
            aro, short_name, bacterium, resistance = meta
            f_fasta.write(f">{seq_id}\n{sequence}\n")
            f_labels.write(f"ARO_{aro},{short_name},{bacterium},{resistance}\n")
    
    print(f"Toplam dizi: {len(sequences)}")
    print(f"FASTA: {output_fasta}")
    print(f"Etiketler: {output_labels}")

if __name__ == "__main__":
    input_fasta = "/home/gulsum/Masaüstü/Dizgen_project/data/1_raw/amr_rRNA_variants.fasta"
    output_fasta = "/home/gulsum/Masaüstü/Dizgen_project/data/2_processed/cleaned_amr_rRNA.fasta"
    output_labels = "/home/gulsum/Masaüstü/Dizgen_project/data/3_labels/amr_labels.csv"
    
    print("CARD rRNA verisi işleniyor...")
    process_card_rrna(input_fasta, output_fasta, output_labels)