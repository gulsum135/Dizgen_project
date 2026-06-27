import re
import os

# Direnç sınıfı anahtar kelime eşleşmeleri
resistance_map = {
    'TEM': 'Beta-lactam', 'SHV': 'Beta-lactam', 'CTX': 'Beta-lactam',
    'NDM': 'Beta-lactam', 'KPC': 'Beta-lactam', 'OXA': 'Beta-lactam',
    'VIM': 'Beta-lactam', 'IMP': 'Beta-lactam', 'ACT': 'Beta-lactam',
    'CMY': 'Beta-lactam', 'CblA': 'Beta-lactam', 'CARB': 'Beta-lactam',
    'LRA': 'Beta-lactam', 'ACT': 'Beta-lactam',
    'tet': 'Tetracycline', 'TET': 'Tetracycline',
    'Erm': 'Macrolide', 'mph': 'Macrolide', 'mef': 'Macrolide',
    'aac': 'Aminoglycoside', 'aph': 'Aminoglycoside', 'ant': 'Aminoglycoside',
    'van': 'Glycopeptide', 'Van': 'Glycopeptide',
    'sul': 'Sulfonamide', 'Sul': 'Sulfonamide',
    'dfr': 'Trimethoprim', 'Dfr': 'Trimethoprim',
    'qnr': 'Fluoroquinolone', 'Qnr': 'Fluoroquinolone',
    'mcr': 'Colistin', 'MCR': 'Colistin',
    'cat': 'Chloramphenicol', 'Cat': 'Chloramphenicol',
}

def get_resistance_class(gene_name):
    for key, value in resistance_map.items():
        if key in gene_name:
            return value
    return 'Other'

def parse_header(header):
    # Bakteri adını çıkar
    bacterium_match = re.search(r'\[(.+?)\]', header)
    bacterium = bacterium_match.group(1) if bacterium_match else 'Unknown'
    bacterium = bacterium.split(' str.')[0].split(' subsp.')[0].strip()
    
    # ARO kodunu çıkar
    aro_match = re.search(r'ARO:(\d+)', header)
    aro = aro_match.group(1) if aro_match else 'Unknown'
    
    # Gen adını çıkar
    gene_match = re.search(r'ARO:\d+\|(.+?)\s*\[', header)
    gene_name = gene_match.group(1).strip() if gene_match else 'Unknown'
    
    resistance = get_resistance_class(gene_name)
    
    return aro, gene_name, bacterium, resistance

def process_protein_fasta(input_fasta, output_fasta, output_labels):
    sequences = {}
    current_id = None
    current_meta = None
    
    with open(input_fasta, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                aro, gene_name, bacterium, resistance = parse_header(line)
                current_id = f"ARO_{aro}|{gene_name}|{bacterium}"
                current_meta = (aro, gene_name, bacterium, resistance)
            elif current_id and line:
                if current_id in sequences:
                    sequences[current_id] = (sequences[current_id][0] + line, current_meta)
                else:
                    sequences[current_id] = (line, current_meta)
    
    with open(output_fasta, 'w') as f_fasta, open(output_labels, 'w') as f_labels:
        f_labels.write("gene_id,gene_name,bacterium,resistance_class\n")
        for seq_id, (sequence, meta) in sequences.items():
            aro, gene_name, bacterium, resistance = meta
            f_fasta.write(f">{seq_id}\n{sequence}\n")
            safe_gene = gene_name.replace(',', ';')
            safe_bact = bacterium.replace(',', ';')
            f_labels.write(f"ARO_{aro},{safe_gene},{safe_bact},{resistance}\n")
    
    print(f"Toplam dizi: {len(sequences)}")
    print(f"FASTA: {output_fasta}")
    print(f"Etiketler: {output_labels}")

if __name__ == "__main__":
    input_fasta = "/home/gulsum/Masaüstü/Dizgen_project/data/1_raw/amr_protein_variants.fasta"
    output_fasta = "/home/gulsum/Masaüstü/Dizgen_project/data/2_processed/cleaned_amr_protein.fasta"
    output_labels = "/home/gulsum/Masaüstü/Dizgen_project/data/3_labels/amr_protein_labels.csv"
    
    print("Protein direnç geni verisi işleniyor...")
    process_protein_fasta(input_fasta, output_fasta, output_labels)
