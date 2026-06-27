import subprocess
import os
import time

BLAST_DB = "/usr/bin"
BLAST_16S_DB = "/home/gulsum/İndirilenler/16S_ribosomal_RNA"

SUBSET_TAXIDS = {
    'enterobacteriaceae': '91347',
    'mycobacterium': '1763',
    'staphylococcus': '1279',
    'streptococcus': '1301',
    'neisseria': '482',
    'acinetobacter': '469',
    'pseudomonas': '286',
    'all': None
}

def run_blast(query_fasta, blast_subset='all', output_file='blast_results.txt'):
    blastn = os.path.join(BLAST_DB, 'blastn')
    
    cmd = [
        blastn,
        '-query', query_fasta,
        '-db', BLAST_16S_DB,
        '-out', output_file,
        '-outfmt', '6 qseqid sseqid pident length evalue bitscore stitle',
        '-num_threads', '4',
        '-max_target_seqs', '5',
        '-perc_identity', '97'
    ]
    
    # Alt küme varsa taxid filtresi ekle
    taxid = SUBSET_TAXIDS.get(blast_subset)
    if taxid:
        cmd.extend(['-taxids', taxid])
    
    start_time = time.time()
    print(f"BLAST başlatılıyor... Alt küme: {blast_subset}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"BLAST tamamlandı! Süre: {elapsed:.2f} saniye")
            print(f"Sonuçlar: {output_file}")
            return output_file, elapsed
        else:
            print(f"BLAST hatası: {result.stderr}")
            return None, elapsed
    except subprocess.TimeoutExpired:
        print("BLAST zaman aşımı!")
        return None, 300

def parse_blast_results(result_file):
    results = []
    if not result_file or not os.path.exists(result_file):
        return results
    
    with open(result_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                results.append({
                    'query': parts[0],
                    'subject': parts[1],
                    'identity': float(parts[2]),
                    'length': int(parts[3]),
                    'evalue': float(parts[4]),
                    'bitscore': float(parts[5]),
                    'title': parts[6] if len(parts) > 6 else ''
                })
    return results

if __name__ == "__main__":
    print("BLAST entegrasyonu hazır.")
    print("Bu modül main.py tarafından çağrılacak.")