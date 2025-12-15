import os
import json
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
input_path = os.path.join('data', 'test.txt')
output_path = os.path.join('data', 'test.bin')
vocab_path = os.path.join('data', 'vocab.json')
CHUNK_SIZE = 1024 * 1024  # Process 1MB of text at a time

# 1. Load Vocab
if not os.path.exists(vocab_path):
    print("Error: vocab.json not found!")
    exit()

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

stoi = {ch: i for i, ch in enumerate(vocab)}

# 2. Get File Size for Progress Bar
if not os.path.exists(input_path):
    print(f"Error: {input_path} not found!")
    exit()

file_size = os.path.getsize(input_path)
print(f"Processing {input_path} ({file_size / (1024*1024):.2f} MB)...")

# 3. Stream Processing
# We open the output file in write-binary mode
with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'wb') as f_out:
    
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar:
        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break
            
            # Encode chunk (ignoring unknown chars)
            ids = [stoi[c] for c in chunk if c in stoi]
            
            # Convert to numpy uint16 (standard for vocab < 65k)
            ids_np = np.array(ids, dtype=np.uint16)
            
            # Write directly to disk
            f_out.write(ids_np.tobytes())
            
            pbar.update(len(chunk.encode('utf-8')))

print(f"\nDone! Saved to {output_path}")

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/prepare_test.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
Processing data/test.txt (37.02 MB)...
Tokenizing: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 38.8M/38.8M [00:02<00:00, 14.7MB/s]
"""