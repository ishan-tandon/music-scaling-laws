import os
import json
import numpy as np

# Configuration
data_dir = 'data'
vocab_path = os.path.join(data_dir, 'vocab.json')

# Load Vocab
with open(vocab_path, 'r') as f:
    vocab = json.load(f)
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode_and_save(split):
    input_path = os.path.join(data_dir, f'{split}.txt')
    output_path = os.path.join(data_dir, f'{split}.bin')
    
    print(f"Processing {split}.txt...")
    
    # Read text
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    n = len(data)
    print(f"  - Length: {n} characters")

    ids = np.zeros(n, dtype=np.uint16)
    
    unknown_count = 0
    for i, ch in enumerate(data):
        ids[i] = stoi.get(ch, 0)
        
    print(f"  - Encoded. Saving to {output_path}...")
    ids.tofile(output_path)
    print(f"  - Saved {(os.path.getsize(output_path)/1024/1024):.2f} MB")

if __name__ == '__main__':
    encode_and_save('train')
    encode_and_save('val')
    print("Done. Data is now binary optimized.")