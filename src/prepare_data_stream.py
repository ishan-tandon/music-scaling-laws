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

def encode_and_save_streaming(split):
    input_path = os.path.join(data_dir, f'{split}.txt')
    output_path = os.path.join(data_dir, f'{split}.bin')
    
    if not os.path.exists(input_path):
        print(f"Skipping {split}.txt (not found)")
        return

    print(f"Processing {split}.txt (Streaming)...")
    
    token_count = 0
    
    # Open both files simultaneously
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'wb') as f_out:
        
        # Read in 1MB chunks to keep RAM usage low
        chunk_size = 1024 * 1024 
        
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
                
            # Encode this chunk only
            ids = [stoi.get(c, 0) for c in chunk]
            
            # Convert to numpy uint16 (2 bytes per token)
            np_ids = np.array(ids, dtype=np.uint16)
            
            # Write binary bytes directly to disk
            f_out.write(np_ids.tobytes())
            
            token_count += len(ids)
            print(f"\r  -> Encoded {token_count/1e6:.2f}M tokens...", end="")

    print(f"\n  -> Done! Saved to {output_path}")

if __name__ == '__main__':
    encode_and_save_streaming('train')
    encode_and_save_streaming('val')

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/prepare_data_stream.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion
Processing train.txt (Streaming)...
  -> Encoded 14115.37M tokens...
  -> Done! Saved to data/train.bin
Processing val.txt (Streaming)...
  -> Encoded 40.20M tokens...
  -> Done! Saved to data/val.bin
"""