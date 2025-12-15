import os
import glob
import random
import json
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/processed_abc"
OUTPUT_DIR = "data"
MIN_LENGTH = 50 
SPLIT_RATIOS = (0.98, 0.01, 0.01)

def main():
    print(f"Scanning for ABC files in {INPUT_DIR}...")
    files = [y for x in os.walk(INPUT_DIR) for y in glob.glob(os.path.join(x[0], '*.abc'))]
    random.shuffle(files)
    
    print(f"Found {len(files)} files. Processing stream...")

    # Open file handles for writing
    f_train = open(os.path.join(OUTPUT_DIR, "train.txt"), "w", encoding="utf-8")
    f_val = open(os.path.join(OUTPUT_DIR, "val.txt"), "w", encoding="utf-8")
    f_test = open(os.path.join(OUTPUT_DIR, "test.txt"), "w", encoding="utf-8")

    # Metrics
    stats = {
        "total_tokens": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "sequence_lengths": [],
        "vocab": set()
    }

    # Calculate split indices
    n_total = len(files)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])
    # remaining goes to test

    for idx, f_path in enumerate(tqdm(files)):
        try:
            with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            
            length = len(content)
            
            if length < MIN_LENGTH:
                stats["files_skipped"] += 1
                continue

            # Update stats
            stats["total_tokens"] += length
            stats["files_processed"] += 1
            stats["vocab"].update(content)
            stats["sequence_lengths"].append(length)

            # Determine where to write
            entry = content + "\n\n"
            
            if idx < n_train:
                f_train.write(entry)
            elif idx < n_train + n_val:
                f_val.write(entry)
            else:
                f_test.write(entry)

        except Exception:
            stats["files_skipped"] += 1

    # Close files
    f_train.close()
    f_val.close()
    f_test.close()

    # Calculate Distribution Stats
    lengths = np.array(stats["sequence_lengths"])
    
    print(f"\n--- Final Deliverable Statistics ---")
    print(f"Vocabulary Size: {len(stats['vocab'])}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Conversion Success Rate: {stats['files_processed'] / len(files) * 100:.1f}%")
    print("\n--- Sequence Length Distribution ---")
    print(f"Min Length: {np.min(lengths)}")
    print(f"Max Length: {np.max(lengths)}")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print(f"Median Length: {np.median(lengths)}")
    print(f"P99 Length: {np.percentile(lengths, 99):.2f}")
    
    # Save Vocab
    vocab_list = sorted(list(stats['vocab']))
    with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
        json.dump(vocab_list, f)

if __name__ == "__main__":
    main()

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/03_prepare_dataset.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
Scanning for ABC files in data/processed_abc...
Found 178561 files. Processing stream...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 178561/178561 [03:08<00:00, 946.03it/s]

--- Final Deliverable Statistics ---
Vocabulary Size: 99
Total Tokens: 14,194,037,371
Conversion Success Rate: 98.5%

--- Sequence Length Distribution ---
Min Length: 125
Max Length: 549109892
Mean Length: 80665.81
Median Length: 19397.0
P99 Length: 87285.60
"""