import os
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Configuration
INPUT_DIR = "data/raw_midi"
OUTPUT_DIR = "data/processed_abc"
NUM_WORKERS = os.cpu_count()  # Uses all available cores

def convert_single_file(file_path):
    """
    Converts a single MIDI file to ABC notation using the 'midi2abc' CLI tool.
    Returns: (status, message/path)
    """
    try:
        # Create a mirrored output path
        # e.g. data/raw_midi/folder/song.mid -> data/processed_abc/folder/song.abc
        rel_path = os.path.relpath(file_path, INPUT_DIR)
        out_path = os.path.join(OUTPUT_DIR, os.path.splitext(rel_path)[0] + ".abc")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Run midi2abc
        # -b : no bar checking (prevents some errors on loose MIDI files)
        # -o : output file
        cmd = ["midi2abc", file_path, "-b", "-o", out_path]
        
        result = subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            timeout=10 # Skip files that hang for >10 seconds
        )
        
        if result.returncode == 0:
            # Check if file was actually created and has content
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return ("success", out_path)
            else:
                return ("empty", file_path)
        else:
            return ("error", file_path)

    except subprocess.TimeoutExpired:
        return ("timeout", file_path)
    except Exception as e:
        return ("exception", str(e))

def main():
    print(f"Scanning for MIDI files in {INPUT_DIR}...")
    # LMD-Full has nested directories, so we use recursive glob
    midi_files = [
        y for x in os.walk(INPUT_DIR) 
        for y in glob.glob(os.path.join(x[0], '*.mid'))
    ]
    
    print(f"Found {len(midi_files)} MIDI files.")
    print(f"Starting conversion with {NUM_WORKERS} workers...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    stats = {
        "success": 0,
        "error": 0,
        "empty": 0,
        "timeout": 0,
        "exception": 0
    }

    # Parallel Processing
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(
            executor.map(convert_single_file, midi_files), 
            total=len(midi_files),
            desc="Converting"
        ))

    # Tally results
    for status, _ in results:
        if status in stats:
            stats[status] += 1
    
    print("\n--- Conversion Report ---")
    print(f"Total Files: {len(midi_files)}")
    print(f"Successfully Converted: {stats['success']}")
    print(f"Errors/Corrupted: {stats['error']}")
    print(f"Empty Outputs: {stats['empty']}")
    print(f"Timeouts: {stats['timeout']}")
    print(f"Success Rate: {stats['success'] / len(midi_files) * 100:.1f}%")

if __name__ == "__main__":
    main()

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/02_process_midi.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
Scanning for MIDI files in data/raw_midi...
Found 178561 MIDI files.
Starting conversion with 24 workers...
Converting: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 178561/178561 [01:02<00:00, 2846.07it/s]

--- Conversion Report ---
Total Files: 178561
Successfully Converted: 174531
Errors/Corrupted: 4024
Empty Outputs: 0
Timeouts: 6
Success Rate: 97.7%
"""