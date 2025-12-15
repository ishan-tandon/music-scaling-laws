import os
import glob
import math
import music21
import time

# --- CONFIG ---
INPUT_DIR = "generated_samples"
OUTPUT_DIR = "midi_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COUNTERS ---
stats = {
    "total":   {"count": 0, "syntax": 0, "midi": 0},
    "uncond":  {"count": 0, "syntax": 0, "midi": 0},
    "cond":    {"count": 0, "syntax": 0, "midi": 0}
}

print(f"--- EVALUATING SAMPLES IN '{INPUT_DIR}' ---\n")

abc_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.abc")))

if not abc_files:
    print("No .abc files found! Make sure you ran the generation scripts first.")
    exit()

start_time = time.time()

for abc_file in abc_files:
    filename = os.path.basename(abc_file)
    
    # Determine Type based on filename
    if "uncond" in filename:
        group = "uncond"
        print(f"[Uncond] {filename}...", end=" ")
    elif "cond" in filename:
        group = "cond"
        print(f"[ Cond ] {filename}...", end=" ")
    else:
        group = "total" # Fallback
        print(f"[Misc  ] {filename}...", end=" ")

    # Update counts
    stats["total"]["count"] += 1
    stats[group]["count"] += 1
    
    try:
        # 1. Attempt to parse ABC (Syntax Check)
        # forceSource=True ensures music21 treats it strictly as ABC text
        score = music21.converter.parse(abc_file, forceSource=True)
        stats["total"]["syntax"] += 1
        stats[group]["syntax"] += 1
        
        # 2. Attempt to write MIDI (Conversion Check)
        midi_filename = filename.replace(".abc", ".mid")
        midi_path = os.path.join(OUTPUT_DIR, midi_filename)
        score.write('midi', fp=midi_path)
        
        stats["total"]["midi"] += 1
        stats[group]["midi"] += 1
        print("SUCCESS")
        
    except Exception as e:
        print(f"FAILED ({str(e)[:20]}...)")

# --- FINAL REPORT GENERATION ---
def print_stat_row(name, data):
    count = data["count"]
    if count == 0:
        print(f"{name:15} | N/A (0 files)")
        return
        
    syn_pct = (data["syntax"] / count) * 100
    mid_pct = (data["midi"] / count) * 100
    print(f"{name:15} | {count:5d} | {data['syntax']:5d} ({syn_pct:5.1f}%) | {data['midi']:5d} ({mid_pct:5.1f}%)")

print("\n" + "="*60)
print(f"FINAL EVALUATION REPORT ({time.time()-start_time:.1f}s)")
print("="*60)
print(f"{'CATEGORY':15} | {'COUNT':5} | {'SYNTAX VALID':13} | {'MIDI SUCCESS':13}")
print("-" * 60)

print_stat_row("Unconditional", stats["uncond"])
print_stat_row("Conditional", stats["cond"])
print("-" * 60)
print_stat_row("TOTAL", stats["total"])
print("="*60)

"""
============================================================
FINAL EVALUATION REPORT (15.6s)
============================================================
CATEGORY        | COUNT | SYNTAX VALID  | MIDI SUCCESS 
------------------------------------------------------------
Unconditional   |   100 |    93 ( 93.0%) |    74 ( 74.0%)
Conditional     |   100 |    96 ( 96.0%) |    95 ( 95.0%)
------------------------------------------------------------
TOTAL           |   200 |   189 ( 94.5%) |   169 ( 84.5%)
============================================================
"""