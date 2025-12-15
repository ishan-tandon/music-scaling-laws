import os
import json
import numpy as np

with open('data/vocab.json', 'r') as f:
    vocab = json.load(f)
itos = {i:ch for i,ch in enumerate(vocab)}

path = 'data/train.bin'
file_size = os.path.getsize(path)

random_start = np.random.randint(0, (file_size // 2) - 200)
offset = random_start * 2 

print(f"Reading from byte offset: {offset}")

data = np.memmap(path, dtype=np.uint16, mode='r', offset=offset, shape=(200,))

print("\n--- DECODED MUSIC SNIPPET ---")
text = "".join([itos.get(int(i), '') for i in data])
print(text)
print("\n-----------------------------")

if "M:4/4" in text or "|" in text or "z" in text:
    print("VERDICT: Data is VALID music.")
else:
    print("VERDICT: Data looks strange.")