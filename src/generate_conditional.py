import torch
import json
import os
from torch.nn import functional as F
from model import GPT, GPTConfig

# --- CONFIG ---
NUM_SAMPLES = 100
MAX_TOKENS = 1024
TEMPERATURE = 0.95 
TOP_K = 200
OUTPUT_DIR = "generated_samples"

# --- EXPANDED PROMPTS ---
PROMPTS = [
    # -- ROCK / POP --
    "X:1\nT:Classic Rock Riff\nM:4/4\nL:1/8\nK:Am\n|: A2 e2 c2 A2",   
    "X:1\nT:Punk Power Chords\nM:4/4\nL:1/8\nK:D\n|: D2 D2 G2 G2",
    "X:1\nT:Pop Ballad Piano\nM:4/4\nL:1/8\nK:C\n|: C2 E2 G2 c2",
    "X:1\nT:Heavy Metal Gallop\nM:4/4\nL:1/16\nK:Em\n|: E2E2 E2E2",
    "X:1\nT:Indie Rock Strum\nM:4/4\nL:1/8\nK:G\n|: G2 B2 d2 B2",

    # -- FOLK / TRADITIONAL --
    "X:1\nT:Irish Jig\nM:6/8\nL:1/8\nK:G\n|: G2 G B2 d",
    "X:1\nT:Scottish Reel\nM:4/4\nL:1/8\nK:D\n|: D2 F2 A2 d2",
    "X:1\nT:Sea Shanty\nM:2/4\nL:1/8\nK:C\n|: C2 E2 G2",
    "X:1\nT:Appalachian Fiddle\nM:4/4\nL:1/8\nK:A\n|: A2 c2 e2 a2",
    "X:1\nT:Eastern Folk\nM:2/4\nL:1/8\nK:Dm\n|: D2 F2 A2",

    # -- JAZZ / BLUES --
    "X:1\nT:Jazz Walking Bass\nM:4/4\nL:1/8\nK:F\n|: F,2 A,2 C2 D2",
    "X:1\nT:Blues Shuffle\nM:12/8\nL:1/8\nK:E\n|: E2 G E2 G",
    "X:1\nT:Bossa Nova Rhythm\nM:4/4\nL:1/8\nK:Dm\n|: D2 z D z2 D2",
    "X:1\nT:Swing Era Sax\nM:4/4\nL:1/8\nK:Bb\n|: B2 d2 f2",
    "X:1\nT:Smooth Jazz Keys\nM:4/4\nL:1/8\nK:Cmaj7\n|: C2 E2 G2 B2",

    # -- CLASSICAL --
    "X:1\nT:Baroque Prelude\nM:4/4\nL:1/16\nK:C\n|: c2e2g2c'2",
    "X:1\nT:Romantic Waltz\nM:3/4\nL:1/4\nK:A\n|: E3 A3",
    "X:1\nT:Classical Adagio\nM:4/4\nL:1/4\nK:F\n|: F4 A4",
    "X:1\nT:Marching Band\nM:2/4\nL:1/8\nK:Eb\n|: E2 G2 B2",
    "X:1\nT:Church Organ\nM:4/4\nL:1/2\nK:G\n|: G4 B4",

    # -- ELECTRONIC / EXPERIMENTAL --
    "X:1\nT:Techno Bass\nM:4/4\nL:1/16\nK:Cm\n|: C2z2 C2z2",
    "X:1\nT:Trance Arpeggio\nM:4/4\nL:1/16\nK:Am\n|: A2c2e2a2 c2e2",
    "X:1\nT:Odd Time Signature\nM:5/4\nL:1/8\nK:D\n|: D2 F2 A2 d2 c2",
    "X:1\nT:Ambient Drone\nM:4/4\nL:1/1\nK:C\n|: C8",
    "X:1\nT:Video Game Chiptune\nM:4/4\nL:1/8\nK:C\n|: c2 G2 E2 C2"
]

# --- SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Model on {device}...")

with open('data/vocab.json', 'r') as f: vocab = json.load(f)
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

checkpoint = torch.load('checkpoints/xxl_final/ckpt.pt', map_location=device)
config = GPTConfig(block_size=1024, vocab_size=len(vocab), n_layer=24, n_head=25, n_embd=1600)
model = GPT(config)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- GENERATION FUNCTION ---
def generate_conditional(index, prompt):
    start_ids = [stoi[c] for c in prompt if c in stoi]
    x = torch.tensor([start_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(MAX_TOKENS):
            x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / TEMPERATURE
            
            # Clamp k to be at most the vocabulary size
            k = min(TOP_K, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)

    output_text = "".join([itos.get(int(i), '') for i in x[0].tolist()])
    filename = f"{OUTPUT_DIR}/sample_cond_{index+1:03d}.abc"
    with open(filename, 'w') as f:
        f.write(output_text)

# --- RUN LOOP ---
print(f"\n--- GENERATING {NUM_SAMPLES} CONDITIONAL SAMPLES ---\n")
for i in range(NUM_SAMPLES):
    prompt = PROMPTS[i % len(PROMPTS)]
    title = prompt.splitlines()[1].replace("T:", "")
    print(f"[{i+1}/{NUM_SAMPLES}] Style: {title}")
    generate_conditional(i, prompt)

print("\nDone! Conditional samples added to 'generated_samples/' folder.")