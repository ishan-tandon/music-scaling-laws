import os
import requests
import tarfile
from tqdm import tqdm

DATASET_URL = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"

DEST_DIR = "data/raw_midi"
TAR_FILE = os.path.join(DEST_DIR, "lmd_full.tar.gz")
EXTRACT_PATH = os.path.join(DEST_DIR, "lmd_full")

def download_file(url, filename):
    print(f"Starting download from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Download failed: {e}")
        # Clean up partial file
        if os.path.exists(filename):
            os.remove(filename)
        raise

def extract_file(filename, dest_dir):
    print(f"Extracting {filename}")
    
    # Check if file is valid tar
    if not tarfile.is_tarfile(filename):
        raise ValueError("File is not a valid tar archive")

    with tarfile.open(filename, "r:gz") as tar:
        # We use a generator to avoid loading all member names into RAM
        members = tar
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=dest_dir)

if __name__ == "__main__":
    # Ensure directory exists
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # 1. Download
    if not os.path.exists(TAR_FILE):
        download_file(DATASET_URL, TAR_FILE)
    else:
        print(f"Found existing archive at {TAR_FILE}, skipping download.")

    # 2. Extract
    if not os.path.exists(EXTRACT_PATH):
        try:
            extract_file(TAR_FILE, DEST_DIR)
            print("Extraction complete.")
        except Exception as e:
            print(f"Extraction failed: {e}")
    else:
        print(f"Dataset appears to be extracted at {EXTRACT_PATH}")

"""
rohan@jsba:~/music-scaling-laws$ docker compose exec dev python src/01_download_data.py
WARN[0000] /home/rohan/music-scaling-laws/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
Starting download from http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz...
Downloading: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.65G/1.65G [03:40<00:00, 8.02MiB/s]
Extracting data/raw_midi/lmd_full.tar.gz
Extracting: 178578it [00:54, 3266.71it/s]
Extraction complete.
"""