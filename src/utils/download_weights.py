import requests
import os
from pathlib import Path
from tqdm import tqdm

URL = "https://huggingface.co/timm/deit_small_patch16_224.fb_in1k/resolve/main/pytorch_model.bin"
OUTPUT_DIR = Path("weights")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "deit_small_patch16_224.bin"

def download_file(url, destination):
    print(f"Downloading fro {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False

if __name__ == "__main__":
    download_file(URL, OUTPUT_FILE)
