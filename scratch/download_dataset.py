import os
import requests
from pathlib import Path

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv"
out_dir = Path("data/catalog")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "spotify_songs_raw.csv"

print(f"Downloading dataset from {url}...")
try:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = 0
    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)
                
    print(f"Successfully downloaded {total_size / 1024 / 1024:.2f} MB to {out_path}")
except Exception as e:
    print(f"Download failed: {e}")
