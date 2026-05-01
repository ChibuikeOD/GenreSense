import os
from dotenv import load_dotenv
from pathlib import Path

# Fix: .env is in the parent directory of scratch/
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

print(f"SPOTIFY_CLIENT_ID: {os.environ.get('SPOTIFY_CLIENT_ID')}")
print(f"SPOTIFY_REDIRECT_URI: {os.environ.get('SPOTIFY_REDIRECT_URI')}")
