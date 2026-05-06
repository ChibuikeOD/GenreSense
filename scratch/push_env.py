import subprocess
import os

env_vars = {
    "FLASK_SECRET_KEY": "dYiOqO0_iJ6Fx6mo5r9hFCa7iq-bhTAkAEjW0dVIovkU2EVoXrQdsL2wVb8vhze4",
    "SPOTIFY_CLIENT_ID": "26b27563491844389dd5d1aaaca342f7",
    "SPOTIFY_CLIENT_SECRET": "e3f0efa32f5f43979179ec0aa7ba3619",
    "SPOTIFY_REDIRECT_URI": "https://genre-pay.vercel.app/callback",
    "SPOTIFY_SCOPE": "user-library-read playlist-modify-public playlist-modify-private",
    "POSTGRES_DSN": "postgresql://postgres.rrqqicohxdjkvbgrngww:Tyler-Joseph21@aws-1-us-east-1.pooler.supabase.com:5432/postgres",
    "POSTGRES_DATASET_NAME": "GenreSense",
    "RAPIDAPI_KEY": "5f9e519fc6msh67128688b054b11p18c55bjsn8cc35912a3d7",
    "RAPIDAPI_HOST": "spotify-extended-audio-features-api.p.rapidapi.com",
    "RAPIDAPI_BASE_URL": "https://spotify-extended-audio-features-api.p.rapidapi.com/v1",
    "RAPIDAPI_TIMEOUT_SECONDS": "20",
    "SESSION_COOKIE_SECURE": "true"
}

for key, value in env_vars.items():
    print(f"Adding {key}...")
    # Use 'echo' to pipe the value to avoid shell escaping issues with special characters
    subprocess.run(f'echo {value} | vercel env add {key} production', shell=True)
