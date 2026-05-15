# MelodicMap

MelodicMap is a Spotify library auditing and recommendation pipeline that now runs as a Flask web app and is deployable on Vercel.

## What it does

- Spotify OAuth login
- Saved track extraction via Spotipy
- Audio feature enrichment via RapidAPI
- PostgreSQL ingestion via `dlt` merge/upsert
- StandardScaler feature normalization preview

## Project layout

```text
.
|-- app.py
|-- UI/landing.html
|-- requirements.txt
|-- pyproject.toml
|-- src/melodicmap
|   |-- config.py
|   |-- features.py
|   |-- pipeline.py
|   |-- schema.py
|   `-- spotify_client.py
`-- tests
    `-- test_features.py
```

## Environment variables

Set these values locally and in Vercel Project Settings:

- `FLASK_SECRET_KEY`
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REDIRECT_URI`
- `SPOTIFY_SCOPE` (optional, default `user-library-read`)
- `POSTGRES_DSN`
- `POSTGRES_DATASET_NAME` (optional, default `spotify_audit`)
- `RAPIDAPI_KEY`
- `RAPIDAPI_HOST` (optional, default `spotify-extended-audio-features-api.p.rapidapi.com`)
- `RAPIDAPI_BASE_URL` (optional, default `https://spotify-extended-audio-features-api.p.rapidapi.com/v1`)
- `RAPIDAPI_TIMEOUT_SECONDS` (optional, default `20`)
- `SESSION_COOKIE_SECURE` (optional, default `true`)

Important: `SPOTIFY_REDIRECT_URI` must exactly match your deployed domain callback.  
Example production value: `https://your-app.vercel.app/callback`

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set env vars in your shell, then run:

```powershell
python app.py
```

Open:

- `http://127.0.0.1:8501/`

## Deploy to Vercel

1. Push this repo to GitHub/GitLab/Bitbucket.
2. Import it as a Vercel project.
3. Add all environment variables listed above.
4. Update Spotify app Redirect URI to your Vercel callback URL:
   - `https://<your-project>.vercel.app/callback`
5. Deploy.

Vercel automatically detects and runs the Flask app from `app.py`.

## Security note

Do not commit real secrets. Rotate any Spotify or database credentials that have already been exposed.
