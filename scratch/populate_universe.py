import os
import sys
import time
import requests
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# 1. Initialize Spotify client
client_id = os.environ.get("SPOTIFY_CLIENT_ID")
client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
if not client_id or not client_secret:
    print("Error: SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set in .env")
    sys.exit(1)

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# 2. Initialize Database Connection
dsn = os.environ.get("POSTGRES_DSN")
if not dsn:
    print("Error: POSTGRES_DSN not set in .env")
    sys.exit(1)

# Try 5432 then 6543
conn = None
for d in [dsn, dsn.replace(":5432/", ":6543/")]:
    try:
        conn = psycopg2.connect(d, connect_timeout=10)
        print("Connected to database successfully!")
        break
    except Exception as e:
        print(f"Failed connection to {d[:40]}... => {e}")

if not conn:
    print("Fatal: Could not connect to database.")
    sys.exit(1)

# 3. Discover the schema to use
cur = conn.cursor()
cur.execute("""
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
    ORDER BY 
      (schema_name = 'genre_sense_staging') DESC,
      (schema_name ILIKE '%genre_sense%') DESC,
      schema_name ASC
    LIMIT 1
""")
row = cur.fetchone()
schema = row[0] if row else "public"
print(f"Using target database schema: {schema}")

# 4. Create standing "music_universe" table
print("Creating clean 'music_universe' standing table...")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS "{schema}".music_universe (
        track_id TEXT PRIMARY KEY,
        track_name TEXT NOT NULL,
        artist_name TEXT NOT NULL,
        danceability FLOAT,
        energy FLOAT,
        valence FLOAT,
        acousticness FLOAT,
        tempo FLOAT,
        loudness FLOAT,
        speechiness FLOAT,
        instrumentalness FLOAT,
        popularity INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
""")
conn.commit()

# 5. Fetch real tracks from Spotify across various genres
genres = [
    "pop", "rock", "hip-hop", "r-n-b", "jazz", 
    "classical", "country", "electronic", "indie", "metal", 
    "blues", "soul", "reggae", "folk", "latin", "afrobeat"
]

tracks_pool = {} # deduplicated by ID

print(f"Gathering popular tracks across {len(genres)} genres from Spotify...")
for genre in genres:
    try:
        print(f"  Fetching genre: {genre}...")
        # Query 2 times with limit 10 to get 20 top tracks per genre (to save RapidAPI quota)
        for offset in range(0, 20, 10):
            results = sp.search(q=f"genre:{genre}", type="track", limit=10, offset=offset)
            items = results.get("tracks", {}).get("items", [])
            for item in items:
                tid = item.get("id")
                if tid and tid not in tracks_pool:
                    artists = item.get("artists", [])
                    artist_name = artists[0].get("name") if artists else "Unknown Artist"
                    tracks_pool[tid] = {
                        "id": tid,
                        "name": item.get("name"),
                        "artist": artist_name,
                        "popularity": item.get("popularity", 0)
                    }
            time.sleep(0.05)
    except Exception as e:
        print(f"  Warning: Error fetching genre {genre}: {e}")

print(f"Collected {len(tracks_pool)} unique track skeletons.")

# 6. Fetch audio features via RapidAPI in batches of 5
rapidapi_key = os.environ.get("RAPIDAPI_KEY")
rapidapi_host = os.environ.get("RAPIDAPI_HOST")
rapidapi_base = os.environ.get("RAPIDAPI_BASE_URL", "https://spotify-extended-audio-features-api.p.rapidapi.com/v1")

if not rapidapi_key or not rapidapi_host:
    print("Error: RAPIDAPI_KEY or RAPIDAPI_HOST not set in .env")
    sys.exit(1)

headers = {
    "x-rapidapi-key": rapidapi_key,
    "x-rapidapi-host": rapidapi_host,
}

all_ids = list(tracks_pool.keys())
final_tracks = []

print(f"Fetching audio features via RapidAPI in batches of 5 (Total batches: {len(all_ids) // 5 + 1})...")

for i in range(0, len(all_ids), 5):
    batch_ids = all_ids[i : i + 5]
    ids_str = ",".join(batch_ids)
    url = f"{rapidapi_base.rstrip('/')}/audio-features?ids={ids_str}"
    
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            features = resp.json().get("audio_features", [])
            for f in features:
                if not f or not isinstance(f, dict):
                    continue
                tid = f.get("id")
                if not tid or tid not in tracks_pool:
                    continue
                
                track_skeleton = tracks_pool[tid]
                final_tracks.append({
                    "id": tid,
                    "name": track_skeleton["name"],
                    "artist": track_skeleton["artist"],
                    "popularity": track_skeleton["popularity"],
                    "danceability": f.get("danceability"),
                    "energy": f.get("energy"),
                    "valence": f.get("valence"),
                    "acousticness": f.get("acousticness"),
                    "tempo": f.get("tempo"),
                    "loudness": f.get("loudness"),
                    "speechiness": f.get("speechiness"),
                    "instrumentalness": f.get("instrumentalness")
                })
        else:
            print(f"  Warning: RapidAPI returned {resp.status_code} for batch starting with {batch_ids[0]}")
        
        if (i // 5) % 10 == 0:
            print(f"  Processed {i + len(batch_ids)} / {len(all_ids)} tracks...")
            
        time.sleep(0.2) # small sleep to avoid burst rate limits
    except Exception as e:
        print(f"  Warning: Request exception in feature batch: {e}")

print(f"Successfully built a dataset of {len(final_tracks)} fully-analyzed tracks!")

# 7. Upsert tracks into database
print(f"Inserting tracks into {schema}.music_universe database table...")
inserted = 0
for t in final_tracks:
    try:
        cur.execute(f"""
            INSERT INTO "{schema}".music_universe (
                track_id, track_name, artist_name, 
                danceability, energy, valence, acousticness, 
                tempo, loudness, speechiness, instrumentalness, 
                popularity
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (track_id) DO UPDATE 
            SET 
                track_name = EXCLUDED.track_name,
                artist_name = EXCLUDED.artist_name,
                danceability = EXCLUDED.danceability,
                energy = EXCLUDED.energy,
                valence = EXCLUDED.valence,
                acousticness = EXCLUDED.acousticness,
                tempo = EXCLUDED.tempo,
                loudness = EXCLUDED.loudness,
                speechiness = EXCLUDED.speechiness,
                instrumentalness = EXCLUDED.instrumentalness,
                popularity = EXCLUDED.popularity
        """, (
            t["id"], t["name"], t["artist"],
            t["danceability"], t["energy"], t["valence"], t["acousticness"],
            t["tempo"], t["loudness"], t["speechiness"], t["instrumentalness"],
            t["popularity"]
        ))
        inserted += 1
    except Exception as e:
        print(f"  Error inserting {t['name']}: {e}")
        conn.rollback()

conn.commit()
print(f"Done! Populated {inserted} tracks into the permanent music universe table.")

cur.close()
conn.close()
