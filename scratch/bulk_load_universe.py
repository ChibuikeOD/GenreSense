import os
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2
from psycopg2.extras import execute_values

# 1. Load CSV data locally
csv_path = "data/catalog/spotify_songs_raw.csv"
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}. Run download script first.")
    sys.exit(1)

print("Loading and cleaning dataset from local CSV...")
df = pd.read_csv(csv_path)

# Deduplicate by track_id, keeping the one with highest popularity if duplicates exist
df = df.sort_values('track_popularity', ascending=False).drop_duplicates('track_id')

# Clean track names/artists (fill NAs with defaults)
df['track_name'] = df['track_name'].fillna('Unknown Track').astype(str)
df['track_artist'] = df['track_artist'].fillna('Unknown Artist').astype(str)

# Map column names to match our DB model
mapped_data = []
for _, row in df.iterrows():
    mapped_data.append((
        str(row['track_id']),
        str(row['track_name']),
        str(row['track_artist']),
        float(row.get('danceability', 0.5)),
        float(row.get('energy', 0.5)),
        float(row.get('valence', 0.5)),
        float(row.get('acousticness', 0.5)),
        float(row.get('tempo', 120.0)),
        float(row.get('loudness', -8.0)),
        float(row.get('speechiness', 0.0)),
        float(row.get('instrumentalness', 0.0)),
        int(row.get('track_popularity', 0))
    ))

print(f"Ready to insert {len(mapped_data)} unique, real Spotify tracks into the database.")

# 2. Initialize DB
dsn = os.environ.get("POSTGRES_DSN")
if not dsn:
    print("Error: POSTGRES_DSN not set in .env")
    sys.exit(1)

# Try 5432 then 6543
conn = None
for d in [dsn, dsn.replace(":5432/", ":6543/")]:
    try:
        conn = psycopg2.connect(d, connect_timeout=15)
        print("Connected to database successfully!")
        break
    except Exception as e:
        print(f"Failed connection to {d[:40]}... => {e}")

if not conn:
    print("Fatal: Could not connect to database.")
    sys.exit(1)

# 3. Find active schema
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
print(f"Target schema is: {schema}")

# 4. Ensure Table structure matches
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

# 5. Bulk UPSERT using execute_values (insanely fast)
print(f"Beginning bulk UPSERT of {len(mapped_data)} rows. This will take just a few seconds...")
try:
    insert_query = f"""
        INSERT INTO "{schema}".music_universe (
            track_id, track_name, artist_name, 
            danceability, energy, valence, acousticness, 
            tempo, loudness, speechiness, instrumentalness, 
            popularity
        ) VALUES %s
        ON CONFLICT (track_id) DO UPDATE SET
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
    """
    
    # Process in chunks of 5000 for extra stability
    chunk_size = 5000
    total_inserted = 0
    
    for i in range(0, len(mapped_data), chunk_size):
        chunk = mapped_data[i : i + chunk_size]
        execute_values(cur, insert_query, chunk)
        conn.commit()
        total_inserted += len(chunk)
        print(f"  Pushed batch: {total_inserted} / {len(mapped_data)} rows...")

    print(f"FINISHED! Successfully bulk-seeded {total_inserted} premium tracks into the Standing Music Database!")
except Exception as e:
    print(f"Fatal error during bulk insertion: {e}")
    conn.rollback()
finally:
    cur.close()
    conn.close()
