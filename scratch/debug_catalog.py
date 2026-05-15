import sys, os
from dotenv import load_dotenv
load_dotenv()
sys.path.append('src')
import psycopg2
import pandas as pd
import traceback

dsn = os.environ.get('POSTGRES_DSN')
print(f'DSN defined: {bool(dsn)}')

# Try ports like the engine does
dsn_variants = [dsn]
if dsn and ":5432/" in dsn:
    dsn_variants.append(dsn.replace(":5432/", ":6543/"))

conn = None
for d in dsn_variants:
    try:
        print(f"Connecting to {d[:40]}...")
        conn = psycopg2.connect(d, connect_timeout=8)
        print("Connected!")
        break
    except Exception as e:
        print(f"Connection failed for {d[:40]}... => {e}")

if not conn:
    print("All connections failed.")
    sys.exit(1)

try:
    cur = conn.cursor()
    cur.execute("""
        SELECT table_schema 
        FROM information_schema.tables 
        WHERE table_name = 'saved_tracks' 
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY 
          (table_schema = 'genre_sense_staging') DESC,
          (table_schema ILIKE '%genre_sense%') DESC,
          table_schema ASC
        LIMIT 1
    """)
    row = cur.fetchone()
    print('Detected Schema:', row)
    if not row:
        print("No 'saved_tracks' table found in any schema!")
        sys.exit(1)
        
    schema = row[0]
    feature_cols_sql = ", ".join(
        f'st."audio_features__{col}" AS "{col}"'
        for col in [
            "danceability", "energy", "valence",
            "acousticness", "tempo", "loudness",
            "speechiness", "instrumentalness",
        ]
    )
    
    sql = f"""
        SELECT
            st.track_id,
            st.track_name,
            COALESCE(a.artist_name, 'Unknown Artist') AS artist_name,
            {feature_cols_sql},
            COALESCE(st.popularity, 0) AS popularity
        FROM "{schema}".saved_tracks st
        LEFT JOIN LATERAL (
            SELECT artist_name
            FROM "{schema}"."saved_tracks__artists"
            WHERE _dlt_root_id = st._dlt_id
            ORDER BY _dlt_list_idx
            LIMIT 1
        ) a ON true
        WHERE st.track_id IS NOT NULL
          AND st.track_name IS NOT NULL
    """
    print('Running full query...')
    try:
        df = pd.read_sql_query(sql, conn)
        print('Full query returned', len(df), 'rows')
        if not df.empty:
            print(df.head(3))
    except Exception as e:
        print("Full query failed:", e)
        
        # Try fallback query
        sql_fallback = f"""
            SELECT
                st.track_id,
                st.track_name,
                'Unknown Artist' AS artist_name,
                {feature_cols_sql},
                COALESCE(st.popularity, 0) AS popularity
            FROM "{schema}".saved_tracks st
            WHERE st.track_id IS NOT NULL
              AND st.track_name IS NOT NULL
        """
        print("Running fallback query...")
        df2 = pd.read_sql_query(sql_fallback, conn)
        print("Fallback query returned", len(df2), 'rows')

except Exception:
    traceback.print_exc()
finally:
    conn.close()
