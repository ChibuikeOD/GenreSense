import psycopg2
import os

dsn = "postgresql://postgres.rrqqicohxdjkvbgrngww:Tyler-Joseph21@aws-1-us-east-1.pooler.supabase.com:5432/postgres"
conn = psycopg2.connect(dsn)
cur = conn.cursor()
cur.execute("SELECT length(image_data) FROM shared_graphics ORDER BY created_at DESC LIMIT 1;")
row = cur.fetchone()
print(f"LATEST GRAPHIC BYTE LENGTH: {row}")
conn.close()
