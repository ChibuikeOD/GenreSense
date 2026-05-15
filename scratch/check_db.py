import os
from dotenv import load_dotenv
load_dotenv('.env')
import psycopg2

dsn = os.environ.get('POSTGRES_DSN', '')
if not dsn:
    print('NO POSTGRES_DSN found in .env')
    exit(1)

# Try port 6543 (Supabase session pooler) if 5432 fails
conn = None
for dsn_try in [dsn, dsn.replace(':5432/', ':6543/')]:
    try:
        conn = psycopg2.connect(dsn_try, connect_timeout=10)
        print(f'Connected with: {dsn_try[:50]}...')
        break
    except Exception as e:
        print(f'Failed: {dsn_try[:50]}... => {e}')

if conn is None:
    print('Could not connect with any DSN variant.')
    exit(1)
cur = conn.cursor()

# List all tables
cur.execute("""
    SELECT table_schema, table_name 
    FROM information_schema.tables 
    WHERE table_schema NOT IN ('pg_catalog', 'information_schema') 
    ORDER BY table_schema, table_name
""")
tables = cur.fetchall()
print("=== TABLES ===")
for t in tables:
    print(f"  {t[0]}.{t[1]}")

# For each table, show row count and sample
for schema, table in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"')
        count = cur.fetchone()[0]
        print(f"\n=== {schema}.{table} ({count} rows) ===")
        cur.execute(f'SELECT * FROM "{schema}"."{table}" LIMIT 3')
        cols = [d[0] for d in cur.description]
        print("  Columns:", cols)
        for row in cur.fetchall():
            print("  ", dict(zip(cols, row)))
    except Exception as e:
        print(f"  Error reading {schema}.{table}: {e}")

conn.close()
