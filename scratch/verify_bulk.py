import sys, os
from dotenv import load_dotenv
load_dotenv('.env')
sys.path.append('src')
from melodicmap.recommendations import SonicRecommendationEngine
from melodicmap.config import RapidApiSettings

engine = SonicRecommendationEngine(
    None, 
    RapidApiSettings(None, None, None), 
    db_dsn=os.environ.get('POSTGRES_DSN'),
    db_dataset=os.environ.get('POSTGRES_DATASET_NAME', 'GenreSense')
)

catalog = engine._load_catalog()
print('=========================================')
print('   STANDING DATABASE STATISTICS')
print('=========================================')
print(f'Total Tracks:   {len(catalog):,}')
print(f'Unique Artists: {catalog["artist_name"].nunique():,}')
print('\n--- RANDOM SAMPLE OF 5 SONGS ---')
print(catalog[['track_name', 'artist_name', 'popularity']].sample(5).to_string(index=False))
print('=========================================')
