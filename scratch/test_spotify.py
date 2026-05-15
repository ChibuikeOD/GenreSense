import os, spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
load_dotenv('.env')
client_id = os.environ.get("SPOTIFY_CLIENT_ID")
client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))
try:
    r = sp.search(q='genre:pop', type='track', limit=10, offset=50)
    print('Limit 10 offset 50 worked:', len(r.get('tracks', {}).get('items', [])))
except Exception as e:
    print('Limit 10 offset 50 failed:', e)
