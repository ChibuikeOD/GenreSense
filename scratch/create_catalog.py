import pandas as pd
import numpy as np
from pathlib import Path

def create_mock_catalog():
    # Create a small but diverse mock dataset for the catalog
    # In a real scenario, this would be the 1.2M+ tracks CSV
    
    genres = ["Rock", "Pop", "Jazz", "Electronic", "Classical", "Hip-Hop", "R&B", "Country"]
    tracks = []
    
    for i in range(1000):
        genre = np.random.choice(genres)
        tracks.append({
            "track_id": f"catalog_{i}",
            "track_name": f"{genre} Track {i}",
            "artist_name": f"Artist {i % 100}",
            "danceability": np.random.random(),
            "energy": np.random.random(),
            "valence": np.random.random(),
            "acousticness": np.random.random(),
            "tempo": 60 + np.random.random() * 140,
            "loudness": -20 + np.random.random() * 20,
            "popularity": np.random.randint(0, 100)
        })
    
    df = pd.DataFrame(tracks)
    output_path = Path("data/catalog/music_universe.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Mock catalog created with {len(df)} tracks at {output_path}")

if __name__ == "__main__":
    create_mock_catalog()
