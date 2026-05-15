from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from melodicmap.features import FeatureNormalizer


class FeatureNormalizerTests(unittest.TestCase):
    def test_fit_transform_adds_scaled_columns(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "track_id": "1",
                    "track_name": "Track One",
                    "danceability": 0.3,
                    "energy": 0.5,
                    "valence": 0.2,
                    "acousticness": 0.1,
                    "tempo": 100.0,
                    "loudness": -8.0,
                },
                {
                    "track_id": "2",
                    "track_name": "Track Two",
                    "danceability": 0.9,
                    "energy": 0.8,
                    "valence": 0.7,
                    "acousticness": 0.4,
                    "tempo": 150.0,
                    "loudness": -4.0,
                },
            ]
        )

        result = FeatureNormalizer().fit_transform(dataframe)

        self.assertIn("scaled_danceability", result.normalized_frame.columns)
        self.assertIn("scaled_tempo", result.normalized_frame.columns)
        self.assertEqual(len(result.normalized_frame), 2)


if __name__ == "__main__":
    unittest.main()
