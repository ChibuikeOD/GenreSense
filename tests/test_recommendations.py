from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genresense.config import normalize_spotify_scope
from genresense.features import FeatureNormalizer
from genresense.recommendations import MathematicalGenreFinder, PlaylistBuilder


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"track_id": "a1", "track_name": "Glowline", "danceability": 0.81, "energy": 0.88, "valence": 0.79, "acousticness": 0.08, "tempo": 128.0, "loudness": -4.5},
            {"track_id": "a2", "track_name": "Neon Lift", "danceability": 0.77, "energy": 0.84, "valence": 0.74, "acousticness": 0.10, "tempo": 126.0, "loudness": -5.0},
            {"track_id": "a3", "track_name": "Pulse Theory", "danceability": 0.79, "energy": 0.82, "valence": 0.72, "acousticness": 0.07, "tempo": 124.0, "loudness": -4.8},
            {"track_id": "b1", "track_name": "Quiet Harbor", "danceability": 0.28, "energy": 0.24, "valence": 0.31, "acousticness": 0.86, "tempo": 84.0, "loudness": -12.4},
            {"track_id": "b2", "track_name": "Soft Static", "danceability": 0.32, "energy": 0.29, "valence": 0.37, "acousticness": 0.81, "tempo": 88.0, "loudness": -11.8},
            {"track_id": "b3", "track_name": "Muted Sky", "danceability": 0.26, "energy": 0.27, "valence": 0.29, "acousticness": 0.90, "tempo": 80.0, "loudness": -13.1},
        ]
    )


class RecommendationTests(unittest.TestCase):
    def setUp(self) -> None:
        feature_set = FeatureNormalizer().fit_transform(_sample_dataframe())
        self.cluster_result = MathematicalGenreFinder(random_state=7).fit(feature_set)

    def test_scope_normalization_adds_playlist_write_permissions(self) -> None:
        scope = normalize_spotify_scope("user-read-email")

        self.assertIn("user-library-read", scope)
        self.assertIn("playlist-modify-private", scope)
        self.assertIn("playlist-modify-public", scope)
        self.assertIn("user-read-email", scope)

    def test_genre_finder_discovers_multiple_clusters(self) -> None:
        self.assertEqual(self.cluster_result.diagnostics.selected_clusters, 2)
        self.assertEqual(len(self.cluster_result.summaries), 2)
        self.assertIn("cluster_label", self.cluster_result.clustered_frame.columns)

    def test_pure_playlist_stays_inside_selected_cluster(self) -> None:
        builder = PlaylistBuilder(random_state=5)
        target_cluster = self.cluster_result.summaries[0].cluster_id

        blueprint = builder.build(self.cluster_result, cluster_id=target_cluster, mode="pure", limit=3)
        selected_clusters = set(
            self.cluster_result.clustered_frame.set_index("track_id").loc[blueprint.track_ids, "cluster_id"].tolist()
        )

        self.assertEqual(len(blueprint.track_ids), 3)
        self.assertEqual(selected_clusters, {target_cluster})

    def test_hybrid_playlist_bridges_into_other_clusters(self) -> None:
        builder = PlaylistBuilder(random_state=5)
        target_cluster = self.cluster_result.summaries[0].cluster_id

        blueprint = builder.build(self.cluster_result, cluster_id=target_cluster, mode="hybrid", limit=4)
        selected_clusters = self.cluster_result.clustered_frame.set_index("track_id").loc[blueprint.track_ids, "cluster_id"].tolist()

        self.assertEqual(len(blueprint.track_ids), 4)
        self.assertGreater(len(set(selected_clusters)), 1)


if __name__ == "__main__":
    unittest.main()
