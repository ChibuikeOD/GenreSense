from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import spotipy
from genresense.config import RapidApiSettings
from genresense.schema import AUDIO_FEATURE_COLUMNS

from genresense.features import FeatureEngineeringError, FeatureSet


class RecommendationError(ValueError):
    """Raised when clustering or playlist generation cannot proceed."""


@dataclass
class ClusterDiagnostics:
    selected_clusters: int
    silhouette_score: float | None
    inertia_by_cluster: dict[int, float]
    silhouette_by_cluster: dict[int, float]


@dataclass
class ClusterSummary:
    cluster_id: int
    label: str
    track_count: int
    share: float
    representative_track: str
    avg_energy: float
    avg_tempo: float
    avg_valence: float


@dataclass
class GenreClusterResult:
    clustered_frame: pd.DataFrame
    summaries: list[ClusterSummary]
    diagnostics: ClusterDiagnostics
    scaled_feature_columns: list[str]


@dataclass
class PlaylistBlueprint:
    mode: str
    cluster_id: int
    cluster_label: str
    track_ids: list[str]
    seed_track_name: str
    description: str


class MathematicalGenreFinder:
    def __init__(self, *, min_clusters: int = 2, max_clusters: int = 8, random_state: int = 42) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

    def fit(self, feature_set: FeatureSet) -> GenreClusterResult:
        frame = feature_set.normalized_frame.copy()
        if frame.empty:
            raise FeatureEngineeringError("Audio feature values are missing for all saved tracks.")

        scaled_feature_columns = [f"scaled_{column}" for column in feature_set.feature_columns]
        matrix = frame[scaled_feature_columns].to_numpy()
        sample_count = len(frame)

        if sample_count == 1:
            frame["cluster_id"] = 0
            frame["cluster_label"] = "Genre 1"
            frame["centroid_distance"] = 0.0
            summaries = self._summaries_from_frame(frame)
            return GenreClusterResult(
                clustered_frame=frame,
                summaries=summaries,
                diagnostics=ClusterDiagnostics(
                    selected_clusters=1,
                    silhouette_score=None,
                    inertia_by_cluster={1: 0.0},
                    silhouette_by_cluster={},
                ),
                scaled_feature_columns=scaled_feature_columns,
            )

        candidate_clusters = list(range(self.min_clusters, min(self.max_clusters, sample_count - 1) + 1))
        inertia_by_cluster: dict[int, float] = {}
        silhouette_by_cluster: dict[int, float] = {}
        best_cluster_count = 1
        best_score = float("-inf")

        for cluster_count in candidate_clusters:
            model = KMeans(n_clusters=cluster_count, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(matrix)
            inertia_by_cluster[cluster_count] = float(model.inertia_)

            if len(set(labels)) < 2:
                continue

            score = float(silhouette_score(matrix, labels))
            silhouette_by_cluster[cluster_count] = score
            if score > best_score:
                best_score = score
                best_cluster_count = cluster_count

        if not silhouette_by_cluster:
            best_cluster_count = min(2, sample_count)

        final_model = KMeans(n_clusters=best_cluster_count, random_state=self.random_state, n_init=10)
        labels = final_model.fit_predict(matrix)
        distances = final_model.transform(matrix).min(axis=1)

        frame["cluster_id"] = labels.astype(int)
        frame["cluster_label"] = frame["cluster_id"].map(lambda value: f"Genre {value + 1}")
        frame["centroid_distance"] = distances
        summaries = self._summaries_from_frame(frame)

        return GenreClusterResult(
            clustered_frame=frame.sort_values(["cluster_id", "centroid_distance", "track_name"]).reset_index(drop=True),
            summaries=summaries,
            diagnostics=ClusterDiagnostics(
                selected_clusters=best_cluster_count,
                silhouette_score=silhouette_by_cluster.get(best_cluster_count),
                inertia_by_cluster=inertia_by_cluster or {best_cluster_count: float(final_model.inertia_)},
                silhouette_by_cluster=silhouette_by_cluster,
            ),
            scaled_feature_columns=scaled_feature_columns,
        )

    def _summaries_from_frame(self, frame: pd.DataFrame) -> list[ClusterSummary]:
        summaries: list[ClusterSummary] = []
        total_tracks = max(len(frame), 1)

        for cluster_id, cluster_frame in frame.groupby("cluster_id", sort=True):
            top_track = cluster_frame.sort_values("centroid_distance").iloc[0]
            summaries.append(
                ClusterSummary(
                    cluster_id=int(cluster_id),
                    label=f"Genre {int(cluster_id) + 1}",
                    track_count=int(len(cluster_frame)),
                    share=float(len(cluster_frame) / total_tracks),
                    representative_track=str(top_track.get("track_name") or "Unknown track"),
                    avg_energy=float(cluster_frame["energy"].mean()),
                    avg_tempo=float(cluster_frame["tempo"].mean()),
                    avg_valence=float(cluster_frame["valence"].mean()),
                )
            )

        return summaries


class PlaylistBuilder:
    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = np.random.default_rng(random_state)

    def build(self, cluster_result: GenreClusterResult, *, cluster_id: int, mode: str, limit: int = 20) -> PlaylistBlueprint:
        if mode not in {"pure", "hybrid"}:
            raise RecommendationError("Playlist mode must be 'pure' or 'hybrid'.")

        cluster_frame = cluster_result.clustered_frame
        if cluster_frame.empty:
            raise RecommendationError("No tracks are available for playlist generation.")

        target_summary = next((item for item in cluster_result.summaries if item.cluster_id == cluster_id), None)
        if target_summary is None:
            raise RecommendationError("The requested mathematical genre was not found.")

        playlist_size = max(1, min(limit, len(cluster_frame)))
        if mode == "pure":
            track_ids, seed_track = self._build_pure(cluster_result, cluster_id, playlist_size)
            description = (
                f"GenreSense Pure Playlist for {target_summary.label}: "
                f"tracks closest to the cluster center for a focused listening lane."
            )
        else:
            track_ids, seed_track = self._build_hybrid(cluster_result, cluster_id, playlist_size)
            description = (
                f"GenreSense Hybrid Playlist seeded from {target_summary.label}: "
                f"a weighted random walk across nearby mathematical genres."
            )

        return PlaylistBlueprint(
            mode=mode,
            cluster_id=cluster_id,
            cluster_label=target_summary.label,
            track_ids=track_ids,
            seed_track_name=seed_track,
            description=description,
        )

    def _build_pure(self, cluster_result: GenreClusterResult, cluster_id: int, playlist_size: int) -> tuple[list[str], str]:
        cluster_frame = cluster_result.clustered_frame
        matches = cluster_frame[cluster_frame["cluster_id"] == cluster_id].sort_values("centroid_distance")
        if matches.empty:
            raise RecommendationError("The selected mathematical genre is empty.")

        selection = matches.head(playlist_size)
        return selection["track_id"].tolist(), str(selection.iloc[0].get("track_name") or "Unknown track")

    def _build_hybrid(self, cluster_result: GenreClusterResult, cluster_id: int, playlist_size: int) -> tuple[list[str], str]:
        frame = cluster_result.clustered_frame.reset_index(drop=True)
        scaled_columns = cluster_result.scaled_feature_columns
        cluster_centroids = frame.groupby("cluster_id")[scaled_columns].mean()

        seed_row = frame[frame["cluster_id"] == cluster_id].sort_values("centroid_distance").iloc[0]
        chosen_indices = [int(seed_row.name)]
        used_track_ids = {str(seed_row["track_id"])}
        current_row = seed_row

        while len(chosen_indices) < playlist_size:
            available = frame[~frame["track_id"].isin(used_track_ids)]
            if available.empty:
                break

            cross_cluster = available[available["cluster_id"] != int(current_row["cluster_id"])]
            candidates = cross_cluster if not cross_cluster.empty else available
            weights = self._candidate_weights(
                current_row=current_row,
                candidates=candidates,
                cluster_centroids=cluster_centroids,
                scaled_columns=scaled_columns,
            )

            if weights.empty:
                fallback_row = candidates.iloc[0]
                next_index = int(fallback_row.name)
            else:
                next_index = int(self.random_state.choice(weights.index.to_numpy(), p=weights.to_numpy()))

            current_row = frame.loc[next_index]
            chosen_indices.append(next_index)
            used_track_ids.add(str(current_row["track_id"]))

        selection = frame.loc[chosen_indices]
        return selection["track_id"].tolist(), str(seed_row.get("track_name") or "Unknown track")

    def _candidate_weights(
        self,
        *,
        current_row: pd.Series,
        candidates: pd.DataFrame,
        cluster_centroids: pd.DataFrame,
        scaled_columns: list[str],
    ) -> pd.Series:
        if candidates.empty:
            return pd.Series(dtype="float64")

        current_vector = current_row[scaled_columns].to_numpy(dtype=float).reshape(1, -1)
        candidate_vectors = candidates[scaled_columns].to_numpy(dtype=float)
        cosine_scores = cosine_similarity(current_vector, candidate_vectors).flatten()

        tempo_range = max(float(candidates["tempo"].max() - candidates["tempo"].min()), 1.0)
        tempo_scores = 1.0 - (candidates["tempo"].sub(float(current_row["tempo"])).abs() / tempo_range).clip(upper=1.0)
        energy_scores = 1.0 - candidates["energy"].sub(float(current_row["energy"])).abs().clip(upper=1.0)

        current_centroid = cluster_centroids.loc[int(current_row["cluster_id"])].to_numpy(dtype=float).reshape(1, -1)
        candidate_centroids = candidates["cluster_id"].map(lambda item: cluster_centroids.loc[int(item)].to_numpy(dtype=float))
        centroid_matrix = np.vstack(candidate_centroids.to_list())
        bridge_scores = cosine_similarity(current_centroid, centroid_matrix).flatten()

        combined = (
            0.5 * np.clip(cosine_scores, 0.0, None)
            + 0.2 * tempo_scores.to_numpy(dtype=float)
            + 0.15 * energy_scores.to_numpy(dtype=float)
            + 0.15 * np.clip(bridge_scores, 0.0, None)
        )
        combined = np.clip(combined, 0.001, None)
        normalized = combined / combined.sum()
        return pd.Series(normalized, index=candidates.index)

class DiscoveryPlaylistBuilder:
    def __init__(self, client: spotipy.Spotify, rapidapi: RapidApiSettings, *, random_state: int = 42) -> None:
        self.client = client
        self.rapidapi = rapidapi
        self.random_state = np.random.default_rng(random_state)

    def build(
        self, 
        cluster_result: GenreClusterResult, 
        raw_preview: pd.DataFrame,
        *, 
        cluster_id: int, 
        limit: int = 20
    ) -> tuple[str, list[dict[str, str]], str]:
        cluster_frame = cluster_result.clustered_frame
        if cluster_frame.empty:
            raise RecommendationError("No tracks are available for discovery generation.")

        target_summary = next((item for item in cluster_result.summaries if item.cluster_id == cluster_id), None)
        if target_summary is None:
            raise RecommendationError("The requested mathematical genre was not found.")

        matches = cluster_frame[cluster_frame["cluster_id"] == cluster_id].sort_values("centroid_distance")
        if matches.empty:
            raise RecommendationError("The selected mathematical genre is empty.")

        seed_track_name = str(matches.iloc[0].get("track_name") or "Unknown track")
        
        # 1. Get Top Artists
        top_artists = matches["artist_name"].dropna().unique()[:3]
        if len(top_artists) == 0:
            top_artists = ["pop"] # fallback generic keyword

        # 2. Scrape Candidates
        candidate_tracks = []
        errors = []
        for artist in top_artists:
            for offset in [0, 10, 20]:
                try:
                    # Spotify max limit is now 10 for unapproved apps
                    results = self.client.search(q=f'artist:"{artist}"', type="track", limit=10, offset=offset)
                    candidate_tracks.extend(results.get("tracks", {}).get("items", []))
                except Exception as e:
                    errors.append(f"Artist search failed: {e}")
        
        # 3. Filter Known Tracks
        known_ids = set(raw_preview["track_id"].tolist())
        seen = set()
        unique_candidates = []
        
        def add_candidates(tracks: list[dict[str, Any]]) -> None:
            for t in tracks:
                tid = t.get("id")
                if tid and tid not in known_ids and tid not in seen:
                    seen.add(tid)
                    unique_candidates.append(t)
                    
        add_candidates(candidate_tracks)

        # 4. Fallback if not enough new tracks found
        if len(unique_candidates) < limit:
            for offset in [0, 10, 20, 30, 40]:
                try:
                    results = self.client.search(q="year:2024", type="track", limit=10, offset=offset)
                    add_candidates(results.get("tracks", {}).get("items", []))
                except Exception as e:
                    errors.append(f"Fallback search failed: {e}")

        if not unique_candidates:
            err_msg = "Could not find any new tracks to recommend. Please try a different mathematical genre or run ingestion again."
            if errors:
                err_msg += f" API Errors encountered: {', '.join(errors)}"
            raise RecommendationError(err_msg)

        unique_candidates = unique_candidates[:40]

        # 4. Fetch Audio Features
        candidate_ids = [t["id"] for t in unique_candidates]
        # Delayed import to avoid circular dependency
        from genresense.spotify_client import SpotifySavedTracksExtractor, build_feature_dataframe
        extractor = SpotifySavedTracksExtractor(self.client, self.rapidapi)
        features_dict, warning = extractor._fetch_audio_features(candidate_ids)
        
        if not features_dict:
            raise RecommendationError(f"Failed to fetch audio features for candidate tracks. {warning}")

        # 5. Build DataFrame
        records = []
        for t in unique_candidates:
            feature_row = features_dict.get(t["id"])
            if feature_row:
                records.append({
                    "track_id": t["id"],
                    "track_name": t.get("name"),
                    "popularity": t.get("popularity"),
                    "audio_features": feature_row
                })
        
        candidate_frame = build_feature_dataframe(records)
        if candidate_frame.empty:
            raise RecommendationError("Could not parse audio features for candidate tracks.")
            
        candidate_frame = candidate_frame.dropna(subset=AUDIO_FEATURE_COLUMNS)
        if candidate_frame.empty:
            raise RecommendationError("Candidates are missing required audio features.")

        # 6. Compute Cosine Similarity
        base_available = raw_preview.dropna(subset=AUDIO_FEATURE_COLUMNS)
        scaler = StandardScaler()
        scaler.fit(base_available[AUDIO_FEATURE_COLUMNS])
        
        candidate_vectors = scaler.transform(candidate_frame[AUDIO_FEATURE_COLUMNS])
        centroid_series = matches[cluster_result.scaled_feature_columns].mean()
        centroid_vector = centroid_series.to_numpy(dtype=float).reshape(1, -1)
        
        cosine_scores = cosine_similarity(centroid_vector, candidate_vectors).flatten()
        candidate_frame["similarity"] = cosine_scores
        
        # 7. Select Top Tracks
        best_candidates = candidate_frame.sort_values("similarity", ascending=False).head(limit)
        
        generated_tracks = []
        for _, row in best_candidates.iterrows():
            generated_tracks.append({
                "id": row["track_id"],
                "name": str(row.get("track_name", "Unknown Track")),
                "url": f"https://open.spotify.com/track/{row['track_id']}"
            })
            
        playlist_name = f"GenreSense Discovery: {target_summary.label}"
        return playlist_name, generated_tracks, seed_track_name
