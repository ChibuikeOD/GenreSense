from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
from urllib.parse import quote

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


# Do not treat these as real input artists for cross-artist exclusion.
_PLACEHOLDER_ARTIST_NORMS = frozenset({"unknown artist", "various artists"})


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
            frame["centroid_distance"] = 0.0
            summaries = self._summaries_from_frame(frame)
            label_map = {s.cluster_id: s.label for s in summaries}
            frame["cluster_label"] = frame["cluster_id"].map(label_map)
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
        frame["centroid_distance"] = distances
        summaries = self._summaries_from_frame(frame)

        # Build a cluster_id -> label lookup from the generated summaries
        label_map = {s.cluster_id: s.label for s in summaries}
        frame["cluster_label"] = frame["cluster_id"].map(label_map)

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
        used_labels: set[str] = set()

        for cluster_id, cluster_frame in frame.groupby("cluster_id", sort=True):
            top_track = cluster_frame.sort_values("centroid_distance").iloc[0]
            label = self._infer_genre_label(cluster_frame, used_labels)
            used_labels.add(label)
            summaries.append(
                ClusterSummary(
                    cluster_id=int(cluster_id),
                    label=label,
                    track_count=int(len(cluster_frame)),
                    share=float(len(cluster_frame) / total_tracks),
                    representative_track=str(top_track.get("track_name") or "Unknown track"),
                    avg_energy=float(cluster_frame["energy"].mean()),
                    avg_tempo=float(cluster_frame["tempo"].mean()),
                    avg_valence=float(cluster_frame["valence"].mean()),
                )
            )

        return summaries

    @staticmethod
    def _infer_genre_label(cluster_frame: pd.DataFrame, used_labels: set[str]) -> str:
        """Derive a human-readable genre name from average audio features."""
        energy = float(cluster_frame["energy"].mean())
        valence = float(cluster_frame["valence"].mean())
        tempo = float(cluster_frame["tempo"].mean())
        acousticness = float(cluster_frame["acousticness"].mean()) if "acousticness" in cluster_frame.columns else 0.0
        danceability = float(cluster_frame["danceability"].mean()) if "danceability" in cluster_frame.columns else 0.0
        speechiness = float(cluster_frame["speechiness"].mean()) if "speechiness" in cluster_frame.columns else 0.0
        instrumentalness = float(cluster_frame["instrumentalness"].mean()) if "instrumentalness" in cluster_frame.columns else 0.0

        # Build descriptors from audio feature thresholds
        if instrumentalness > 0.5:
            if energy < 0.4:
                base = "Ambient"
            elif tempo > 120:
                base = "Electronic"
            else:
                base = "Instrumental"
        elif speechiness > 0.33:
            if energy > 0.7:
                base = "High-Energy Hip-Hop"
            elif valence > 0.5:
                base = "Feel-Good Rap"
            else:
                base = "Lyrical Hip-Hop"
        elif acousticness > 0.6:
            if valence > 0.5:
                base = "Acoustic Pop"
            elif energy < 0.4:
                base = "Acoustic Ballad"
            else:
                base = "Acoustic Indie"
        elif danceability > 0.7 and energy > 0.7:
            if valence > 0.6:
                base = "Dance Pop"
            else:
                base = "Club Banger"
        elif energy > 0.75:
            if tempo > 140:
                base = "High-Energy Rock"
            elif valence > 0.5:
                base = "Upbeat Pop"
            else:
                base = "Intense Anthems"
        elif energy < 0.35:
            if valence < 0.3:
                base = "Melancholic Slow Burn"
            elif acousticness > 0.4:
                base = "Lo-Fi Chill"
            else:
                base = "Dreamy Downtempo"
        elif valence > 0.65:
            base = "Feel-Good Vibes"
        elif valence < 0.3:
            base = "Dark & Moody"
        elif danceability > 0.65:
            base = "Groovy R&B"
        elif tempo > 130:
            base = "Uptempo Drive"
        else:
            base = "Mid-Tempo Mix"

        # Deduplicate: if label is already taken, append a distinguishing qualifier
        if base not in used_labels:
            return base
        qualifiers = ["II", "III", "IV", "V"]
        for q in qualifiers:
            candidate = f"{base} {q}"
            if candidate not in used_labels:
                return candidate
        return f"{base} ({len(used_labels) + 1})"


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


class SonicRecommendationEngine:
    def __init__(self, client: spotipy.Spotify, rapidapi: RapidApiSettings):
        self.client = client
        self.rapidapi = rapidapi
        self.catalog_path = Path(__file__).resolve().parents[2] / "data" / "catalog" / "music_universe.csv"

    def _load_catalog(self) -> pd.DataFrame:
        if not self.catalog_path.exists():
            return pd.DataFrame()
        catalog = pd.read_csv(self.catalog_path)
        for column in ["track_name", "artist_name"]:
            if column not in catalog.columns:
                return pd.DataFrame()
        return catalog

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return " ".join(str(value or "").lower().replace("-", " ").split())

    def _parse_query(self, query: str) -> tuple[str, str | None]:
        parts = [part.strip() for part in query.split(" - ", 1)]
        if len(parts) == 2:
            return parts[0], parts[1]
        return query.strip(), None

    def seed_input_artist_norms(
        self,
        seeds: list[dict[str, Any]],
        track_queries: list[str] | None = None,
    ) -> set[str]:
        """Normalized artist names to exclude from cross-artist recommendations.

        Combines Spotify/catalog seed metadata with ``track - artist`` hints from
        form fields so exclusion still works when API objects omit ``artists``.
        """
        norms: set[str] = set()
        for track in seeds:
            for artist in track.get("artists") or []:
                name = artist.get("name")
                if not name:
                    continue
                n = self._normalize_text(name)
                if n and n not in _PLACEHOLDER_ARTIST_NORMS:
                    norms.add(n)
        if track_queries:
            for q in track_queries:
                qs = str(q).strip()
                if not qs:
                    continue
                _, artist = self._parse_query(qs)
                if artist:
                    n = self._normalize_text(artist)
                    if n and n not in _PLACEHOLDER_ARTIST_NORMS:
                        norms.add(n)
        return norms

    @staticmethod
    def _artist_norm_alias_tokens(norm: str) -> set[str]:
        n = (norm or "").strip()
        if not n:
            return set()
        aliases = {n}
        if n.startswith("the ") and len(n) > 4:
            aliases.add(n[4:].strip())
        return aliases

    def _expand_artist_exclusion_tokens(self, norms: set[str] | None) -> set[str]:
        if not norms:
            return set()
        toks: set[str] = set()
        for n in norms:
            toks |= self._artist_norm_alias_tokens(n)
        return toks

    def _catalog_artist_excluded(self, catalog_norm: str, exclude_tokens: set[str]) -> bool:
        if not exclude_tokens:
            return False
        return bool(self._artist_norm_alias_tokens(catalog_norm) & exclude_tokens)

    @staticmethod
    def default_album_cover_data_uri() -> str:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
            "<rect fill='%23111827' width='64' height='64' rx='8'/>"
            "<path fill='%231db954' d='M20 42V24l18-3v18' opacity='0.9'/>"
            "<rect x='40' y='21' width='4' height='22' fill='%2364748b' rx='1'/>"
            "</svg>"
        )
        return "data:image/svg+xml," + quote(svg, safe="")

    @staticmethod
    def spotify_track_url(track_id: str) -> str | None:
        tid = str(track_id or "").strip()
        if len(tid) == 22 and tid.isalnum():
            return f"https://open.spotify.com/track/{tid}"
        return None

    def _catalog_row_to_track(self, row: pd.Series) -> dict[str, Any]:
        tid = str(row.get("track_id"))
        url = self.spotify_track_url(tid)
        name = str(row.get("track_name"))
        artist = str(row.get("artist_name"))
        search_q = f"{name} {artist}".strip()
        return {
            "id": tid,
            "name": name,
            "artists": [{"name": artist}],
            "album": {"images": [{"url": ""}]},
            "audio_features": {
                column: float(row[column])
                for column in AUDIO_FEATURE_COLUMNS
                if column in row and pd.notna(row[column])
            },
            "popularity": int(row.get("popularity", 0) or 0),
            "open_url": url or f"https://open.spotify.com/search/{quote(search_q)}",
        }

    def seed_from_query(self, query: str) -> dict[str, Any]:
        track_name, artist_name = self._parse_query(query)
        artist_name = artist_name or "Unknown artist"
        return {
            "id": f"input_{self._normalize_text(query).replace(' ', '_')}",
            "name": track_name or query.strip() or "Input song",
            "artists": [{"name": artist_name}],
            "album": {"images": [{"url": ""}]},
            "audio_features": {
                "energy": 0.5,
                "valence": 0.5,
                "danceability": 0.5,
            },
            "is_input_only": True,
        }

    def resolve_seeds(self, track_queries: list[str]) -> list[dict[str, Any]]:
        """Resolve user-entered tracks against the standing catalog first."""
        catalog = self._load_catalog()
        seeds = []
        seen_ids: set[str] = set()

        for q in track_queries:
            if not q.strip():
                continue

            fallback_seed = self.seed_from_query(q)
            if not catalog.empty:
                track_query, artist_query = self._parse_query(q)
                track_norm = self._normalize_text(track_query)
                artist_norm = self._normalize_text(artist_query)
                candidates = catalog.copy()
                candidates["_track_norm"] = candidates["track_name"].map(self._normalize_text)
                candidates["_artist_norm"] = candidates["artist_name"].map(self._normalize_text)

                if artist_norm:
                    artist_matches = candidates[candidates["_artist_norm"].str.contains(artist_norm, regex=False, na=False)]
                    if not artist_matches.empty:
                        candidates = artist_matches
                    else:
                        candidates = candidates.iloc[0:0]

                exact = candidates[candidates["_track_norm"] == track_norm]
                contains = candidates[candidates["_track_norm"].str.contains(track_norm, regex=False, na=False)]
                match = exact if not exact.empty else contains

                if not match.empty:
                    row = match.sort_values("popularity", ascending=False).iloc[0]
                    track = self._catalog_row_to_track(row)
                    if track["id"] not in seen_ids:
                        seeds.append(track)
                        seen_ids.add(track["id"])
                    continue

            # Fallback only resolves the seed if it is not present in the standing catalog.
            try:
                results = self.client.search(q=q, type="track", limit=1)
                tracks = results.get("tracks", {}).get("items", [])
                if tracks and tracks[0]["id"] not in seen_ids:
                    seeds.append(tracks[0])
                    seen_ids.add(tracks[0]["id"])
            except Exception:
                pass

            if fallback_seed["id"] not in seen_ids:
                seeds.append(fallback_seed)
                seen_ids.add(fallback_seed["id"])

        return seeds

    def attach_seed_audio_features(self, seeds: list[dict[str, Any]]) -> None:
        """Spotify search results omit audio features; merge them in for viz and anchoring."""
        pending: list[str] = []
        by_id: dict[str, dict[str, Any]] = {}
        for track in seeds:
            tid = str(track.get("id", ""))
            if len(tid) != 22 or not tid.isalnum():
                continue
            existing = track.get("audio_features")
            if isinstance(existing, dict):
                ev, vv, dv = existing.get("energy"), existing.get("valence"), existing.get("danceability")
                if ev is not None and vv is not None and dv is not None:
                    continue
            if tid not in by_id:
                pending.append(tid)
                by_id[tid] = track

        if not pending or not self.client:
            return

        try:
            for start in range(0, len(pending), 100):
                chunk = pending[start : start + 100]
                rows = self.client.audio_features(chunk) or []
                for tid, row in zip(chunk, rows):
                    if not row:
                        continue
                    target = by_id.get(tid)
                    if not target:
                        continue

                    def _pick(key: str, default: float = 0.5) -> float:
                        v = row.get(key)
                        try:
                            return float(v) if v is not None else default
                        except (TypeError, ValueError):
                            return default

                    target["audio_features"] = {
                        "energy": _pick("energy"),
                        "valence": _pick("valence"),
                        "danceability": _pick("danceability"),
                    }
        except Exception:
            return

    def calculate_anchor(self, seeds: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate the average Energy, Valence, and Danceability of the seeds."""
        energies: list[float] = []
        valences: list[float] = []
        danceabilities: list[float] = []
        for t in seeds:
            af = t.get("audio_features")
            if not isinstance(af, dict):
                continue
            try:
                if af.get("energy") is not None:
                    energies.append(float(af["energy"]))
                if af.get("valence") is not None:
                    valences.append(float(af["valence"]))
                if af.get("danceability") is not None:
                    danceabilities.append(float(af["danceability"]))
            except (TypeError, ValueError):
                continue

        return {
            "energy": sum(energies) / len(energies) if energies else 0.5,
            "valence": sum(valences) / len(valences) if valences else 0.5,
            "danceability": sum(danceabilities) / len(danceabilities) if danceabilities else 0.5,
        }

    def expand_pool(self, seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Search for ~150 candidate tracks related to the seeds for a rich landscape."""
        artist_ids = set()
        for t in seeds:
            for artist in t.get("artists", []):
                if artist.get("id"):
                    artist_ids.add(artist["id"])
        
        pool = []
        seen_ids = {t["id"] for t in seeds}
        
        # 1. Artist Top Tracks & Related Artists
        for artist_id in list(artist_ids)[:5]:
            try:
                # Top Tracks
                results = self.client.artist_top_tracks(artist_id, country='US')
                for t in results.get("tracks", []):
                    if t["id"] not in seen_ids:
                        pool.append(t)
                        seen_ids.add(t["id"])
                
                # Related Artists
                related = self.client.artist_related_artists(artist_id)
                for ra in related.get("artists", [])[:3]:
                    ra_top = self.client.artist_top_tracks(ra["id"], country='US')
                    for t in ra_top.get("tracks", []):
                        if t["id"] not in seen_ids:
                            pool.append(t)
                            seen_ids.add(t["id"])
            except Exception:
                continue
        
        # 2. Keyword Search based on seed genres/names if pool is still small
        if len(pool) < 40:
            for s in seeds[:3]:
                try:
                    search_results = self.client.search(q=f'track:"{s["name"]}"', type="track", limit=20)
                    for t in search_results.get("tracks", {}).get("items", []):
                        if t["id"] not in seen_ids:
                            pool.append(t)
                            seen_ids.add(t["id"])
                except Exception:
                    continue

        # 3. Global Fallback if still empty (ensure visualization works)
        if len(pool) < 10:
            try:
                # Search for current popular tracks as a general baseline
                results = self.client.search(q="year:2024", type="track", limit=50)
                for t in results.get("tracks", {}).get("items", []):
                    if t["id"] not in seen_ids:
                        pool.append(t)
                        seen_ids.add(t["id"])
            except Exception:
                pass

        return pool[:150]

    def recommend(
        self,
        seeds: list[dict[str, Any]],
        limit: int = 20,
        *,
        exclude_seed_artists: bool = False,
        track_queries: list[str] | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Recommend standing-catalog tracks near the seed anchor in feature space.

        By default, prefers tracks whose catalog artist matches a seed artist (playlist discovery).
        With ``exclude_seed_artists=True``, only non-seed artists are returned (audio similarity only).
        """
        anchor = self.calculate_anchor(seeds)
        catalog = self._load_catalog()
        if catalog.empty:
            return anchor, []

        seed_ids = {str(track.get("id")) for track in seeds}
        seed_artists = {
            self._normalize_text(artist.get("name"))
            for track in seeds
            for artist in track.get("artists", [])
            if artist.get("name")
        }
        exclude_artists = self.seed_input_artist_norms(seeds, track_queries)
        exclude_toks = self._expand_artist_exclusion_tokens(exclude_artists)

        pool_df = catalog.copy()
        pool_df["_artist_norm"] = pool_df["artist_name"].map(self._normalize_text)
        pool_df = pool_df[~pool_df["track_id"].astype(str).isin(seed_ids)]

        if exclude_seed_artists and exclude_toks:
            pool_df = pool_df[
                ~pool_df["_artist_norm"].map(lambda x: self._catalog_artist_excluded(str(x), exclude_toks))
            ]

        if pool_df.empty:
            return anchor, []

        target_features = ["energy", "valence", "danceability"]
        if exclude_seed_artists:
            for feature in target_features:
                if feature not in pool_df.columns:
                    pool_df[feature] = 0.5
            target_vec = np.array([anchor.get(f, 0.5) for f in target_features])
            catalog_matrix = pool_df[target_features].fillna(0.5).to_numpy()
            pool_df = pool_df.copy()
            pool_df["distance"] = np.linalg.norm(catalog_matrix - target_vec, axis=1)
            top_matches = pool_df.sort_values(["distance", "popularity"], ascending=[True, False]).head(limit)
        else:
            artist_pool = pool_df[pool_df["_artist_norm"].isin(seed_artists)]
            if artist_pool.empty:
                artist_pool = pool_df.sort_values("popularity", ascending=False).head(max(limit * 4, 50))
            else:
                artist_pool = artist_pool.copy()

            for feature in target_features:
                if feature not in artist_pool.columns:
                    artist_pool[feature] = 0.5

            target_vec = np.array([anchor.get(f, 0.5) for f in target_features])
            catalog_matrix = artist_pool[target_features].fillna(0.5).to_numpy()
            distances = np.linalg.norm(catalog_matrix - target_vec, axis=1)
            artist_pool["distance"] = distances
            artist_pool["artist_overlap"] = artist_pool["_artist_norm"].isin(seed_artists).astype(int)

            top_matches = artist_pool.sort_values(
                ["artist_overlap", "distance", "popularity"],
                ascending=[False, True, False],
            )

            if len(top_matches) < limit:
                remaining = pool_df[~pool_df["track_id"].astype(str).isin(set(top_matches["track_id"].astype(str)))]
                remaining = remaining.copy()
                for feature in target_features:
                    if feature not in remaining.columns:
                        remaining[feature] = 0.5
                remaining_matrix = remaining[target_features].fillna(0.5).to_numpy()
                remaining["distance"] = np.linalg.norm(remaining_matrix - target_vec, axis=1)
                remaining["artist_overlap"] = 0
                top_matches = pd.concat(
                    [
                        top_matches,
                        remaining.sort_values(["distance", "popularity"], ascending=[True, False]).head(
                            limit - len(top_matches)
                        ),
                    ],
                    ignore_index=True,
                )
            else:
                top_matches = top_matches.head(limit)

        recommendations = []
        for _, row in top_matches.iterrows():
            track = self._catalog_row_to_track(row)
            track["distance"] = float(row["distance"])
            recommendations.append(track)

        return anchor, recommendations

    def get_artist_discovery_playlist(self, seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return up to 3 standing-catalog songs per liked artist."""
        catalog = self._load_catalog()
        if catalog.empty:
            return []

        seed_ids = {str(track.get("id")) for track in seeds}
        seed_artists = []
        for track in seeds:
            for artist in track.get("artists", []):
                name = artist.get("name")
                if name and name not in seed_artists:
                    seed_artists.append(name)

        discovery_tracks = []
        seen_ids = set(seed_ids)

        catalog["_artist_norm"] = catalog["artist_name"].map(self._normalize_text)
        for artist_name in seed_artists:
            artist_rows = catalog[catalog["_artist_norm"] == self._normalize_text(artist_name)]
            artist_rows = artist_rows[~artist_rows["track_id"].astype(str).isin(seen_ids)]
            for _, row in artist_rows.sort_values("popularity", ascending=False).head(3).iterrows():
                track = self._catalog_row_to_track(row)
                discovery_tracks.append(track)
                seen_ids.add(track["id"])
                
        return discovery_tracks

    def playlist_three_per_seed(self, seeds: list[dict[str, Any]], *, per_seed: int = 3, track_queries: list[str] | None = None) -> list[dict[str, Any]]:
        """Pick ``per_seed`` standing-catalog tracks per seed by nearest E/V/D vector.

        Skips any seed artist so the playlist discovers **other** artists with a similar profile.
        """
        if per_seed < 1:
            return []

        catalog = self._load_catalog()
        if catalog.empty or not seeds:
            return []

        features = ["energy", "valence", "danceability"]
        cat = catalog.copy()
        for col in features:
            if col not in cat.columns:
                cat[col] = 0.5
        cat["_artist_norm"] = cat["artist_name"].map(self._normalize_text)

        seed_ids = {str(s.get("id")) for s in seeds}
        seed_artist_norms = self.seed_input_artist_norms(seeds, track_queries)
        exclude_toks = self._expand_artist_exclusion_tokens(seed_artist_norms)
        seen: set[str] = set(seed_ids)
        picks: list[dict[str, Any]] = []

        for seed in seeds:
            af = seed.get("audio_features") if isinstance(seed.get("audio_features"), dict) else {}
            vec = np.array(
                [float(af.get(col)) if af.get(col) is not None else 0.5 for col in features],
                dtype=float,
            )

            pool = cat[~cat["track_id"].astype(str).isin(seen)]
            if exclude_toks:
                pool = pool[
                    ~pool["_artist_norm"].map(lambda x: self._catalog_artist_excluded(str(x), exclude_toks))
                ]
            if pool.empty:
                continue

            batch_ids: list[str] = []
            mat = pool[features].fillna(0.5).to_numpy(dtype=float)
            dist = np.linalg.norm(mat - vec, axis=1)
            order = np.argsort(dist)
            for idx in order:
                if len(batch_ids) >= per_seed:
                    break
                tid = str(pool.iloc[int(idx)]["track_id"])
                if tid not in batch_ids:
                    batch_ids.append(tid)

            for tid in batch_ids[:per_seed]:
                tid = str(tid)
                if tid in seen:
                    continue
                row_match = cat[cat["track_id"].astype(str) == tid]
                if row_match.empty:
                    continue
                picks.append(self._catalog_row_to_track(row_match.iloc[0]))
                seen.add(tid)

        return picks

    @staticmethod
    def _seed_feature_vector(
        seed: dict[str, Any],
        columns: list[str],
        fallback: np.ndarray,
    ) -> np.ndarray:
        """Build a numeric feature row for a seed; missing keys use ``fallback`` (e.g. catalog medians)."""
        af = seed.get("audio_features") if isinstance(seed.get("audio_features"), dict) else {}
        vals: list[float] = []
        for i, col in enumerate(columns):
            v = af.get(col)
            try:
                vals.append(float(v) if v is not None else float(fallback[i]))
            except (TypeError, ValueError):
                vals.append(float(fallback[i]))
        return np.asarray(vals, dtype=float)

    def playlist_knn_per_seed(
        self,
        seeds: list[dict[str, Any]],
        *,
        neighbors_per_seed: int = 3,
        track_queries: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """For each seed, take *k* catalog tracks closest in scaled audio feature space.

        Never returns tracks whose catalog ``artist_name`` matches any seed artist (normalized),
        so recommendations are **cross-artist** neighbors.
        """
        if neighbors_per_seed < 1:
            return []

        catalog = self._load_catalog()
        if catalog.empty or not seeds:
            return []

        columns = [c for c in AUDIO_FEATURE_COLUMNS if c in catalog.columns]
        if not columns:
            return []

        cat = catalog.copy()
        cat["_artist_norm"] = cat["artist_name"].map(self._normalize_text)
        seed_artist_norms = self.seed_input_artist_norms(seeds, track_queries)
        exclude_toks = self._expand_artist_exclusion_tokens(seed_artist_norms)

        mat = cat[columns].apply(pd.to_numeric, errors="coerce")
        col_medians = mat.median(numeric_only=True)
        mat = mat.fillna(col_medians).fillna(0.5)
        fallback = mat.median(numeric_only=True).reindex(columns).fillna(0.5).to_numpy(dtype=float)

        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat.to_numpy(dtype=float))

        seed_ids = {str(s.get("id")) for s in seeds}
        eligible_mask = ~cat["track_id"].astype(str).isin(seed_ids)
        if exclude_toks:
            excluded = cat["_artist_norm"].map(lambda x: self._catalog_artist_excluded(str(x), exclude_toks))
            eligible_mask &= ~excluded
        eligible_idx = np.flatnonzero(eligible_mask.to_numpy())

        if eligible_idx.size == 0:
            return []

        picks: list[dict[str, Any]] = []
        seen: set[str] = set(seed_ids)

        for seed in seeds:
            vec = self._seed_feature_vector(seed, columns, fallback).reshape(1, -1)
            vec_scaled = scaler.transform(vec)
            subset_scaled = mat_scaled[eligible_idx]
            dists = np.linalg.norm(subset_scaled - vec_scaled, axis=1)
            order = np.argsort(dists)
            added = 0
            for j in order:
                if added >= neighbors_per_seed:
                    break
                row_pos = int(eligible_idx[int(j)])
                tid = str(cat.iloc[row_pos]["track_id"])
                if tid in seen:
                    continue
                picks.append(self._catalog_row_to_track(cat.iloc[row_pos]))
                seen.add(tid)
                added += 1

        return picks

    def knn_landscape_neighbors(
        self,
        anchor: dict[str, float],
        *,
        exclude_ids: set[str],
        limit: int = 32,
        exclude_artist_norms: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Catalog tracks closest to the seed anchor in energy/valence/danceability (for viz only)."""
        catalog = self._load_catalog()
        if catalog.empty:
            return []

        target_features = ["energy", "valence", "danceability"]
        pool = catalog.copy()
        for feature in target_features:
            if feature not in pool.columns:
                pool[feature] = 0.5

        ex = {str(x) for x in exclude_ids}
        pool = pool[~pool["track_id"].astype(str).isin(ex)]
        if exclude_artist_norms:
            pool["_artist_norm"] = pool["artist_name"].map(self._normalize_text)
            ex_toks = self._expand_artist_exclusion_tokens(exclude_artist_norms)
            pool = pool[
                ~pool["_artist_norm"].map(lambda x: self._catalog_artist_excluded(str(x), ex_toks))
            ]
        if pool.empty:
            return []

        target_vec = np.array([float(anchor.get(f, 0.5)) for f in target_features])
        matrix = pool[target_features].fillna(0.5).to_numpy()
        pool = pool.copy()
        pool["distance"] = np.linalg.norm(matrix - target_vec, axis=1)
        top = pool.nsmallest(limit, "distance")

        out: list[dict[str, Any]] = []
        for _, row in top.iterrows():
            track = self._catalog_row_to_track(row)
            track["distance"] = float(row["distance"])
            track["viz_role"] = "landscape"
            out.append(track)
        return out
