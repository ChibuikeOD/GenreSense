from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd
import requests
import spotipy
import json
import time
from pathlib import Path

from melodicmap.config import RapidApiSettings
from melodicmap.schema import AUDIO_FEATURE_COLUMNS


@dataclass
class ExtractionResult:
    records: list[dict[str, Any]]
    dataframe: pd.DataFrame
    audio_features_warning: str | None = None


@dataclass
class PlaylistCreationResult:
    playlist_id: str
    playlist_name: str
    playlist_url: str | None
    track_count: int


class SpotifySavedTracksExtractor:
    def __init__(self, client: spotipy.Spotify, rapidapi: RapidApiSettings) -> None:
        self.client = client
        self.rapidapi = rapidapi
        self.http = requests.Session()

    #region agent log
    _DEBUG_LOG_PATH = Path(__file__).resolve().parents[2] / ".logs" / "debug-005787.log"
    _DEBUG_SESSION_ID = "005787"

    @classmethod
    def _agent_log(cls, *, hypothesis_id: str, message: str, data: dict[str, Any] | None = None, run_id: str = "pre") -> None:
        payload = {
            "sessionId": cls._DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": "spotify_client.py",
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        try:
            with cls._DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
    #endregion

    def extract(self, progress_callback: Callable[[float, str], None] | None = None) -> ExtractionResult:
        self._agent_log(
            hypothesis_id="C",
            message="extract start",
            data={
                "rapidapi_host": self.rapidapi.api_host,
                "rapidapi_timeout_seconds_type": type(self.rapidapi.timeout_seconds).__name__,
                "rapidapi_timeout_seconds": str(self.rapidapi.timeout_seconds),
            },
        )
        if progress_callback:
            progress_callback(0.05, "Fetching saved tracks from Spotify...")
        saved_items = self._fetch_saved_tracks()
        self._agent_log(hypothesis_id="C", message="saved tracks fetched", data={"items": len(saved_items)})
        
        valid_items = [item for item in saved_items if item.get("track", {}).get("id")]
        track_ids = [item["track"]["id"] for item in valid_items]
        
        if progress_callback:
            progress_callback(0.20, f"Found {len(track_ids)} tracks. Fetching audio features...")
            
        audio_features, warning = self._fetch_audio_features(track_ids, progress_callback)
        self._agent_log(
            hypothesis_id="C",
            message="audio features fetched",
            data={"track_ids": len(track_ids), "features": len(audio_features), "has_warning": bool(warning)},
        )
        records = [self._to_record(item, audio_features) for item in valid_items]
        dataframe = build_feature_dataframe(records)
        
        if progress_callback:
            progress_callback(0.90, "Finalizing feature set...")
            
        return ExtractionResult(records=records, dataframe=dataframe, audio_features_warning=warning)

    def _fetch_saved_tracks(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0
        limit = 50
        max_tracks = 300

        while len(items) < max_tracks:
            self._agent_log(hypothesis_id="C", message="fetch_saved_tracks batch", data={"offset": offset, "limit": limit})
            response = self.client.current_user_saved_tracks(limit=limit, offset=offset)
            batch = response.get("items", [])
            items.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

        return items[:max_tracks]

    def _fetch_audio_features(
        self, 
        track_ids: list[str], 
        progress_callback: Callable[[float, str], None] | None = None
    ) -> tuple[dict[str, dict[str, Any]], str | None]:
        features: dict[str, dict[str, Any]] = {}
        if not track_ids:
            return features, None

        failed_requests = 0
        warning: str | None = None
        headers = {
            "x-rapidapi-key": self.rapidapi.api_key,
            "x-rapidapi-host": self.rapidapi.api_host,
        }
        base_url = self.rapidapi.base_url.rstrip("/")
        batch_size = 5  # max supported by the RapidAPI endpoint

        # Chunk track_ids into batches
        batches = [track_ids[i:i + batch_size] for i in range(0, len(track_ids), batch_size)]
        total_batches = len(batches)
        self._agent_log(
            hypothesis_id="C",
            message="fetch_audio_features start",
            data={"track_ids": len(track_ids), "batches": total_batches, "timeout_seconds": str(self.rapidapi.timeout_seconds)},
        )
        
        # Track shared state across threads
        stop_all = False
        auth_error = None
        rate_limit_hit = False

        def fetch_batch(batch_idx: int, batch_ids: list[str]):
            nonlocal stop_all, auth_error, rate_limit_hit
            if stop_all:
                return None
                
            ids_param = ",".join(batch_ids)
            url = f"{base_url}/audio-features?ids={ids_param}"
            
            try:
                # Use isolated requests.get instead of sharing self.http session to avoid connection pool deadlocks
                resp = requests.get(url, headers=headers, timeout=self.rapidapi.timeout_seconds)
                if resp.status_code == 200:
                    return resp.json().get("audio_features", [])
                
                # Fallback to Spotify native audio features if RapidAPI fails
                try:
                    spotify_features = self.client.audio_features(batch_ids)
                    if spotify_features and any(spotify_features):
                        return [f for f in spotify_features if f]
                except Exception:
                    pass
                
                # Log non-200 responses for easier debugging
                cls._agent_log(
                    hypothesis_id="R",
                    message="RapidAPI response error",
                    data={
                        "status_code": resp.status_code,
                        "batch_size": len(batch_ids),
                        "first_id": batch_ids[0] if batch_ids else None
                    },
                    run_id="pre"
                )

                if resp.status_code in (401, 403):
                    auth_error = "RapidAPI rejected audio feature requests (401/403). Check your key."
                    stop_all = True
                elif resp.status_code == 429:
                    rate_limit_hit = True
                    stop_all = True
            except requests.RequestException:
                pass
            return None

        # Execute batches synchronously in serverless to avoid thread termination issues
        completed_count = 0
        for i, batch_ids in enumerate(batches):
            if stop_all:
                break
            
            batch_data = fetch_batch(i, batch_ids)
            if batch_data:
                for item in batch_data:
                    if isinstance(item, dict) and item.get("id"):
                        features[item["id"]] = item
            else:
                if not stop_all:
                    failed_requests += len(batch_ids)

            completed_count += 1
            if progress_callback and not stop_all:
                progress = 0.20 + (0.60 * (completed_count / total_batches))
                progress_callback(progress, f"Analyzing audio features ({completed_count}/{total_batches})...")

        if auth_error:
            return features, auth_error
        if rate_limit_hit:
            return features, "RapidAPI rate limit reached; results are partial."
        if failed_requests and not features:
            return {}, "All RapidAPI requests failed. Check your network."
        if failed_requests:
            return features, f"Some requests failed ({failed_requests} tracks). Results are partial."

        return features, None

    def _to_record(self, item: dict[str, Any], audio_features: dict[str, dict[str, Any]]) -> dict[str, Any]:
        track = item["track"]
        feature_row = audio_features.get(track["id"], {})

        return {
            "track_id": track["id"],
            "saved_at": item.get("added_at"),
            "track_name": track.get("name"),
            "popularity": track.get("popularity"),
            "duration_ms": track.get("duration_ms"),
            "explicit": track.get("explicit"),
            "preview_url": track.get("preview_url"),
            "external_urls": track.get("external_urls", {}),
            "album": {
                "album_id": track.get("album", {}).get("id"),
                "album_name": track.get("album", {}).get("name"),
                "release_date": track.get("album", {}).get("release_date"),
                "total_tracks": track.get("album", {}).get("total_tracks"),
                "album_type": track.get("album", {}).get("album_type"),
            },
            "artists": [
                {
                    "artist_id": artist.get("id"),
                    "artist_name": artist.get("name"),
                    "artist_uri": artist.get("uri"),
                }
                for artist in track.get("artists", [])
            ],
            "audio_features": {
                "danceability": feature_row.get("danceability"),
                "energy": feature_row.get("energy"),
                "valence": feature_row.get("valence"),
                "acousticness": feature_row.get("acousticness"),
                "tempo": feature_row.get("tempo"),
                "loudness": feature_row.get("loudness"),
                "speechiness": feature_row.get("speechiness"),
                "instrumentalness": feature_row.get("instrumentalness"),
                "liveness": feature_row.get("liveness"),
                "key": feature_row.get("key"),
                "mode": feature_row.get("mode"),
                "time_signature": feature_row.get("time_signature"),
            },
        }


def build_feature_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in records:
        features = record.get("audio_features", {})
        artists = record.get("artists", [])
        artist_name = artists[0].get("artist_name") if artists else "Unknown"
        
        rows.append(
            {
                "track_id": record.get("track_id"),
                "track_name": record.get("track_name"),
                "artist_name": artist_name,
                "saved_at": record.get("saved_at"),
                "popularity": record.get("popularity"),
                **{column: features.get(column) for column in AUDIO_FEATURE_COLUMNS},
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe

    dataframe["saved_at"] = pd.to_datetime(dataframe["saved_at"], utc=True, errors="coerce")
    return dataframe


class SpotifyPlaylistPublisher:
    def __init__(self, client: spotipy.Spotify) -> None:
        self.client = client

    def create_playlist(
        self,
        *,
        user_id: str,
        name: str,
        description: str,
        track_ids: list[str],
        public: bool = False,
    ) -> PlaylistCreationResult:
        if not track_ids:
            raise ValueError("At least one track is required to create a Spotify playlist.")

        playlist = self.client.user_playlist_create(user=user_id, name=name, public=public, description=description)
        playlist_id = str(playlist["id"])
        track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]

        for index in range(0, len(track_uris), 100):
            self.client.playlist_add_items(playlist_id, track_uris[index : index + 100])

        external_urls = playlist.get("external_urls", {})
        return PlaylistCreationResult(
            playlist_id=playlist_id,
            playlist_name=str(playlist.get("name") or name),
            playlist_url=external_urls.get("spotify"),
            track_count=len(track_uris),
        )
