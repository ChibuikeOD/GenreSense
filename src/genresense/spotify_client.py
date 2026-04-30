from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
import spotipy

from genresense.config import RapidApiSettings
from genresense.schema import AUDIO_FEATURE_COLUMNS


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

    def extract(self) -> ExtractionResult:
        saved_items = self._fetch_saved_tracks()
        valid_items = [item for item in saved_items if item.get("track", {}).get("id")]
        track_ids = [item["track"]["id"] for item in valid_items]
        audio_features, warning = self._fetch_audio_features(track_ids)
        records = [self._to_record(item, audio_features) for item in valid_items]
        dataframe = build_feature_dataframe(records)
        return ExtractionResult(records=records, dataframe=dataframe, audio_features_warning=warning)

    def _fetch_saved_tracks(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        offset = 0
        limit = 50
        max_tracks = 300

        while len(items) < max_tracks:
            response = self.client.current_user_saved_tracks(limit=limit, offset=offset)
            batch = response.get("items", [])
            items.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

        return items[:max_tracks]

    def _fetch_audio_features(self, track_ids: list[str]) -> tuple[dict[str, dict[str, Any]], str | None]:
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

        for track_id in track_ids:
            url = f"{base_url}/audio-features/{track_id}"
            
            try:
                response = self.http.get(url, headers=headers, timeout=self.rapidapi.timeout_seconds)
            except requests.RequestException:
                failed_requests += 1
                continue

            if response.status_code == 200:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("id"):
                    features[payload["id"]] = payload
                else:
                    failed_requests += 1
                continue

            if response.status_code in (401, 403):
                return (
                    features,
                    "RapidAPI rejected audio feature requests (401/403). Check your RapidAPI key and subscription status.",
                )
            if response.status_code == 429:
                warning = "RapidAPI rate limit was reached while fetching audio features; results may be partial."
                break

            failed_requests += 1

        if warning:
            return features, warning
        if failed_requests and not features:
            return (
                {},
                "RapidAPI audio feature requests failed. Check your network and RapidAPI endpoint configuration.",
            )
        if failed_requests:
            return (
                features,
                f"Some RapidAPI audio feature requests failed ({failed_requests} tracks). Results are partially complete.",
            )

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
