from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when required application secrets are missing."""


REQUIRED_SPOTIFY_SCOPES = (
    "user-library-read",
    "playlist-modify-private",
    "playlist-modify-public",
)


@dataclass(frozen=True)
class SpotifySettings:
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: str = "user-library-read playlist-modify-private playlist-modify-public"


@dataclass(frozen=True)
class PostgresSettings:
    dsn: str
    dataset_name: str = "spotify_audit"


@dataclass(frozen=True)
class RapidApiSettings:
    api_key: str
    api_host: str = "spotify-extended-audio-features-api.p.rapidapi.com"
    base_url: str = "https://spotify-extended-audio-features-api.p.rapidapi.com/v1"
    timeout_seconds: float = 20.0


@dataclass(frozen=True)
class AppSettings:
    spotify: SpotifySettings
    postgres: PostgresSettings
    rapidapi: RapidApiSettings

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AppSettings":
        spotify = mapping.get("spotify")
        postgres = mapping.get("postgres")
        rapidapi = mapping.get("rapidapi")

        if not spotify:
            raise ConfigurationError("Missing [spotify] configuration in Streamlit secrets.")
        if not postgres:
            raise ConfigurationError("Missing [postgres] configuration in Streamlit secrets.")
        if not rapidapi:
            raise ConfigurationError("Missing [rapidapi] configuration in Streamlit secrets.")

        return cls(
            spotify=SpotifySettings(
                client_id=_require(spotify, "client_id", "spotify"),
                client_secret=_require(spotify, "client_secret", "spotify"),
                redirect_uri=_require(spotify, "redirect_uri", "spotify"),
                scope=normalize_spotify_scope(str(spotify.get("scope", "user-library-read"))),
            ),
            postgres=PostgresSettings(
                dsn=_require(postgres, "dsn", "postgres"),
                dataset_name=postgres.get("dataset_name", "spotify_audit"),
            ),
            rapidapi=RapidApiSettings(
                api_key=_require(rapidapi, "api_key", "rapidapi"),
                api_host=str(rapidapi.get("api_host", "spotify-extended-audio-features-api.p.rapidapi.com")),
                base_url=str(rapidapi.get("base_url", "https://spotify-extended-audio-features-api.p.rapidapi.com/v1")),
                timeout_seconds=float(rapidapi.get("timeout_seconds", 20.0)),
            ),
        )


def _require(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not value:
        raise ConfigurationError(f"Missing '{key}' in [{section_name}] secrets configuration.")
    return str(value)


def normalize_spotify_scope(scope: str | None) -> str:
    provided = []
    if scope:
        provided.extend(part.strip() for part in str(scope).split() if part.strip())

    merged: list[str] = []
    for item in [*provided, *REQUIRED_SPOTIFY_SCOPES]:
        if item not in merged:
            merged.append(item)

    return " ".join(merged)
