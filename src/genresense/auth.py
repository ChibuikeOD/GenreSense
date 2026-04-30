from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spotipy
import streamlit as st
from spotipy.cache_handler import CacheHandler
from spotipy.oauth2 import SpotifyOAuth

from genresense.config import SpotifySettings


TOKEN_INFO_KEY = "spotify_token_info"
STATE_KEY = "spotify_oauth_state"


class StreamlitSessionCacheHandler(CacheHandler):
    """Stores the Spotify token in Streamlit session state."""

    def get_cached_token(self) -> Any:
        return st.session_state.get(TOKEN_INFO_KEY)

    def save_token_to_cache(self, token_info: Any) -> None:
        st.session_state[TOKEN_INFO_KEY] = token_info

    def delete_cached_token(self) -> None:
        st.session_state.pop(TOKEN_INFO_KEY, None)


@dataclass
class AuthStatus:
    authenticated: bool
    authorize_url: str | None = None
    profile: dict[str, Any] | None = None
    client: spotipy.Spotify | None = None
    message: str | None = None


class SpotifyAuthService:
    def __init__(self, settings: SpotifySettings) -> None:
        self.settings = settings
        self.cache_handler = StreamlitSessionCacheHandler()

    def build_oauth(self) -> SpotifyOAuth:
        state = st.session_state.get(STATE_KEY, "genresense-state")
        st.session_state[STATE_KEY] = state
        return SpotifyOAuth(
            client_id=self.settings.client_id,
            client_secret=self.settings.client_secret,
            redirect_uri=self.settings.redirect_uri,
            scope=self.settings.scope,
            open_browser=False,
            show_dialog=False,
            cache_handler=self.cache_handler,
            state=state,
        )

    def authenticate(self) -> AuthStatus:
        oauth = self.build_oauth()
        code = st.query_params.get("code")
        error = st.query_params.get("error")

        if error:
            return AuthStatus(authenticated=False, message=f"Spotify authorization failed: {error}")

        token_info = oauth.validate_token(self.cache_handler.get_cached_token())
        if token_info:
            client = spotipy.Spotify(auth_manager=oauth)
            profile = client.current_user()
            return AuthStatus(authenticated=True, client=client, profile=profile)

        if code:
            oauth.get_access_token(code=code, check_cache=False)
            st.query_params.clear()
            st.rerun()

        authorize_url = oauth.get_authorize_url()
        return AuthStatus(
            authenticated=False,
            authorize_url=authorize_url,
            message="Connect your Spotify account to load saved tracks and audio features.",
        )

    def logout(self) -> None:
        self.cache_handler.delete_cached_token()
        st.query_params.clear()

