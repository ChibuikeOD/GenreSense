from __future__ import annotations

from dataclasses import dataclass
import html
import json
import os
import re
import secrets
import sys
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import spotipy
from dotenv import load_dotenv
from flask import Flask, redirect, request, session, url_for, Response, stream_with_context
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genresense.config import AppSettings, ConfigurationError
from genresense.features import FeatureEngineeringError, FeatureNormalizer
from genresense.pipeline import SpotifyLibraryPipeline
from genresense.recommendations import GenreClusterResult, MathematicalGenreFinder, PlaylistBuilder, RecommendationError, SonicRecommendationEngine
from genresense.spotify_client import SpotifyPlaylistPublisher, SpotifySavedTracksExtractor
from genresense.persistence import AnalysisPersistence

if not os.environ.get("VERCEL"):
    load_dotenv(ROOT / ".env", override=False)


LANDING_TEMPLATE_PATH = ROOT / "UI" / "landing.html"
TOKEN_INFO_KEY = "spotify_token_info"
STATE_KEY = "spotify_oauth_state"
PROFILE_KEY = "spotify_profile"
# Use /tmp for the cache path on Vercel as the root filesystem is read-only.
def _get_cache_path() -> str:
    # Use a session-specific cache file to allow multiple users on the same instance.
    session_id = session.get("session_id")
    if not session_id:
        session_id = secrets.token_hex(16)
        session["session_id"] = session_id
    return str(Path("/tmp") / f".spotify_cache_{session_id}")

REQUIRED_ENV_VARS = [
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "SPOTIFY_REDIRECT_URI",
    "POSTGRES_DSN",
    "RAPIDAPI_KEY",
    "FLASK_SECRET_KEY",
]

#region agent log
_DEBUG_LOG_PATH = ROOT / ".logs" / "debug-005787.log"
_DEBUG_SESSION_ID = "005787"


def _agent_log(*, hypothesis_id: str, message: str, data: dict[str, Any] | None = None, run_id: str = "pre") -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "app.py",
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
#endregion

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "true").lower() == "true"
_agent_log(
    hypothesis_id="A",
    message="Flask app configured",
    data={
        "session_cookie_secure": bool(app.config.get("SESSION_COOKIE_SECURE")),
        "session_cookie_samesite": str(app.config.get("SESSION_COOKIE_SAMESITE")),
        "session_cookie_secure_env": os.environ.get("SESSION_COOKIE_SECURE"),
    },
)


# Spotipy's token payload can exceed typical cookie size limits, but Flask's session
# uses a signed cookie by default. For local dev, keep the token in the session
# (works if you're using a server-side session extension; otherwise you may hit limits).


def _oauth_cache_handler() -> CacheFileHandler:
    # Store OAuth tokens server-side to avoid cookie size limits.
    return CacheFileHandler(cache_path=_get_cache_path())

# Persistence layer for UserAnalysis instances.
def _get_persistence() -> AnalysisPersistence | None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        return None
    return AnalysisPersistence(dsn)

@dataclass
class LibraryAnalysis:
    records_loaded: int
    dataset_name: str | None
    raw_preview: pd.DataFrame
    scaled_preview: pd.DataFrame | None
    cluster_result: GenreClusterResult | None
    audio_features_warning: str | None


def _settings_from_env() -> AppSettings:
    mapping = {
        "spotify": {
            "client_id": os.environ.get("SPOTIFY_CLIENT_ID", "").strip() or None,
            "client_secret": os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip() or None,
            "redirect_uri": os.environ.get("SPOTIFY_REDIRECT_URI", "").strip() or None,
            "scope": os.environ.get("SPOTIFY_SCOPE", "user-library-read").strip(),
        },
        "postgres": {
            "dsn": os.environ.get("POSTGRES_DSN", "").strip() or None,
            "dataset_name": os.environ.get("POSTGRES_DATASET_NAME", "spotify_audit").strip(),
        },
        "rapidapi": {
            "api_key": os.environ.get("RAPIDAPI_KEY", "").strip() or None,
            "api_host": os.environ.get("RAPIDAPI_HOST", "spotify-extended-audio-features-api.p.rapidapi.com").strip(),
            "base_url": os.environ.get("RAPIDAPI_BASE_URL", "https://spotify-extended-audio-features-api.p.rapidapi.com/v1").strip(),
            "timeout_seconds": os.environ.get("RAPIDAPI_TIMEOUT_SECONDS", "20").strip(),
        },
    }
    return AppSettings.from_mapping(mapping)


def _build_oauth(settings: AppSettings, state: str | None = None) -> SpotifyOAuth:
    if not state:
        try:
            state = session.get(STATE_KEY)
        except RuntimeError:
            # Outside of request context (e.g. background thread)
            state = None
            
    if not state:
        try:
            state = secrets.token_urlsafe(16)
            session[STATE_KEY] = state
        except RuntimeError:
            pass

    return SpotifyOAuth(
        client_id=settings.spotify.client_id,
        client_secret=settings.spotify.client_secret,
        redirect_uri=settings.spotify.redirect_uri,
        scope=settings.spotify.scope,
        open_browser=False,
        show_dialog=False,
        state=state,
        cache_handler=_oauth_cache_handler(),
    )


def _get_client_credentials_client(settings: AppSettings) -> spotipy.Spotify:
    cache_path = str(Path("/tmp") / ".spotify_client_credentials_cache")
    auth_manager = SpotifyClientCredentials(
        client_id=settings.spotify.client_id,
        client_secret=settings.spotify.client_secret,
        cache_handler=CacheFileHandler(cache_path=cache_path)
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _get_authenticated_client(settings: AppSettings) -> tuple[spotipy.Spotify | None, dict[str, Any] | None]:
    _t0 = time.time()
    _agent_log(hypothesis_id="E", message="_get_authenticated_client start", data={}, run_id="pre")
    oauth = _build_oauth(settings)
    _t1 = time.time()
    try:
        token_info = oauth.validate_token(oauth.cache_handler.get_cached_token())
        _agent_log(
            hypothesis_id="E",
            message="oauth.validate_token returned",
            data={
                "elapsed_ms": int((time.time() - _t1) * 1000),
                "has_token_info": bool(token_info),
            },
            run_id="pre",
        )
    except Exception as exc:  # noqa: BLE001
        _agent_log(
            hypothesis_id="E",
            message="oauth.validate_token raised",
            data={"elapsed_ms": int((time.time() - _t1) * 1000), "exc": str(exc)},
            run_id="pre",
        )
        raise
    if not token_info:
        # Attempt proactive refresh if token exists but is invalid/expired
        cached_token = oauth.cache_handler.get_cached_token()
        if cached_token:
            _agent_log(hypothesis_id="E", message="attempting proactive token refresh", run_id="pre")
            token_info = oauth.refresh_access_token(cached_token["refresh_token"])
            
    if not token_info:
        _agent_log(
            hypothesis_id="E",
            message="_get_authenticated_client no token_info",
            data={"elapsed_ms_total": int((time.time() - _t0) * 1000)},
            run_id="pre",
        )
        return None, None
    # Spotipy uses `requests` under the hood; enforce a finite timeout so
    # a stuck network call can't hang the Flask request forever.
    client = spotipy.Spotify(auth_manager=oauth, requests_timeout=20, retries=0)

    # Fetch profile if not in session to avoid 429s while still having a real ID.
    profile = session.get(PROFILE_KEY)
    if not profile:
        try:
            profile = client.current_user()
            session[PROFILE_KEY] = profile
            _agent_log(hypothesis_id="E", message="fetched profile from spotify", data={"id": profile.get("id")}, run_id="pre")
        except Exception as exc:
            _agent_log(hypothesis_id="E", message="failed to fetch profile", data={"exc": str(exc)}, run_id="pre")
            profile = {"id": "unknown", "display_name": "Spotify User"}

    _agent_log(
        hypothesis_id="E",
        message="_get_authenticated_client done",
        data={"elapsed_ms_total": int((time.time() - _t0) * 1000), "user_id": profile.get("id")},
        run_id="pre",
    )
    return client, profile


def _render_landing(status_message: str) -> str:
    if not LANDING_TEMPLATE_PATH.exists():
        return "<h1>Landing template missing</h1><p>Create UI/landing.html.</p>"
    template = LANDING_TEMPLATE_PATH.read_text(encoding="utf-8")
    
    # Strip ad blocks if ads are disabled
    enable_ads = os.environ.get("ENABLE_ADS", "false").lower() == "true"
    if not enable_ads:
        template = re.sub(r'<!-- AD_BLOCK_START -->.*?<!-- AD_BLOCK_END -->', '', template, flags=re.DOTALL)

    if status_message:
        status_node = (
            '<p id="auth-status" style="text-align:center;margin-top:14px;color:#64748b;font-size:13px;line-height:1.45;">'
            f"{html.escape(status_message)}</p>"
        )
        if "</main>" in template:
            template = template.replace("</main>", status_node + "\n</main>", 1)
        else:
            template += status_node
            
    return template


def _config_error_page(error_message: str) -> str:
    required = "".join(f"<li><code>{name}</code></li>" for name in REQUIRED_ENV_VARS)
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>GenreSense Configuration Error</title>
      <style>
        body {{ font-family: Arial, sans-serif; max-width: 760px; margin: 48px auto; padding: 0 16px; color: #1f2937; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; background: #fff; }}
        h1 {{ margin-top: 0; }}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Configuration Error</h1>
        <p>{html.escape(error_message)}</p>
        <p>Set these environment variables before starting the app:</p>
        <ul>{required}</ul>
      </div>
    </body>
    </html>
    """


def _genre_summary_html(
    cluster_result: GenreClusterResult,
    *,
    selected_cluster_id: int,
    selected_playlist_mode: str,
    selected_playlist_size: int,
    selected_visibility: str,
) -> str:
    score = cluster_result.diagnostics.silhouette_score
    score_label = f"{score:.3f}" if score is not None else "n/a"

    cards = []
    options = []
    for summary in cluster_result.summaries:
        is_selected = summary.cluster_id == selected_cluster_id
        border = "#1db954" if is_selected else "rgba(255,255,255,0.1)"
        background = "rgba(29, 185, 84, 0.1)" if is_selected else "rgba(0,0,0,0.3)"
        width = max(summary.share * 100.0, 6.0)
        cards.append(
            f"""
            <div style="border:1px solid {border};border-radius:14px;padding:16px;background:{background};color:#fff;">
              <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                <div>
                  <div style="font-size:22px;font-weight:800;">{html.escape(summary.label)}</div>
                  <div style="font-size:14px;color:#a1a1aa;margin-top:2px;">{summary.track_count} tracks</div>
                </div>
                <div style="font-size:12px;color:#a1a1aa;">{summary.share * 100:.1f}%</div>
              </div>
              <div style="margin-top:12px;background:rgba(255,255,255,0.1);border-radius:999px;height:12px;overflow:hidden;">
                <div style="width:{width:.2f}%;height:100%;background:linear-gradient(90deg,#16a34a,#22c55e);"></div>
              </div>
              <div style="margin-top:12px;color:#e2e8f0;font-size:13px;line-height:1.6;">
                <strong>Anchor track:</strong> {html.escape(summary.representative_track)}<br>
                <strong>Energy:</strong> {summary.avg_energy:.2f} &nbsp; <strong>Tempo:</strong> {summary.avg_tempo:.1f} BPM
              </div>
            </div>
            """
        )
        options.append(
            f'<option value="{summary.cluster_id}"{" selected" if is_selected else ""}>{html.escape(summary.label)} ({summary.track_count} tracks)</option>'
        )

    return f"""
    <section style="margin-top:24px;display:grid;gap:16px;">
      <div>
        <h2 style="margin:0 0 6px;color:#fff;">Mathematical Genres</h2>
        <p style="margin:0;color:#a1a1aa;line-height:1.6;">K-Means grouped your songs by how they sound, not by Spotify metadata labels. Each cluster represents a distinct sonic profile.</p>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;">
        {''.join(cards)}
      </div>
      <div style="border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:18px;background:rgba(0,0,0,0.3);color:#fff;">
        <h3 style="margin:0 0 8px;color:#fff;">Generate Playlist</h3>
        <p style="margin:0 0 14px;color:#a1a1aa;line-height:1.6;">Select a mathematical genre to generate a high-precision pure lane playlist.</p>
        <form method="post" action="/generate-playlist" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;align-items:end;">
          <input type="hidden" name="playlist_mode" value="pure">
          <label style="display:grid;gap:6px;font-size:13px;color:#a1a1aa;">
            Mathematical Genre
            <select name="cluster_id" style="padding:10px 12px;border:1px solid rgba(255,255,255,0.2);border-radius:10px;background:#1e293b;color:#fff;">
              {''.join(options)}
            </select>
          </label>
          <label style="display:grid;gap:6px;font-size:13px;color:#a1a1aa;">
            Track Count
            <input name="playlist_size" type="number" min="5" max="50" value="{selected_playlist_size}" style="padding:10px 12px;border:1px solid rgba(255,255,255,0.2);border-radius:10px;background:#1e293b;color:#fff;">
          </label>
          <label style="display:grid;gap:6px;font-size:13px;color:#a1a1aa;">
            Visibility
            <select name="playlist_visibility" style="padding:10px 12px;border:1px solid rgba(255,255,255,0.2);border-radius:10px;background:#1e293b;color:#fff;">
              <option value="private"{" selected" if selected_visibility == "private" else ""}>Private</option>
              <option value="public"{" selected" if selected_visibility == "public" else ""}>Public</option>
            </select>
          </label>
          <button class="btn btn-primary" type="submit" style="height:44px;">Generate Playlist</button>
        </form>
      </div>
    </section>
    """


def _build_scaled_preview(cluster_result: GenreClusterResult) -> pd.DataFrame:
    scaled_columns = [column for column in cluster_result.clustered_frame.columns if column.startswith("scaled_")]
    preview_columns = ["cluster_label", "track_name", *scaled_columns]
    return cluster_result.clustered_frame[preview_columns]


def _analyze_library(
    settings: AppSettings, 
    client: spotipy.Spotify, 
    *, 
    load_to_db: bool,
    progress_callback: Callable[[float, str], None] | None = None
) -> LibraryAnalysis:
    extractor = SpotifySavedTracksExtractor(client, settings.rapidapi)
    extraction = extractor.extract(progress_callback)
    if extraction.dataframe.empty:
        return LibraryAnalysis(
            records_loaded=0,
            dataset_name=settings.postgres.dataset_name,
            raw_preview=extraction.dataframe,
            scaled_preview=None,
            cluster_result=None,
            audio_features_warning=extraction.audio_features_warning,
        )

    load_result = None
    if load_to_db:
        if progress_callback:
            progress_callback(0.92, "Syncing to Postgres database...")
        pipeline = SpotifyLibraryPipeline(settings.postgres)
        load_result = pipeline.load_saved_tracks(extraction.records)

    warning_message = extraction.audio_features_warning
    cluster_result: GenreClusterResult | None = None
    scaled_preview: pd.DataFrame | None = None

    try:
        if progress_callback:
            progress_callback(0.95, "Running K-Means cluster analysis...")
        normalizer = FeatureNormalizer()
        feature_set = normalizer.fit_transform(extraction.dataframe)
        cluster_result = MathematicalGenreFinder().fit(feature_set)
        scaled_preview = _build_scaled_preview(cluster_result)
    except (FeatureEngineeringError, RecommendationError) as exc:
        warning_message = f"{warning_message} {exc}".strip() if warning_message else str(exc)

    if progress_callback:
        progress_callback(1.0, "Analysis complete!")

    return LibraryAnalysis(
        records_loaded=load_result.loaded_rows if load_result is not None else len(extraction.records),
        dataset_name=load_result.dataset_name if load_result is not None else settings.postgres.dataset_name,
        raw_preview=extraction.dataframe,
        scaled_preview=scaled_preview,
        cluster_result=cluster_result,
        audio_features_warning=warning_message,
    )


def _dashboard_html(
    profile: dict[str, Any],
    *,
    info: str | None = None,
    warning: str | None = None,
    error: str | None = None,
    loaded_rows: int | None = None,
    dataset_name: str | None = None,
    raw_preview: pd.DataFrame | None = None,
    scaled_preview: pd.DataFrame | None = None,
    cluster_result: GenreClusterResult | None = None,
    generated_playlist_tracks: list[dict[str, str]] | None = None,
    generated_playlist_name: str | None = None,
    selected_cluster_id: int | None = None,
    selected_playlist_mode: str = "pure",
    selected_playlist_size: int = 20,
    selected_visibility: str = "private",
    vibe_data: dict[str, Any] | None = None,
) -> str:
    alerts: list[str] = []
    if info:
        alerts.append(f'<div style="background:#ecfeff;border:1px solid #a5f3fc;padding:12px;border-radius:8px;color:#000;">{html.escape(info)}</div>')
    if warning:
        alerts.append(
            f'<div style="background:#fffbeb;border:1px solid #fde68a;padding:12px;border-radius:8px;color:#000;">{html.escape(warning)}</div>'
        )
    if error:
        alerts.append(
            f'<div style="background:#fef2f2;border:1px solid #fecaca;padding:12px;border-radius:8px;color:#000;">{html.escape(error)}</div>'
        )
    
    if generated_playlist_tracks is not None:
        track_items = "".join(
            f'<li style="margin-bottom:6px;"><a href="{html.escape(t["url"], quote=True)}" target="_blank" style="color:#16a34a;text-decoration:none;font-weight:600;">{html.escape(t["name"])}</a></li>'
            for t in generated_playlist_tracks
        )
        csv_content = "Track Name,Spotify URL\n" + "\n".join(f'"{t["name"]}","{t["url"]}"' for t in generated_playlist_tracks)
        csv_data_uri = f"data:text/csv;charset=utf-8,{html.escape(csv_content, quote=True)}"
        
        generated_list_html = f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;padding:16px;border-radius:8px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <h3 style="margin:0;color:#166534;">Generated: {html.escape(generated_playlist_name or 'Playlist')}</h3>
                <a href="{csv_data_uri}" download="{html.escape(generated_playlist_name or 'playlist')}.csv" class="btn btn-primary" style="font-size:13px;padding:6px 12px;text-decoration:none;">Download CSV</a>
            </div>
            <ul style="margin:0;padding-left:20px;color:#334155;">
                {track_items}
            </ul>
        </div>
        """
        alerts.append(generated_list_html)

    metric_html = ""

    viz_html = ""
    if vibe_data:
        import json
        seeds = vibe_data.get("seeds", [])
        anchor = vibe_data.get("anchor", {})
        recommendations = vibe_data.get("recommendations", [])
        
        viz_json = json.dumps({
            "seeds": seeds,
            "anchor": anchor,
            "recommendations": recommendations
        })
        
        viz_html = """
        <div class="card" style="background:rgba(0,0,0,0.3); border-color:rgba(255,255,255,0.1); padding:20px; border-radius:12px; margin-top:20px; overflow:hidden;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                <h2 style="margin:0; color:#f8fafc; font-size:20px; font-weight:800;">KNN Sonic Landscape</h2>
                <div style="display:flex; gap:12px; font-size:11px; color:#94a3b8;">
                    <div style="display:flex; align-items:center; gap:4px;"><div style="width:8px; height:8px; border-radius:50%; background:#fff;"></div> Seed</div>
                    <div style="display:flex; align-items:center; gap:4px;"><div style="width:8px; height:8px; border-radius:50%; background:#1db954;"></div> Neighbor</div>
                </div>
            </div>
            <div id="sonic-target-container" style="position:relative; width:100%; height:500px; background:radial-gradient(circle at center, #111827 0%, #000 100%); border-radius:8px; overflow:hidden; border: 1px solid rgba(255,255,255,0.05);">
                <canvas id="sonic-target-canvas"></canvas>
                <div style="position:absolute; bottom:10px; right:10px; color:#64748b; font-size:11px; pointer-events:none;">High Valence &rarr;</div>
                <div style="position:absolute; top:10px; left:10px; color:#64748b; font-size:11px; pointer-events:none;">&uarr; High Energy</div>
            </div>
        </div>
        
        <script>
        (function() {
            const data = """ + viz_json + """;
            const container = document.getElementById('sonic-target-container');
            const canvas = document.getElementById('sonic-target-canvas');
            const ctx = canvas.getContext('2d');
            
            let width, height;
            function resize() {
                width = container.offsetWidth;
                height = container.offsetHeight;
                canvas.width = width * window.devicePixelRatio;
                canvas.height = height * window.devicePixelRatio;
                canvas.style.width = width + 'px';
                canvas.style.height = height + 'px';
                ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            }
            window.addEventListener('resize', resize);
            resize();
            
            const nodes = [];
            
            // Map tracks to coordinates based on Energy and Valence
            // Energy -> Y (inverted), Valence -> X
            data.seeds.forEach((s) => {
                const x = 50 + (s.audio_features.valence - data.anchor.valence) * 300;
                const y = 50 - (s.audio_features.energy - data.anchor.energy) * 300;
                nodes.push({
                    type: 'seed',
                    targetX: (s.audio_features.valence * 0.8 + 0.1) * width,
                    targetY: (1 - (s.audio_features.energy * 0.8 + 0.1)) * height,
                    x: Math.random() * width,
                    y: Math.random() * height,
                    label: s.name,
                    artist: s.artists[0].name,
                    color: '#fff',
                    size: 8,
                    glow: 15
                });
            });
            
            data.recommendations.forEach((r) => {
                nodes.push({
                    type: 'neighbor',
                    targetX: (r.audio_features.valence * 0.8 + 0.1) * width,
                    targetY: (1 - (r.audio_features.energy * 0.8 + 0.1)) * height,
                    x: Math.random() * width,
                    y: Math.random() * height,
                    label: r.name,
                    artist: r.artists[0].name,
                    url: 'https://open.spotify.com/track/' + r.id,
                    color: '#1db954',
                    size: 5,
                    glow: 10
                });
            });
            
            function draw() {
                ctx.fillStyle = 'rgba(0,0,0,0.2)';
                ctx.fillRect(0, 0, width, height);
                
                // Draw grid lines
                ctx.strokeStyle = 'rgba(255,255,255,0.03)';
                ctx.lineWidth = 1;
                for(let i=1; i<10; i++) {
                    ctx.beginPath(); ctx.moveTo(i*width/10, 0); ctx.lineTo(i*width/10, height); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(0, i*height/10); ctx.lineTo(width, i*height/10); ctx.stroke();
                }

                // Smoothly animate nodes to their targets
                nodes.forEach(n => {
                    n.x += (n.targetX - n.x) * 0.05;
                    n.y += (n.targetY - n.y) * 0.05;
                    
                    // Draw connections for seeds
                    if (n.type === 'seed') {
                        nodes.forEach(m => {
                            if (m.type === 'neighbor') {
                                const dist = Math.sqrt((n.x-m.x)**2 + (n.y-m.y)**2);
                                if (dist < 150) {
                                    ctx.beginPath();
                                    ctx.strokeStyle = `rgba(29, 185, 84, ${0.2 * (1 - dist/150)})`;
                                    ctx.moveTo(n.x, n.y);
                                    ctx.lineTo(m.x, m.y);
                                    ctx.stroke();
                                }
                            }
                        });
                    }

                    ctx.shadowBlur = n.glow;
                    ctx.shadowColor = n.color;
                    ctx.fillStyle = n.color;
                    ctx.beginPath();
                    ctx.arc(n.x, n.y, n.size, 0, Math.PI * 2);
                    ctx.fill();
                    
                    ctx.shadowBlur = 0;
                    if (n.type === 'seed') {
                        ctx.fillStyle = 'rgba(255,255,255,0.8)';
                        ctx.font = '600 11px Inter';
                        ctx.textAlign = 'center';
                        ctx.fillText(n.label, n.x, n.y + n.size + 15);
                    }
                });
                
                requestAnimationFrame(draw);
            }
            draw();
            
            canvas.onclick = (e) => {
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                
                nodes.forEach(n => {
                    const dx = n.x - mx;
                    const dy = n.y - my;
                    if (Math.sqrt(dx*dx + dy*dy) < 15 && n.url) {
                        window.open(n.url, '_blank');
                    }
                });
            };
        })();
        </script>

        <div style="margin-top:40px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
            <div>
                <h3 style="color:#f8fafc; font-size:18px; font-weight:700; margin-bottom:20px; display:flex; align-items:center; gap:8px;">
                    <span class="material-symbols-outlined text-[#1db954]">hub</span>
                    Nearest Neighbors (KNN)
                </h3>
                <div style="display:grid; gap:12px;">
                    """ + "".join([f'''
                    <a href="https://open.spotify.com/track/{r['id']}" target="_blank" 
                       class="card" style="display:flex; align-items:center; gap:12px; padding:10px; background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; text-decoration:none; transition:all 0.2s; overflow:hidden;">
                        <img src="{r['album']['images'][0]['url'] if r.get('album', {}).get('images') else ''}" 
                             style="width:40px; height:40px; border-radius:4px; object-fit:cover; background:#222;" alt="Album Art">
                        <div style="flex:1; min-width:0;">
                            <div style="color:#fff; font-weight:600; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{r['name']}</div>
                            <div style="color:#94a3b8; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{r['artists'][0]['name']}</div>
                        </div>
                    </a>
                    ''' for r in recommendations[:10]]) + """
                </div>
            </div>
            <div>
                <h3 style="color:#f8fafc; font-size:18px; font-weight:700; margin-bottom:20px; display:flex; align-items:center; gap:8px;">
                    <span class="material-symbols-outlined text-[#1db954]">person_search</span>
                    Artist Discovery (3/Artist)
                </h3>
                <div style="display:grid; gap:12px;">
                    """ + "".join([f'''
                    <a href="https://open.spotify.com/track/{r['id']}" target="_blank" 
                       class="card" style="display:flex; align-items:center; gap:12px; padding:10px; background:rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; text-decoration:none; transition:all 0.2s; overflow:hidden;">
                        <img src="{r['album']['images'][0]['url'] if r.get('album', {}).get('images') else ''}" 
                             style="width:40px; height:40px; border-radius:4px; object-fit:cover; background:#222;" alt="Album Art">
                        <div style="flex:1; min-width:0;">
                            <div style="color:#fff; font-weight:600; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{r['name']}</div>
                            <div style="color:#94a3b8; font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{r['artists'][0]['name']}</div>
                        </div>
                    </a>
                    ''' for r in vibe_data.get('artist_discovery', [])]) + """
                </div>
            </div>
        </div>
        """
    elif cluster_result is not None and not cluster_result.clustered_frame.empty:
        import json
        viz_data = cluster_result.clustered_frame
        chart_data = viz_data[["track_name", "artist_name", "cluster_id", "valence", "energy"]].to_dict(orient="records")
        chart_json = json.dumps(chart_data)
        
        legend_items = []
        for summary in cluster_result.summaries:
            color = ['#10b981', '#a78bfa'][summary.cluster_id % 2]
            legend_items.append(f'<div style="display:flex;align-items:center;gap:6px;"><div style="width:10px;height:10px;border-radius:50%;background:{color};box-shadow:0 0 5px {color};"></div><span style="font-size:11px;color:#94a3b8;">{html.escape(summary.label)}</span></div>')
        
        viz_html = f"""
        <div class="card" id="knn-card" style="background:rgba(0,0,0,0.3); border-color:rgba(255,255,255,0.1); padding:20px; overflow:hidden; margin-top:20px; border-radius:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:10px;">
            <div style="display:flex; align-items:center; gap:12px;">
              <h2 style="margin:0; color:#f8fafc; font-size:20px; font-weight:800;">
                Your Acoustic Landscape
              </h2>
              <button onclick="shareLandscape()" class="btn" style="background:#1db954; color:#000; font-size:12px; padding:4px 10px; display:flex; align-items:center; gap:4px; border:0; cursor:pointer; border-radius:6px; font-weight:700;">
                <span class="material-symbols-outlined" style="font-size:16px;">share</span> Share
              </button>
            </div>
            <div style="display:flex;gap:14px;">
              {''.join(legend_items)}
            </div>
          </div>
          <div style="position:relative; width:100%; height:400px; background:radial-gradient(circle at center, #111827 0%, #000 100%); border-radius:8px; border:1px solid rgba(255,255,255,0.1); overflow:hidden;" id="knn-container">
            <!-- Grid lines -->
            <div style="position:absolute; top:50%; left:0; right:0; height:1px; background:rgba(255,255,255,0.05);"></div>
            <div style="position:absolute; top:0; bottom:0; left:50%; width:1px; background:rgba(255,255,255,0.05);"></div>
            
            <!-- Axis Labels -->
            <span style="position:absolute; bottom:10px; right:10px; color:#64748b; font-size:11px;">High Valence &rarr;</span>
            <span style="position:absolute; top:10px; left:10px; color:#64748b; font-size:11px;">&uarr; High Energy</span>
            
            <!-- JS renders nodes here -->
          </div>
        </div>
        
        <script>
          (function() {{
            const data = {chart_json};
            const container = document.getElementById('knn-container');
            const colors = ['#10b981', '#a78bfa'];
            const REPEL_RADIUS = 60;
            const REPEL_STRENGTH = 18;
            const RETURN_SPEED = 0.08;
            const nodes = [];
            let mouseX = -1000, mouseY = -1000;
            let raf;
            
            data.forEach((point) => {{
              const node = document.createElement('div');
              node.className = 'node';
              
              const homeLeft = 5 + (point.valence * 90);
              const homeTop = 95 - (point.energy * 90);
              
              node.style.left = homeLeft + '%';
              node.style.top = homeTop + '%';
              
              const color = colors[point.cluster_id % colors.length];
              node.style.backgroundColor = color;
              node.style.boxShadow = '0 0 8px ' + color;
              node.dataset.color = color;
              
              const tooltip = document.createElement('div');
              tooltip.className = 'node-tooltip';
              tooltip.innerHTML = '<strong>' + point.track_name.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</strong><br/><span style="color:#94a3b8">' + point.artist_name.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</span><br/><span style="color:' + color + '; font-size:10px; margin-top:4px; display:inline-block;">Genre ' + (point.cluster_id + 1) + '</span>';
              
              node.appendChild(tooltip);
              container.appendChild(node);
              
              nodes.push({{
                el: node,
                homeX: homeLeft,
                homeY: homeTop,
                offsetX: 0,
                offsetY: 0,
                color: color
              }});
            }});
            
            container.addEventListener('mousemove', (e) => {{
              const rect = container.getBoundingClientRect();
              mouseX = e.clientX - rect.left;
              mouseY = e.clientY - rect.top;
            }});
            
            container.addEventListener('mouseleave', () => {{
              mouseX = -1000;
              mouseY = -1000;
            }});
            
            function tick() {{
              const cw = container.offsetWidth;
              const ch = container.offsetHeight;
              
              nodes.forEach((n) => {{
                const cx = (n.homeX / 100) * cw + n.offsetX;
                const cy = (n.homeY / 100) * ch + n.offsetY;
                const dx = cx - mouseX;
                const dy = cy - mouseY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < REPEL_RADIUS && dist > 0) {{
                  const force = (1 - dist / REPEL_RADIUS) * REPEL_STRENGTH;
                  n.offsetX += (dx / dist) * force;
                  n.offsetY += (dy / dist) * force;
                }}
                
                n.offsetX *= (1 - RETURN_SPEED);
                n.offsetY *= (1 - RETURN_SPEED);
                
                if (Math.abs(n.offsetX) < 0.01) n.offsetX = 0;
                if (Math.abs(n.offsetY) < 0.01) n.offsetY = 0;
                
                n.el.style.transform = 'translate(calc(-50% + ' + n.offsetX.toFixed(1) + 'px), calc(-50% + ' + n.offsetY.toFixed(1) + 'px))';
                
                if (dist < REPEL_RADIUS) {{
                  const glow = Math.round((1 - dist / REPEL_RADIUS) * 15) + 8;
                  n.el.style.boxShadow = '0 0 ' + glow + 'px ' + n.color;
                }} else {{
                  n.el.style.boxShadow = '0 0 8px ' + n.color;
                }}
              }});
              
              raf = requestAnimationFrame(tick);
            }}
            
            tick();

            window.shareLandscape = async function() {{
              const card = document.getElementById('knn-card');
              const btn = event.currentTarget;
              const originalText = btn.innerHTML;
              
              try {{
                btn.innerHTML = 'Capturing...';
                btn.disabled = true;
                
                const canvas = await html2canvas(card, {{
                  backgroundColor: '#064e3b',
                  scale: 2,
                  useCORS: true,
                  logging: false
                }});
                
                const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
                const file = new File([blob], 'my-acoustic-landscape.png', {{ type: 'image/png' }});
                
                if (navigator.share && navigator.canShare && navigator.canShare({{ files: [file] }})) {{
                  await navigator.share({{
                    title: 'My Acoustic Landscape',
                    text: 'Check out my musical identity on GenreSense! 🎵',
                    url: window.location.origin,
                    files: [file]
                  }});
                }} else {{
                  // Fallback for desktop: Download + copy link
                  const url = canvas.toDataURL('image/png');
                  const link = document.createElement('a');
                  link.download = 'my-acoustic-landscape.png';
                  link.href = url;
                  link.click();
                  
                  await navigator.clipboard.writeText(window.location.origin);
                  alert('Landscape downloaded! Share it on social media with this link: ' + window.location.origin + ' (Link copied to clipboard)');
                }}
              }} catch (err) {{
                console.error('Share failed:', err);
                alert('Could not share. You can try taking a screenshot!');
              }} finally {{
                btn.innerHTML = originalText;
                btn.disabled = false;
              }}
            }};
          }})();
        </script>
        """

    raw_table = ""
    if raw_preview is not None and not raw_preview.empty:
        raw_table = "<h3>Raw Feature Snapshot</h3>" + raw_preview.head(25).to_html(index=False, border=0)

    scaled_table = ""
    if scaled_preview is not None and not scaled_preview.empty:
        scaled_table = "<h3>Scaled Feature Matrix</h3>" + scaled_preview.head(25).to_html(index=False, border=0)

    genre_html = ""
    if cluster_result is not None and cluster_result.summaries:
        default_cluster_id = cluster_result.summaries[0].cluster_id if selected_cluster_id is None else selected_cluster_id
        genre_html = _genre_summary_html(
            cluster_result,
            selected_cluster_id=default_cluster_id,
            selected_playlist_mode=selected_playlist_mode,
            selected_playlist_size=selected_playlist_size,
            selected_visibility=selected_visibility,
        )

    ad_slot_html = ""
    if os.environ.get("ENABLE_ADS", "false").lower() == "true":
        ad_slot_html = """
        <!-- Dashboard Ad Slot -->
        <div style="margin-top:40px; padding:20px; background:rgba(0,0,0,0.25); border:1px dashed rgba(255,255,255,0.1); border-radius:12px; text-align:center;">
            <iframe data-aa='2436572' src='//acceptable.a-ads.com/2436572/?size=Adaptive'
                    style='border:0; padding:0; width:100%; height:90px; overflow:hidden; display:block; margin:auto;'
                    allowtransparency="true"></iframe>
        </div>
        """

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>GenreSense Dashboard</title>
      <style>
        body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #064e3b; color: #fff; }}
        .wrap {{ max-width: 1100px; margin: 0 auto; padding: 26px 18px 42px; }}
        .top {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px; margin-bottom: 20px; }}
        .btn {{ display:inline-block; padding:10px 14px; border-radius:8px; text-decoration:none; font-weight:600; transition: all 0.2s; }}
        .btn-primary {{ background:#1db954; color:#000; border:0; cursor:pointer; }}
        .btn-primary:hover {{ background:#1ed760; transform: translateY(-1px); }}
        .btn-outline {{ border:1px solid rgba(255,255,255,0.2); color:#fff; background:transparent; }}
        .btn-outline:hover {{ background:rgba(255,255,255,0.05); }}
        table {{ width:100%; border-collapse: collapse; background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.1); border-radius: 8px; overflow:hidden; color:#fff; }}
        th, td {{ padding: 12px 14px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px; text-align: left; }}
        th {{ background: rgba(0,0,0,0.2); font-weight: 600; color: #1db954; }}
        .node {{
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            cursor: pointer;
            z-index: 10;
            will-change: transform, box-shadow;
        }}
        .node:hover {{
            transform: translate(-50%, -50%) scale(2.2) !important;
            filter: brightness(1.5);
            z-index: 50;
        }}
        .node .node-tooltip {{
            visibility: hidden;
            opacity: 0;
            position: absolute;
            bottom: 15px;
            left: 50%;
            transform: translateX(-50%);
            background: #000;
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 11px;
            white-space: nowrap;
            transition: opacity 0.2s ease, bottom 0.2s ease;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            pointer-events: none;
        }}
        .node:hover .node-tooltip {{
            visibility: visible;
            opacity: 1;
            bottom: 20px;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div style="text-align:center; margin-bottom:32px;">
          <h1 style="margin:0; font-size:42px; font-weight:900; letter-spacing:-0.04em; color:#fff;">GenreSense</h1>
          <p style="margin:8px 0 0; color:#a1a1aa; font-size:14px;">Authenticated as {html.escape(profile.get("display_name") or profile.get("id") or "Spotify user")}.</p>
        </div>
        <div class="top">
          <div style="display:flex;gap:10px;justify-content:flex-end;width:100%;">
            <a class="btn btn-outline" href="/logout">Disconnect</a>
            <form method="post" action="/run-pipeline" style="margin:0;">
              <button class="btn btn-primary" type="submit">Refresh Mathematical Genres</button>
            </form>
          </div>
        </div>
        <div style="margin-top:14px;display:grid;gap:10px;">
          {''.join(alerts)}
        </div>
        {genre_html}
        {viz_html}
        {ad_slot_html}
      </div>
    </body>
    </html>
    """


def _starting_pipeline_html(profile: dict[str, Any], auto_submit: bool = False) -> str:
    display_name = profile.get("display_name") or profile.get("id") or "Spotify user"
    
    auto_submit_html = ""
    if auto_submit:
        auto_submit_html = """
        <form id="start-process-form" method="post" action="/run-pipeline" style="display:none;"></form>
        <script>
          window.addEventListener("load", () => {
            const form = document.getElementById("start-process-form");
            if (form) form.submit();
          });
        </script>
        """

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Analyzing Library - GenreSense</title>
      <style>
        body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #064e3b; color: #fff; }}
        .wrap {{ min-height: 100vh; display: grid; place-items: center; padding: 24px; }}
        .card {{ width: min(560px, 100%); background: #000; border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 40px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }}
        .progress-container {{ width: 100%; height: 8px; background: #1e293b; border-radius: 999px; margin: 24px 0; overflow: hidden; }}
        #progress-bar {{ width: 5%; height: 100%; background: #1db954; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 15px rgba(29, 185, 84, 0.5); }}
        .status-text {{ font-size: 14px; color: #94a3b8; font-weight: 500; min-height: 20px; }}
        .spinner {{ width: 24px; height: 24px; border-radius: 999px; border: 3px solid rgba(29, 185, 84, 0.2); border-top-color: #1db954; animation: spin 0.8s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
      </style>
    </head>
    <body>
      <div class="wrap" id="main-content">
        <div class="card" id="progress-card">
          <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
            <div class="spinner"></div>
            <div style="font-weight: 800; font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; color: #1db954;">Processing Taste Engine</div>
          </div>
          <h1 style="margin:0 0 8px; font-size:28px; font-weight: 800; letter-spacing: -0.02em;">Auditing your library...</h1>
          <p style="margin:0; color:#94a3b8; font-size:15px; line-height: 1.6;">Extracting audio features and identifying behavioral patterns.</p>
          
          <div class="progress-container">
            <div id="progress-bar"></div>
          </div>
          
          <div class="status-text" id="status-message">Initializing Spotify connection...</div>
        </div>
      </div>

      {auto_submit_html}

      <script>
        function updateProgress(percent, message) {{
          const bar = document.getElementById('progress-bar');
          const status = document.getElementById('status-message');
          if (bar) bar.style.width = percent + '%';
          if (status) status.innerText = message;
        }}

        // #region agent log
        window.addEventListener("load", () => {{
          try {{
            const img = new Image();
            img.src = "/__beacon?event=start_load&ts=" + Date.now();
          }} catch (e) {{}}
        }});
        // #endregion
      </script>
    </body>
    </html>
    """


@app.get("/__beacon")
def beacon() -> Any:
    # region agent log
    event = request.args.get("event", "unknown")
    _agent_log(
        hypothesis_id="F",
        message="beacon",
        data={
            "event": event,
            "ua": request.headers.get("User-Agent", "")[:120],
            "referer": request.headers.get("Referer"),
            "has_session_token": TOKEN_INFO_KEY in session,
        },
        run_id="pre",
    )
    # endregion
    return ("", 204)


@app.get("/")
def index() -> Any:
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    status = session.pop("status_message", "Enter 5 songs to find your musical DNA.")
    return _render_landing(status)


@app.get("/ads.txt")
def ads_txt() -> Any:
    ads_file = ROOT / "ads.txt"
    if os.environ.get("ENABLE_ADS", "false").lower() == "true" and ads_file.exists():
        return Response(ads_file.read_text(), mimetype="text/plain")
    return "Not Found", 404


@app.post("/analyze-vibe")
def analyze_vibe() -> Any:
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    track_queries = [
        request.form.get("track_1", ""),
        request.form.get("track_2", ""),
        request.form.get("track_3", ""),
        request.form.get("track_4", ""),
        request.form.get("track_5", ""),
    ]
    client = _get_client_credentials_client(settings)
    engine = SonicRecommendationEngine(client, settings.rapidapi)
    
    seeds = engine.resolve_seeds(track_queries)
    if not seeds:
        session["status_message"] = "Could not find any of those tracks. Try being more specific!"
        return redirect(url_for("index"))
    
    anchor, recommendations = engine.recommend(seeds)
    if not recommendations:
        session["status_message"] = "I could not find recommendation matches in the standing Spotify dataset. Try songs by artists represented in the catalog."
        return redirect(url_for("index"))

    artist_discovery = engine.get_artist_discovery_playlist(seeds)
    print(f"DEBUG: Found {len(seeds)} seeds, {len(recommendations)} recommendations, and {len(artist_discovery)} artist discovery tracks.")
    
    # Minimize data stored in session to avoid 4KB cookie limit
    def minimize_track(t):
        return {
            "id": t["id"],
            "name": t["name"],
            "artists": [{"name": t["artists"][0]["name"]}],
            "album": {"images": [{"url": t["album"]["images"][0]["url"]}]} if t.get("album", {}).get("images") else {},
            "audio_features": {
                "energy": t["audio_features"].get("energy"),
                "valence": t["audio_features"].get("valence"),
                "danceability": t["audio_features"].get("danceability")
            } if "audio_features" in t else {}
        }

    vibe_data = {
        "seeds": [minimize_track(s) for s in seeds],
        "anchor": anchor,
        "recommendations": [minimize_track(r) for r in recommendations],
        "artist_discovery": [minimize_track(ad) for ad in artist_discovery]
    }
    
    session["vibe_data"] = vibe_data
    session["analysis_just_finished"] = True
    return redirect(url_for("dashboard_view"))


@app.get("/search-tracks")
def search_tracks() -> Any:
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return {"tracks": []}
    
    try:
        settings = _settings_from_env()
        client = _get_client_credentials_client(settings)
        results = client.search(q=query, type="track", limit=5)
        tracks = []
        for t in results.get("tracks", {}).get("items", []):
            tracks.append({
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "display": f"{t['name']} - {t['artists'][0]['name']}"
            })
        return {"tracks": tracks}
    except Exception as exc:
        return {"tracks": [], "error": str(exc)}


@app.get("/logout")
def logout() -> Any:
    session.clear()
    return redirect(url_for("index"))


@app.post("/run-pipeline")
def run_pipeline() -> Any:
    """
    Refactored for serverless stability:
    1. Executes pipeline synchronously (with streaming response for UI feedback).
    2. Vercel will not terminate this request until it finishes (up to timeout).
    3. Persists results to Postgres immediately.
    """
    _agent_log(
        hypothesis_id="C",
        message="POST /run-pipeline entry (synchronous)",
        data={
            "has_session_token": TOKEN_INFO_KEY in session,
            "scheme": request.scheme,
        },
    )
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        session["status_message"] = "Your Spotify session expired. Please connect again."
        return redirect(url_for("index"))

    def generate():
        _agent_log(hypothesis_id="C", message="pipeline stream start", data={}, run_id="pre")
        yield _starting_pipeline_html(profile, auto_submit=False) + (" " * 4096) + "\n"

        def progress_callback(percent: float, message: str):
            yield f"<script>updateProgress({percent * 100}, {json.dumps(message)});</script>\n" + (" " * 1024)

        try:
            # Execute analysis directly in the request thread
            # stream_with_context allows yielding progress to the browser
            analysis = _analyze_library(settings, client, load_to_db=True, progress_callback=progress_callback)
            
            # Since _analyze_library doesn't yield, we simulate progress milestones if needed
            # Or refactor _analyze_library to accept a yielding callback.
            # For now, we run it and then update the UI.
            
            # Persist to Postgres
            persistence = _get_persistence()
            if persistence:
                user_id = profile.get("id", "default_user")
                persistence.save(user_id, analysis)
                _agent_log(hypothesis_id="C", message="pipeline results persisted", data={"user_id": user_id})

            # For debug: we bypass load on external devices, but keep session flag
            session["analysis_just_finished"] = True
            yield "<script>window.location.href = '/dashboard';</script>\n"
            
        except Exception as exc:
            _agent_log(hypothesis_id="C", message="pipeline execution failed", data={"exc": repr(exc)}, run_id="pre")
            session["status_message"] = f"Pipeline error: {repr(exc)}"
            yield "<script>window.location.href = '/';</script>"

    return Response(stream_with_context(generate()), mimetype='text/html')


@app.post("/generate-playlist")
def generate_playlist() -> Any:
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        session["status_message"] = "Your Spotify session expired. Please connect again."
        return redirect(url_for("index"))

    try:
        cluster_id = int(request.form.get("cluster_id", "0"))
        playlist_mode = str(request.form.get("playlist_mode", "pure")).strip().lower()
        playlist_size = max(5, min(int(request.form.get("playlist_size", "20")), 50))
        is_public = request.form.get("playlist_visibility", "private") == "public"
    except ValueError:
        return _dashboard_html(profile, error="Playlist options were invalid. Please try again.")

    try:
        analysis = _analyze_library(settings, client, load_to_db=False)
        if analysis.cluster_result is None:
            return _dashboard_html(profile, warning="No clusterable tracks are available yet. Run analysis again after loading saved tracks.")

        if playlist_mode == "discovery":
            from genresense.recommendations import DiscoveryPlaylistBuilder
            playlist_name, generated_tracks, seed_track = DiscoveryPlaylistBuilder(client, settings.rapidapi).build(
                analysis.cluster_result,
                analysis.raw_preview,
                cluster_id=cluster_id,
                limit=playlist_size,
            )
        else:
            blueprint = PlaylistBuilder().build(
                analysis.cluster_result,
                cluster_id=cluster_id,
                mode=playlist_mode,
                limit=playlist_size,
            )
            playlist_name = f"GenreSense {blueprint.cluster_label} {'Pure' if blueprint.mode == 'pure' else 'Hybrid'}"
            seed_track = blueprint.seed_track_name
            
            generated_tracks = []
            if analysis.raw_preview is not None and not analysis.raw_preview.empty:
                for track_id in blueprint.track_ids:
                    row = analysis.raw_preview[analysis.raw_preview["track_id"] == track_id]
                    if not row.empty:
                        track_name = str(row.iloc[0].get("track_name", "Unknown Track"))
                        generated_tracks.append({
                            "id": track_id,
                            "name": track_name,
                            "url": f"https://open.spotify.com/track/{track_id}"
                        })

    except Exception as exc:  # noqa: BLE001
        return _dashboard_html(profile, error=f"Playlist generation failed: {exc}")

    return _dashboard_html(
        profile,
        info=f"Successfully generated {playlist_name} using {seed_track} as the seed track.",
        warning=analysis.audio_features_warning,
        loaded_rows=analysis.records_loaded,
        dataset_name=analysis.dataset_name,
        raw_preview=analysis.raw_preview,
        scaled_preview=analysis.scaled_preview,
        cluster_result=analysis.cluster_result,
        generated_playlist_tracks=generated_tracks,
        generated_playlist_name=playlist_name,
        selected_cluster_id=cluster_id,
        selected_playlist_mode=playlist_mode,
        selected_playlist_size=playlist_size,
        selected_visibility="public" if is_public else "private",
    )


@app.get("/dashboard")
def dashboard_view() -> Any:
    settings = _settings_from_env()
    
    # If we have vibe_data in session, use it
    vibe_data = session.get("vibe_data")
    if vibe_data:
        # Create a dummy profile for the dashboard
        profile = {"display_name": "Music Explorer", "id": "anonymous"}
        return _dashboard_html(
            profile,
            info="Your Sonic Target is ready.",
            vibe_data=vibe_data
        )

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        return redirect(url_for("index"))
    return _dashboard_html(profile, warning="No recommendation data is available yet. Enter 5 songs to build your playlist.")


@app.get("/debug-config")
def debug_config() -> Any:
    settings = _settings_from_env()
    return {
        "configured_redirect_uri": settings.spotify.redirect_uri,
        "request_host": request.headers.get("Host"),
        "request_scheme": request.scheme,
        "x_forwarded_proto": request.headers.get("X-Forwarded-Proto"),
        "client_id_prefix": settings.spotify.client_id[:5] if settings.spotify.client_id else None,
    }



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True, use_reloader=False)

