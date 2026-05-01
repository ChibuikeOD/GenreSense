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
from spotipy.oauth2 import SpotifyOAuth

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genresense.config import AppSettings, ConfigurationError
from genresense.features import FeatureEngineeringError, FeatureNormalizer
from genresense.pipeline import SpotifyLibraryPipeline
from genresense.recommendations import GenreClusterResult, MathematicalGenreFinder, PlaylistBuilder, RecommendationError
from genresense.spotify_client import SpotifyPlaylistPublisher, SpotifySavedTracksExtractor

load_dotenv(ROOT / ".env", override=True)


LANDING_TEMPLATE_PATH = ROOT / "UI" / "landing.html"
TOKEN_INFO_KEY = "spotify_token_info"
STATE_KEY = "spotify_oauth_state"
PROFILE_KEY = "spotify_profile"
SPOTIPY_CACHE_PATH = ROOT / ".spotify_cache"

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
    return CacheFileHandler(cache_path=str(SPOTIPY_CACHE_PATH))

# Store the latest analysis in memory to allow a clean redirect after the streaming pipeline finishes.
# In a production app, this would be in Redis or a database.
ANALYSIS_CACHE: dict[str, LibraryAnalysis] = {}

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
            "client_id": os.environ.get("SPOTIFY_CLIENT_ID"),
            "client_secret": os.environ.get("SPOTIFY_CLIENT_SECRET"),
            "redirect_uri": os.environ.get("SPOTIFY_REDIRECT_URI"),
            "scope": os.environ.get("SPOTIFY_SCOPE", "user-library-read"),
        },
        "postgres": {
            "dsn": os.environ.get("POSTGRES_DSN"),
            "dataset_name": os.environ.get("POSTGRES_DATASET_NAME", "spotify_audit"),
        },
        "rapidapi": {
            "api_key": os.environ.get("RAPIDAPI_KEY"),
            "api_host": os.environ.get("RAPIDAPI_HOST", "spotify-extended-audio-features-api.p.rapidapi.com"),
            "base_url": os.environ.get("RAPIDAPI_BASE_URL", "https://spotify-extended-audio-features-api.p.rapidapi.com/v1"),
            "timeout_seconds": os.environ.get("RAPIDAPI_TIMEOUT_SECONDS", "20"),
        },
    }
    return AppSettings.from_mapping(mapping)


def _build_oauth(settings: AppSettings) -> SpotifyOAuth:
    state = session.get(STATE_KEY)
    if not state:
        state = secrets.token_urlsafe(16)
        session[STATE_KEY] = state

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

    # IMPORTANT: avoid calling /v1/me at page-load time (it can 429 and cause loops).
    profile: dict[str, Any] = {"id": "spotify_user", "display_name": "Spotify user"}
    _agent_log(
        hypothesis_id="E",
        message="_get_authenticated_client done",
        data={"elapsed_ms_total": int((time.time() - _t0) * 1000)},
        run_id="pre",
    )
    return client, profile


def _inject_login_into_landing(template: str, login_url: str, status_message: str) -> str:
    connect_anchor = (
        f'<a href="{html.escape(login_url, quote=True)}" '
        'class="bg-primary-container text-on-primary-container px-5 py-2.5 rounded-full font-semibold text-sm '
        'active:scale-[0.98] transition-transform hover:opacity-90 flex items-center gap-2">'
        '<span class="material-symbols-outlined text-[18px]">brand_awareness</span>Connect to Spotify</a>'
    )
    template = re.sub(
        r"<button[^>]*>\s*<span[^>]*>brand_awareness</span>\s*Connect to Spotify\s*</button>",
        connect_anchor,
        template,
        flags=re.IGNORECASE,
    )

    hero_anchor = (
        f'<a href="{html.escape(login_url, quote=True)}" '
        'class="bg-[#1DB954] text-white px-8 py-4 rounded-full font-semibold text-lg flex items-center gap-3 '
        'shadow-lg shadow-[#1DB954]/20 hover:translate-y-[-1px] transition-all active:scale-[0.98]">'
        '<span class="material-symbols-outlined" style="font-variation-settings: \'FILL\' 1;">music_note</span>'
        "Connect to Spotify</a>"
    )
    template = re.sub(
        r"<button[^>]*>\s*<span[^>]*>music_note</span>\s*Connect to Spotify\s*</button>",
        hero_anchor,
        template,
        flags=re.IGNORECASE,
    )

    status_node = (
        '<p id="auth-status" style="margin-top:14px;color:#64748b;font-size:13px;max-width:560px;line-height:1.45;">'
        f"{html.escape(status_message)}</p>"
    )

    connect_script = f"""
<script>
(() => {{
  const loginUrl = {json.dumps(login_url)};
  const connectNodes = Array.from(document.querySelectorAll("button, a"))
    .filter((node) => /connect\\s+to\\s+spotify/i.test((node.textContent || "").trim()));

  connectNodes.forEach((node) => {{
    if (node.tagName.toLowerCase() === "a") {{
      node.setAttribute("href", loginUrl);
      node.setAttribute("target", "_self");
      return;
    }}
    node.setAttribute("type", "button");
    node.onclick = (event) => {{
      event.preventDefault();
      window.location.assign(loginUrl);
    }};
    node.style.cursor = "pointer";
  }});

  // #region agent log
  window.addEventListener("load", () => {{
    try {{
      const img = new Image();
      img.src = "/__beacon?event=landing_load&ts=" + Date.now();
    }} catch (e) {{}}
  }});
  // #endregion
}})();
</script>
"""

    if "</main>" in template:
        template = template.replace("</main>", status_node + "\n</main>", 1)
    else:
        template = template + status_node

    if "</body>" in template:
        return template.replace("</body>", connect_script + "\n</body>", 1)
    return template + connect_script


def _render_landing(status_message: str) -> str:
    _agent_log(
        hypothesis_id="G",
        message="render landing",
        data={
            "status_message": (status_message or "")[:180],
            "has_session_token": TOKEN_INFO_KEY in session,
            "has_session_state": STATE_KEY in session,
        },
        run_id="pre",
    )
    if not LANDING_TEMPLATE_PATH.exists():
        return "<h1>Landing template missing</h1><p>Create UI/landing.html.</p>"
    template = LANDING_TEMPLATE_PATH.read_text(encoding="utf-8")
    return _inject_login_into_landing(template, url_for("login"), status_message)


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
    if cluster_result is not None and not cluster_result.clustered_frame.empty:
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

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>GenreSense Dashboard</title>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
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
    _agent_log(
        hypothesis_id="A",
        message="GET / entry",
        data={
            "args_keys": sorted(list(request.args.keys())),
            "has_code": "code" in request.args,
            "has_error": "error" in request.args,
            "has_session_token": TOKEN_INFO_KEY in session,
            "has_session_state": STATE_KEY in session,
            "request_is_secure": bool(getattr(request, "is_secure", False)),
            "scheme": request.scheme,
        },
    )
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    if "code" in request.args or "error" in request.args:
        return redirect(url_for("callback", **request.args.to_dict()))

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        status = session.pop("status_message", "Connect your Spotify account to start data ingestion and feature prep.")
        return _render_landing(status)

    return _dashboard_html(profile)


@app.get("/login")
def login() -> Any:
    _agent_log(
        hypothesis_id="G",
        message="GET /login entry",
        data={
            "has_session_state": STATE_KEY in session,
            "has_session_token": TOKEN_INFO_KEY in session,
            "scheme": request.scheme,
        },
        run_id="pre",
    )
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    session[STATE_KEY] = secrets.token_urlsafe(16)
    oauth = _build_oauth(settings)
    authorize_url = oauth.get_authorize_url()
    _agent_log(
        hypothesis_id="G",
        message="redirecting to spotify authorize",
        data={
            "redirect_uri": settings.spotify.redirect_uri,
            "scope": settings.spotify.scope,
            "state_set": bool(session.get(STATE_KEY)),
        },
        run_id="pre",
    )
    return redirect(authorize_url)


@app.get("/callback")
def callback() -> Any:
    _agent_log(
        hypothesis_id="G",
        message="GET /callback entry",
        data={
            "args_keys": sorted(list(request.args.keys())),
            "has_code": bool(request.args.get("code")),
            "has_error": bool(request.args.get("error")),
            "remote_state_present": bool(request.args.get("state")),
            "local_state_present": bool(session.get(STATE_KEY)),
            "scheme": request.scheme,
        },
        run_id="pre",
    )
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    error = request.args.get("error")
    if error:
        session["status_message"] = f"Spotify authorization failed: {error}"
        _agent_log(hypothesis_id="G", message="callback error param", data={"error": error}, run_id="pre")
        return redirect(url_for("index"))

    remote_state = request.args.get("state")
    local_state = session.get(STATE_KEY)
    if local_state and remote_state and local_state != remote_state:
        session["status_message"] = "Spotify authorization failed: state mismatch."
        _agent_log(
            hypothesis_id="G",
            message="callback state mismatch",
            data={"local_state_prefix": str(local_state)[:6], "remote_state_prefix": str(remote_state)[:6]},
            run_id="pre",
        )
        return redirect(url_for("index"))

    code = request.args.get("code")
    if not code:
        session["status_message"] = "Spotify authorization failed: callback code was missing."
        _agent_log(hypothesis_id="G", message="callback missing code", data={}, run_id="pre")
        return redirect(url_for("index"))

    oauth = _build_oauth(settings)
    try:
        _t_tok = time.time()
        token_info = oauth.get_access_token(code=code, check_cache=False)
    except Exception as exc:  # noqa: BLE001
        session["status_message"] = f"Spotify token exchange failed: {exc}"
        _agent_log(hypothesis_id="G", message="token exchange failed", data={"exc": str(exc)}, run_id="pre")
        return redirect(url_for("index"))

    try:
        oauth.cache_handler.save_token_to_cache(token_info)
    except Exception as exc:  # noqa: BLE001
        session["status_message"] = f"Spotify token storage failed: {exc}"
        _agent_log(hypothesis_id="G", message="token storage failed", data={"exc": str(exc)}, run_id="pre")
        return redirect(url_for("index"))
    session["status_message"] = "Spotify connected successfully. Starting ingestion and feature prep."
    _agent_log(hypothesis_id="G", message="callback stored token in session", data={}, run_id="pre")
    return redirect(url_for("start_process"))


@app.get("/start")
def start_process() -> Any:
    _agent_log(
        hypothesis_id="B",
        message="GET /start entry",
        data={
            "has_session_token": TOKEN_INFO_KEY in session,
            "has_session_state": STATE_KEY in session,
            "scheme": request.scheme,
        },
    )
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        session["status_message"] = "Connect your Spotify account to start the process."
        return redirect(url_for("index"))

    return _starting_pipeline_html(profile, auto_submit=True)


@app.get("/logout")
def logout() -> Any:
    try:
        if SPOTIPY_CACHE_PATH.exists():
            SPOTIPY_CACHE_PATH.unlink()
    except Exception:
        pass
    session.pop(PROFILE_KEY, None)
    session.pop(STATE_KEY, None)
    session["status_message"] = "Disconnected from Spotify."
    return redirect(url_for("index"))


@app.post("/run-pipeline")
def run_pipeline() -> Any:
    print("DEBUG: POST /run-pipeline entered")
    _agent_log(
        hypothesis_id="C",
        message="POST /run-pipeline entry",
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
        _agent_log(hypothesis_id="C", message="pipeline stream generator start", data={}, run_id="pre")
        # Yield the starting HTML (without auto-submit) and pad to force browser flush
        yield _starting_pipeline_html(profile, auto_submit=False) + (" " * 4096) + "\n"

        q = queue.Queue()
        
        def progress_callback(percent: float, message: str):
            # Push script tags to the queue instead of yielding
            q.put(f"<script>updateProgress({percent * 100}, {json.dumps(message)});</script>\n")

        def worker():
            try:
                with app.app_context():
                    _agent_log(hypothesis_id="C", message="pipeline worker start", data={}, run_id="pre")
                    try:
                        # Run the heavy lifting in this background thread
                        res = _analyze_library(settings, client, load_to_db=True, progress_callback=progress_callback)
                        _agent_log(
                            hypothesis_id="C",
                            message="pipeline worker finished analyze_library",
                            data={
                                "records_loaded": int(res.records_loaded),
                                "raw_empty": bool(res.raw_preview.empty),
                                "has_cluster_result": res.cluster_result is not None,
                                "has_warning": bool(res.audio_features_warning),
                            },
                            run_id="pre",
                        )
                        q.put(("data", res))
                    except spotipy.SpotifyException as exc:
                        _agent_log(
                            hypothesis_id="C",
                            message="pipeline worker SpotifyException",
                            data={"http_status": getattr(exc, "http_status", None)},
                            run_id="pre",
                        )
                        q.put(("error", f"Spotify API request failed ({getattr(exc, 'http_status', 'unknown')})."))
                    except Exception as exc:
                        # Use repr to avoid potential __str__ issues in background threads
                        _agent_log(hypothesis_id="C", message="pipeline worker Exception", data={"exc": repr(exc)}, run_id="pre")
                        q.put(("error", f"Unexpected pipeline error: {repr(exc)}"))
            finally:
                # Signal that the worker is finished (success or failure)
                q.put(None)

        # Start the worker thread
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        analysis = None
        error_msg = None

        # Yield progress updates as they arrive from the queue
        while True:
            try:
                # Use a timeout so we can periodically check if the worker is still alive
                item = q.get(timeout=1.0)
                if item is None:
                    # Sentinel received, worker is done
                    break
                
                if isinstance(item, tuple):
                    if item[0] == "data":
                        analysis = item[1]
                    else:
                        error_msg = item[1]
                    # We break the loop after receiving the final data or error
                    break
                else:
                    # Add padding and newline to force browser flush
                    yield item + (" " * 1024) + "\n"
            except queue.Empty:
                if not thread.is_alive():
                    # If the queue is empty and the thread is dead, something went wrong
                    error_msg = "Pipeline execution failed (background thread exited unexpectedly)."
                    break
                continue

        if error_msg:
            safe_html = json.dumps(_dashboard_html(profile, error=error_msg)).replace("<", "\\u003c")
            yield f"<script>document.open(); document.write({safe_html}); document.close();</script>"
            return

        if analysis.raw_preview.empty:
            safe_html = json.dumps(_dashboard_html(profile, warning='No saved tracks were found.')).replace("<", "\\u003c")
            yield f"<script>document.open(); document.write({safe_html}); document.close();</script>"
            return

        info_message = "Pipeline completed successfully."
        if analysis.audio_features_warning:
            info_message = "Pipeline completed with partial audio-feature coverage."

        final_dashboard = _dashboard_html(
            profile,
            info=info_message,
            warning=analysis.audio_features_warning,
            loaded_rows=analysis.records_loaded,
            dataset_name=analysis.dataset_name,
            raw_preview=analysis.raw_preview,
            scaled_preview=analysis.scaled_preview,
            cluster_result=analysis.cluster_result,
        )
        
        # Store the result in the global cache for the redirect
        # We use a static key for this demo; in production use a session-specific ID
        user_id = profile.get("id", "default_user")
        ANALYSIS_CACHE[user_id] = analysis
        
        # Redirect the browser to the clean dashboard route
        yield "<script>window.location.href = '/dashboard';</script>\n"
        print(f"DEBUG: Redirecting user {user_id} to /dashboard")

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
    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        return redirect(url_for("index"))
    
    user_id = profile.get("id", "default_user")
    analysis = ANALYSIS_CACHE.get(user_id)
    
    if not analysis:
        # If no analysis in cache, go back to index with a message
        session["status_message"] = "No analysis found. Please run the pipeline first."
        return redirect(url_for("index"))
        
    return _dashboard_html(
        profile,
        info="Analysis loaded from cache.",
        warning=analysis.audio_features_warning,
        loaded_rows=analysis.records_loaded,
        dataset_name=analysis.dataset_name,
        raw_preview=analysis.raw_preview,
        scaled_preview=analysis.scaled_preview,
        cluster_result=analysis.cluster_result,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/test-dashboard")
def test_dashboard():
    import numpy as np
    from genresense.recommendations import ClusterSummary, ClusterDiagnostics, GenreClusterResult
    
    # Mock profile
    profile = {"id": "test_user", "display_name": "Test User (Mock)"}
    
    # 50 tracks
    n = 50
    data = {
        "track_id": [f"track_{i}" for i in range(n)],
        "track_name": [f"Mock Track {i}" for i in range(n)],
        "artist_name": [f"Mock Artist {i%5}" for i in range(n)],
        "valence": np.random.rand(n),
        "energy": np.random.rand(n),
        "tempo": np.random.uniform(60, 180, n),
        "cluster_id": [i % 2 for i in range(n)],
        "centroid_distance": np.random.rand(n),
        "track_url": [f"https://open.spotify.com/track/mock_{i}" for i in range(n)]
    }
    df = pd.DataFrame(data)
    df["cluster_label"] = df["cluster_id"].map({0: "Deep Emerald Beats", 1: "Lavender Chill"})
    
    summaries = [
        ClusterSummary(
            cluster_id=0,
            label="Deep Emerald Beats",
            track_count=25,
            share=0.5,
            representative_track="Mock Track 0",
            avg_energy=0.7,
            avg_tempo=120.0,
            avg_valence=0.4
        ),
        ClusterSummary(
            cluster_id=1,
            label="Lavender Chill",
            track_count=25,
            share=0.5,
            representative_track="Mock Track 1",
            avg_energy=0.3,
            avg_tempo=90.0,
            avg_valence=0.6
        )
    ]
    
    diagnostics = ClusterDiagnostics(
        selected_clusters=2,
        silhouette_score=0.45,
        inertia_by_cluster={2: 12.3},
        silhouette_by_cluster={2: 0.45}
    )
    
    cluster_result = GenreClusterResult(
        clustered_frame=df,
        summaries=summaries,
        diagnostics=diagnostics,
        scaled_feature_columns=["valence", "energy"]
    )
    
    # Renders the dashboard using the mock data
    return _dashboard_html(
        profile,
        info="Mock data loaded for UI testing.",
        loaded_rows=n,
        dataset_name="mock_dataset",
        raw_preview=df,
        scaled_preview=df,
        cluster_result=cluster_result,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True, use_reloader=False)

