from __future__ import annotations

from dataclasses import dataclass
import html
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import spotipy
from dotenv import load_dotenv
from flask import Flask, redirect, request, session, url_for
from spotipy.cache_handler import CacheHandler
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

REQUIRED_ENV_VARS = [
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "SPOTIFY_REDIRECT_URI",
    "POSTGRES_DSN",
    "RAPIDAPI_KEY",
    "FLASK_SECRET_KEY",
]


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "true").lower() == "true"


class FlaskSessionCacheHandler(CacheHandler):
    def get_cached_token(self) -> Any:
        return session.get(TOKEN_INFO_KEY)

    def save_token_to_cache(self, token_info: Any) -> None:
        session[TOKEN_INFO_KEY] = token_info

    def delete_cached_token(self) -> None:
        session.pop(TOKEN_INFO_KEY, None)


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
        cache_handler=FlaskSessionCacheHandler(),
    )


def _get_authenticated_client(settings: AppSettings) -> tuple[spotipy.Spotify | None, dict[str, Any] | None]:
    oauth = _build_oauth(settings)
    token_info = oauth.validate_token(session.get(TOKEN_INFO_KEY))
    if not token_info:
        return None, None
    session[TOKEN_INFO_KEY] = token_info
    client = spotipy.Spotify(auth_manager=oauth)
    profile = client.current_user()
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
        border = "#16a34a" if is_selected else "#dbe4ee"
        background = "#f0fdf4" if is_selected else "#ffffff"
        width = max(summary.share * 100.0, 6.0)
        cards.append(
            f"""
            <div style="border:1px solid {border};border-radius:14px;padding:16px;background:{background};">
              <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
                <div>
                  <div style="font-size:12px;letter-spacing:0.08em;text-transform:uppercase;color:#64748b;">{html.escape(summary.label)}</div>
                  <div style="font-size:22px;font-weight:800;margin-top:4px;">{summary.track_count} tracks</div>
                </div>
                <div style="font-size:12px;color:#475569;">{summary.share * 100:.1f}% of library</div>
              </div>
              <div style="margin-top:12px;background:#e2e8f0;border-radius:999px;height:12px;overflow:hidden;">
                <div style="width:{width:.2f}%;height:100%;background:linear-gradient(90deg,#16a34a,#22c55e);"></div>
              </div>
              <div style="margin-top:12px;color:#334155;font-size:13px;line-height:1.6;">
                <strong>Anchor track:</strong> {html.escape(summary.representative_track)}<br>
                <strong>Energy:</strong> {summary.avg_energy:.2f} &nbsp; <strong>Tempo:</strong> {summary.avg_tempo:.1f} BPM &nbsp; <strong>Valence:</strong> {summary.avg_valence:.2f}
              </div>
            </div>
            """
        )
        options.append(
            f'<option value="{summary.cluster_id}"{" selected" if is_selected else ""}>{html.escape(summary.label)} ({summary.track_count} tracks)</option>'
        )

    return f"""
    <section style="margin-top:24px;display:grid;gap:16px;">
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="border:1px solid #dbe4ee;border-radius:12px;padding:12px 14px;background:#fff;">
          <div style="font-size:12px;color:#64748b;">Mathematical Genres</div>
          <div style="font-size:22px;font-weight:800;">{cluster_result.diagnostics.selected_clusters}</div>
        </div>
        <div style="border:1px solid #dbe4ee;border-radius:12px;padding:12px 14px;background:#fff;">
          <div style="font-size:12px;color:#64748b;">Silhouette Score</div>
          <div style="font-size:22px;font-weight:800;">{score_label}</div>
        </div>
      </div>
      <div>
        <h2 style="margin:0 0 6px;">Mathematical Genres</h2>
        <p style="margin:0;color:#475569;line-height:1.6;">K-Means grouped your songs by how they sound, not by Spotify metadata labels. Higher silhouette scores mean the discovered genres are more distinct.</p>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;">
        {''.join(cards)}
      </div>
      <div style="border:1px solid #dbe4ee;border-radius:16px;padding:18px;background:#fff;">
        <h3 style="margin:0 0 8px;">Generate Playlist</h3>
        <p style="margin:0 0 14px;color:#475569;line-height:1.6;">Pick one mathematical genre for a pure lane, or let the SentiLink hybrid mode walk into nearby clusters using cosine similarity across tempo and energy.</p>
        <form method="post" action="/generate-playlist" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;align-items:end;">
          <label style="display:grid;gap:6px;font-size:13px;color:#475569;">
            Mathematical Genre
            <select name="cluster_id" style="padding:10px 12px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;">
              {''.join(options)}
            </select>
          </label>
          <label style="display:grid;gap:6px;font-size:13px;color:#475569;">
            Playlist Type
            <select name="playlist_mode" style="padding:10px 12px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;">
              <option value="pure"{" selected" if selected_playlist_mode == "pure" else ""}>Pure Playlist</option>
              <option value="hybrid"{" selected" if selected_playlist_mode == "hybrid" else ""}>SentiLink Hybrid</option>
              <option value="discovery"{" selected" if selected_playlist_mode == "discovery" else ""}>True Discovery</option>
            </select>
          </label>
          <label style="display:grid;gap:6px;font-size:13px;color:#475569;">
            Track Count
            <input name="playlist_size" type="number" min="5" max="50" value="{selected_playlist_size}" style="padding:10px 12px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;">
          </label>
          <label style="display:grid;gap:6px;font-size:13px;color:#475569;">
            Visibility
            <select name="playlist_visibility" style="padding:10px 12px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;">
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


def _analyze_library(settings: AppSettings, client: spotipy.Spotify, *, load_to_db: bool) -> LibraryAnalysis:
    extractor = SpotifySavedTracksExtractor(client, settings.rapidapi)
    extraction = extractor.extract()
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
        pipeline = SpotifyLibraryPipeline(settings.postgres)
        load_result = pipeline.load_saved_tracks(extraction.records)

    warning_message = extraction.audio_features_warning
    cluster_result: GenreClusterResult | None = None
    scaled_preview: pd.DataFrame | None = None

    try:
        normalizer = FeatureNormalizer()
        feature_set = normalizer.fit_transform(extraction.dataframe)
        cluster_result = MathematicalGenreFinder().fit(feature_set)
        scaled_preview = _build_scaled_preview(cluster_result)
    except (FeatureEngineeringError, RecommendationError) as exc:
        warning_message = f"{warning_message} {exc}".strip() if warning_message else str(exc)

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
        alerts.append(f'<div style="background:#ecfeff;border:1px solid #a5f3fc;padding:12px;border-radius:8px;">{html.escape(info)}</div>')
    if warning:
        alerts.append(
            f'<div style="background:#fffbeb;border:1px solid #fde68a;padding:12px;border-radius:8px;">{html.escape(warning)}</div>'
        )
    if error:
        alerts.append(
            f'<div style="background:#fef2f2;border:1px solid #fecaca;padding:12px;border-radius:8px;">{html.escape(error)}</div>'
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
    if loaded_rows is not None and dataset_name is not None:
        metric_html = f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0 18px;">
          <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;background:#fff;">
            <div style="font-size:12px;color:#6b7280;">Saved Tracks Loaded</div>
            <div style="font-size:18px;font-weight:700;">{loaded_rows}</div>
          </div>
          <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;background:#fff;">
            <div style="font-size:12px;color:#6b7280;">Dataset</div>
            <div style="font-size:18px;font-weight:700;">{html.escape(dataset_name)}</div>
          </div>
          <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;background:#fff;">
            <div style="font-size:12px;color:#6b7280;">Tracks With Features</div>
            <div style="font-size:18px;font-weight:700;">{len(scaled_preview) if scaled_preview is not None else 0}</div>
          </div>
        </div>
        """

    viz_html = ""
    if cluster_result is not None and not cluster_result.clustered_frame.empty:
        import json
        viz_data = cluster_result.clustered_frame
        chart_data = viz_data[["track_name", "artist_name", "cluster_id", "valence", "energy"]].to_dict(orient="records")
        chart_json = json.dumps(chart_data)
        
        viz_html = f"""
        <div class="card" style="background:#0f172a; border-color:#334155; padding:20px; overflow:hidden; margin-top:20px; border-radius:12px;">
          <h2 style="margin:0 0 16px; color:#f8fafc; font-size:16px; border-bottom:1px solid #334155; padding-bottom:10px;">
            Acoustic Landscape (KNN Visualizer)
          </h2>
          <div style="position:relative; width:100%; height:400px; background:radial-gradient(circle at center, #1e293b 0%, #0f172a 100%); border-radius:8px; border:1px solid #334155; overflow:hidden;" id="knn-container">
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
      <style>
        body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
        .wrap {{ max-width: 1100px; margin: 0 auto; padding: 26px 18px 42px; }}
        .top {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }}
        .btn {{ display:inline-block; padding:10px 14px; border-radius:8px; text-decoration:none; font-weight:600; }}
        .btn-primary {{ background:#16a34a; color:#fff; border:0; cursor:pointer; }}
        .btn-outline {{ border:1px solid #cbd5e1; color:#0f172a; background:#fff; }}
        table {{ width:100%; border-collapse: collapse; background:#fff; border:1px solid #e5e7eb; border-radius: 8px; overflow:hidden; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #eef2f7; font-size: 13px; text-align: left; }}
        th {{ background: #f8fafc; font-weight: 600; }}
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
            background: #1e293b;
            border: 1px solid #334155;
            color: #fff;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 11px;
            white-space: nowrap;
            transition: opacity 0.2s ease, bottom 0.2s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
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
        <div class="top">
          <div>
            <h1 style="margin:0;">GenreSense</h1>
            <p style="margin:6px 0 0;color:#475569;">Authenticated as {html.escape(profile.get("display_name") or profile.get("id") or "Spotify user")}.</p>
          </div>
          <div style="display:flex;gap:10px;">
            <a class="btn btn-outline" href="/logout">Disconnect</a>
            <form method="post" action="/run-pipeline" style="margin:0;">
              <button class="btn btn-primary" type="submit">Refresh Mathematical Genres</button>
            </form>
          </div>
        </div>
        <div style="margin-top:14px;display:grid;gap:10px;">
          {''.join(alerts)}
        </div>
        {metric_html}
        {viz_html}
        {genre_html}
        {raw_table}
        {scaled_table}
      </div>
    </body>
    </html>
    """


def _starting_pipeline_html(profile: dict[str, Any]) -> str:
    display_name = profile.get("display_name") or profile.get("id") or "Spotify user"
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Starting GenreSense</title>
      <style>
        body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: linear-gradient(180deg, #f8fafc 0%, #ecfdf5 100%); color: #0f172a; }}
        .wrap {{ min-height: 100vh; display: grid; place-items: center; padding: 24px; }}
        .card {{ width: min(560px, 100%); background: rgba(255, 255, 255, 0.96); border: 1px solid #d1fae5; border-radius: 20px; padding: 32px; box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08); }}
        .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px; background: #dcfce7; color: #166534; font-size: 13px; font-weight: 700; letter-spacing: 0.01em; }}
        .spinner {{ width: 18px; height: 18px; border-radius: 999px; border: 2px solid #86efac; border-top-color: #16a34a; animation: spin 0.85s linear infinite; }}
        .btn {{ display:inline-block; padding:12px 18px; border-radius:10px; text-decoration:none; font-weight:700; border:0; cursor:pointer; }}
        .btn-primary {{ background:#16a34a; color:#fff; }}
        .btn-outline {{ background:#fff; border:1px solid #cbd5e1; color:#0f172a; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="card">
          <div class="pill">
            <span class="spinner" aria-hidden="true"></span>
            Spotify connected
          </div>
          <h1 style="margin:18px 0 10px;font-size:34px;line-height:1.1;">Starting your GenreSense process...</h1>
          <p style="margin:0 0 10px;color:#475569;line-height:1.6;">
            Signed in as {html.escape(display_name)}. We're moving you straight into saved-track ingestion and feature prep now.
          </p>
          <p style="margin:0 0 24px;color:#64748b;font-size:14px;line-height:1.6;">
            If the process does not begin automatically, use the button below.
          </p>
          <form id="start-process-form" method="post" action="/run-pipeline" style="display:flex;gap:12px;flex-wrap:wrap;">
            <button class="btn btn-primary" type="submit">Start Process</button>
            <a class="btn btn-outline" href="/">Back to Dashboard</a>
          </form>
        </div>
      </div>
      <script>
        window.addEventListener("load", () => {{
          const form = document.getElementById("start-process-form");
          if (form) {{
            form.requestSubmit();
          }}
        }});
      </script>
    </body>
    </html>
    """


@app.get("/")
def index() -> Any:
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
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    session[STATE_KEY] = secrets.token_urlsafe(16)
    oauth = _build_oauth(settings)
    return redirect(oauth.get_authorize_url())


@app.get("/callback")
def callback() -> Any:
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    error = request.args.get("error")
    if error:
        session["status_message"] = f"Spotify authorization failed: {error}"
        return redirect(url_for("index"))

    remote_state = request.args.get("state")
    local_state = session.get(STATE_KEY)
    if local_state and remote_state and local_state != remote_state:
        session["status_message"] = "Spotify authorization failed: state mismatch."
        return redirect(url_for("index"))

    code = request.args.get("code")
    if not code:
        session["status_message"] = "Spotify authorization failed: callback code was missing."
        return redirect(url_for("index"))

    oauth = _build_oauth(settings)
    try:
        token_info = oauth.get_access_token(code=code, check_cache=False)
    except Exception as exc:  # noqa: BLE001
        session["status_message"] = f"Spotify token exchange failed: {exc}"
        return redirect(url_for("index"))

    session[TOKEN_INFO_KEY] = token_info
    session["status_message"] = "Spotify connected successfully. Starting ingestion and feature prep."
    return redirect(url_for("start_process"))


@app.get("/start")
def start_process() -> Any:
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        session["status_message"] = "Connect your Spotify account to start the process."
        return redirect(url_for("index"))

    return _starting_pipeline_html(profile)


@app.get("/logout")
def logout() -> Any:
    session.pop(TOKEN_INFO_KEY, None)
    session.pop(STATE_KEY, None)
    session["status_message"] = "Disconnected from Spotify."
    return redirect(url_for("index"))


@app.post("/run-pipeline")
def run_pipeline() -> Any:
    try:
        settings = _settings_from_env()
    except ConfigurationError as exc:
        return _config_error_page(str(exc))

    client, profile = _get_authenticated_client(settings)
    if not client or not profile:
        session["status_message"] = "Your Spotify session expired. Please connect again."
        return redirect(url_for("index"))

    try:
        analysis = _analyze_library(settings, client, load_to_db=True)
    except spotipy.SpotifyException as exc:
        return _dashboard_html(profile, error=f"Spotify API request failed ({exc.http_status}).")
    except Exception as exc:  # noqa: BLE001
        return _dashboard_html(profile, error=f"Unexpected pipeline error: {exc}")

    if analysis.raw_preview.empty:
        return _dashboard_html(profile, warning="No saved tracks were found for this Spotify account.")

    info_message = "Pipeline completed successfully."
    if analysis.audio_features_warning:
        info_message = "Pipeline completed with partial audio-feature coverage."

    return _dashboard_html(
        profile,
        info=info_message,
        warning=analysis.audio_features_warning,
        loaded_rows=analysis.records_loaded,
        dataset_name=analysis.dataset_name,
        raw_preview=analysis.raw_preview,
        scaled_preview=analysis.scaled_preview,
        cluster_result=analysis.cluster_result,
    )


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True)

