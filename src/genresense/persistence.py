from __future__ import annotations
import json
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app import LibraryAnalysis

logger = logging.getLogger(__name__)

class AnalysisPersistence:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._ensure_table()

    def _ensure_table(self):
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_analysis (
                        user_id TEXT PRIMARY KEY,
                        analysis_data JSONB,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            conn.commit()

    def save(self, user_id: str, analysis: 'LibraryAnalysis'):
        # We need to serialize the LibraryAnalysis dataclass
        # Many fields are DataFrames or complex types
        data = {
            "records_loaded": analysis.records_loaded,
            "dataset_name": analysis.dataset_name,
            "audio_features_warning": analysis.audio_features_warning,
            "raw_preview_json": analysis.raw_preview.to_json(orient="records") if analysis.raw_preview is not None else None,
            "scaled_preview_json": analysis.scaled_preview.to_json(orient="records") if analysis.scaled_preview is not None else None,
            "cluster_result": self._serialize_cluster_result(analysis.cluster_result) if analysis.cluster_result else None
        }
        
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_analysis (user_id, analysis_data, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO UPDATE
                    SET analysis_data = EXCLUDED.analysis_data, updated_at = CURRENT_TIMESTAMP;
                """, (user_id, json.dumps(data)))
            conn.commit()

    def load(self, user_id: str) -> 'LibraryAnalysis | None':
        from app import LibraryAnalysis
        from genresense.recommendations import GenreClusterResult, ClusterSummary, ClusterDiagnostics

        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT analysis_data FROM user_analysis WHERE user_id = %s", (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                
                data = row['analysis_data']
                
                # Reconstruct DataFrames
                raw_preview = pd.read_json(data['raw_preview_json'], orient="records") if data.get('raw_preview_json') else pd.DataFrame()
                scaled_preview = pd.read_json(data['scaled_preview_json'], orient="records") if data.get('scaled_preview_json') else None
                
                # Reconstruct ClusterResult
                cluster_result = None
                cr_data = data.get('cluster_result')
                if cr_data:
                    clustered_frame = pd.read_json(cr_data['clustered_frame_json'], orient="records")
                    summaries = [ClusterSummary(**s) for s in cr_data['summaries']]
                    diag = ClusterDiagnostics(**cr_data['diagnostics'])
                    cluster_result = GenreClusterResult(
                        clustered_frame=clustered_frame,
                        summaries=summaries,
                        diagnostics=diag,
                        scaled_feature_columns=cr_data['scaled_feature_columns']
                    )

                return LibraryAnalysis(
                    records_loaded=data['records_loaded'],
                    dataset_name=data['dataset_name'],
                    raw_preview=raw_preview,
                    scaled_preview=scaled_preview,
                    cluster_result=cluster_result,
                    audio_features_warning=data['audio_features_warning']
                )

    def _serialize_cluster_result(self, result):
        return {
            "clustered_frame_json": result.clustered_frame.to_json(orient="records"),
            "summaries": [vars(s) for s in result.summaries],
            "diagnostics": vars(result.diagnostics),
            "scaled_feature_columns": result.scaled_feature_columns
        }
