from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dlt
from dlt.destinations import postgres

from genresense.config import PostgresSettings


@dataclass
class LoadResult:
    loaded_rows: int
    dataset_name: str
    load_info: Any


class SpotifyLibraryPipeline:
    def __init__(self, settings: PostgresSettings) -> None:
        self.settings = settings

    def load_saved_tracks(self, records: list[dict[str, Any]]) -> LoadResult:
        pipeline = dlt.pipeline(
            pipeline_name="genresense_saved_tracks",
            destination=postgres(credentials=self.settings.dsn),
            dataset_name=self.settings.dataset_name,
        )

        load_info = pipeline.run(saved_tracks_resource(records))
        return LoadResult(
            loaded_rows=len(records),
            dataset_name=self.settings.dataset_name,
            load_info=load_info,
        )


@dlt.resource(
    name="saved_tracks",
    primary_key="track_id",
    write_disposition={"disposition": "merge", "strategy": "upsert"},
)
def saved_tracks_resource(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records
