from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from genresense.schema import AUDIO_FEATURE_COLUMNS


class FeatureEngineeringError(ValueError):
    """Raised when audio features are unavailable for scaling."""


@dataclass
class FeatureSet:
    input_frame: pd.DataFrame
    normalized_frame: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler


class FeatureNormalizer:
    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = feature_columns or AUDIO_FEATURE_COLUMNS

    def fit_transform(self, dataframe: pd.DataFrame) -> FeatureSet:
        if dataframe.empty:
            raise FeatureEngineeringError("No saved tracks were returned from Spotify.")

        working_frame = dataframe.copy()
        available = working_frame.dropna(subset=self.feature_columns).reset_index(drop=True)

        if available.empty:
            raise FeatureEngineeringError("Audio feature values are missing for all saved tracks.")

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(available[self.feature_columns])

        normalized_frame = available.copy()
        for index, column in enumerate(self.feature_columns):
            normalized_frame[f"scaled_{column}"] = scaled_values[:, index]

        return FeatureSet(
            input_frame=available,
            normalized_frame=normalized_frame,
            feature_columns=self.feature_columns,
            scaler=scaler,
        )
