"""sklearn preprocessing pipeline for the feature matrix.

KNN-impute numeric columns → RobustScaler; most-frequent impute categorical →
OneHotEncoder. Pipeline is fit on train and applied to test; column names are
preserved for downstream introspection.

Ported from `FantasyApiModel.normalize_dataset`, restructured so the transformer
is a reusable object that can be serialized with joblib.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_preprocessor(
    knn_neighbors: int = 5,
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for numeric + categorical features."""
    numeric_pipe = make_pipeline(
        KNNImputer(n_neighbors=knn_neighbors, weights="uniform"),
        RobustScaler(),
    )
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_include=object)),
        ]
    )


def fit_transform(
    df: pd.DataFrame,
    drop_columns: Iterable[str] | None = None,
    target_field: str | None = None,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Fit a preprocessor on `df` and return (normalized_frame, fitted_preprocessor).

    The target column (if present and named) is held out from the preprocessing
    step and re-joined at the end so downstream code can split X / y cleanly.
    """
    df = df.copy()
    target_series: pd.Series | None = None
    drop = list(drop_columns or [])

    if target_field is not None and target_field in df.columns:
        target_series = df[target_field].copy()
        drop = list({*drop, target_field})

    present_drop = [c for c in drop if c in df.columns]
    if present_drop:
        df = df.drop(columns=present_drop)

    preprocessor = preprocessor or build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    if scipy.sparse.issparse(transformed):
        transformed = transformed.toarray()

    feature_names = [
        col.split("__", 1)[1] if "__" in col else col
        for col in preprocessor.get_feature_names_out()
    ]
    normalized = pd.DataFrame(transformed, columns=feature_names, index=df.index)

    if target_series is not None:
        normalized = normalized.join(target_series)
    return normalized, preprocessor


def transform(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    drop_columns: Iterable[str] | None = None,
    target_field: str | None = None,
) -> pd.DataFrame:
    """Apply an already-fitted preprocessor to `df` (for test/inference)."""
    df = df.copy()
    target_series: pd.Series | None = None
    drop = list(drop_columns or [])

    if target_field is not None and target_field in df.columns:
        target_series = df[target_field].copy()
        drop = list({*drop, target_field})

    present_drop = [c for c in drop if c in df.columns]
    if present_drop:
        df = df.drop(columns=present_drop)

    transformed = preprocessor.transform(df)
    if scipy.sparse.issparse(transformed):
        transformed = transformed.toarray()

    feature_names = [
        col.split("__", 1)[1] if "__" in col else col
        for col in preprocessor.get_feature_names_out()
    ]
    normalized = pd.DataFrame(transformed, columns=feature_names, index=df.index)
    if target_series is not None:
        normalized = normalized.join(target_series)
    return normalized
