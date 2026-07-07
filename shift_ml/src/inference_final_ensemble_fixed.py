#!/usr/bin/env python
"""
Inference for the final 12.66 car-price ensemble.

The script loads prepared test features and all saved model artifacts,
computes component predictions, applies the weighted ensemble and the
conditional disagreement calibration, then writes submission.csv and ZIP.

Run from the project root:
    python src/inference_final_ensemble.py

Or specify another root:
    python src/inference_final_ensemble.py --project-root "C:\\temp\\shift_ml"
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a competition submission from the saved final "
            "12.66 ensemble."
        )
    )

    default_root = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help=(
            "Path to the project root. By default the parent directory "
            "of src/ is used."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Path for submission CSV. Default: "
            "<project-root>/submission/"
            "submission_final_ensemble_reproduced.csv"
        ),
    )

    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Create CSV only and do not create a ZIP archive.",
    )

    return parser.parse_args()


def require_files(paths: dict[str, Path]) -> None:
    missing = {
        label: path
        for label, path in paths.items()
        if not path.exists()
    }

    if missing:
        formatted = "\n".join(
            f"  - {label}: {path}"
            for label, path in missing.items()
        )

        raise FileNotFoundError(
            "Missing required inference artifacts:\n"
            f"{formatted}\n\n"
            "Run 01_data_eda_features.ipynb and "
            "03_final_model_training.ipynb first. "
            "For exact conditional calibration, Notebook 3 must also "
            "save the V6 model and V6 schema."
        )


def load_json(path: Path) -> dict:
    with open(path, mode="r", encoding="utf-8") as file:
        return json.load(file)


def align_canonical_by_id(
    canonical_frame: pd.DataFrame,
    ids: pd.Series,
    id_column: str,
) -> pd.DataFrame:
    result = canonical_frame.copy()
    result[id_column] = result[id_column].astype(str)

    if result[id_column].duplicated().any():
        duplicated = result.loc[
            result[id_column].duplicated(),
            id_column,
        ].head(5).tolist()

        raise ValueError(
            "Canonical test table contains duplicate IDs. "
            f"Examples: {duplicated}"
        )

    result = result.set_index(id_column, drop=False)
    wanted_ids = ids.astype(str)

    missing_ids = set(wanted_ids) - set(result.index)

    if missing_ids:
        raise KeyError(
            "Some IDs from test_model_input are absent from "
            "X_test_canonical. Examples: "
            f"{list(missing_ids)[:5]}"
        )

    return result.loc[wanted_ids].reset_index(drop=True)


def prepare_catboost_frame(
    frame: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in categorical_columns:
        if column not in result.columns:
            raise KeyError(
                f"Categorical feature is absent: {column}"
            )

        result[column] = (
            result[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return result


def make_group_key(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    key = None

    for column in columns:
        if column not in frame.columns:
            raise KeyError(
                "Target-statistics feature is absent: "
                f"{column}"
            )

        series = frame[column]

        if pd.api.types.is_numeric_dtype(series):
            part = (
                pd.to_numeric(
                    series,
                    errors="coerce",
                )
                .round(4)
                .astype("Float64")
                .astype("string")
            )
        else:
            part = series.astype("string")

        part = (
            part.fillna("__MISSING__")
            .str.strip()
            .str.upper()
        )

        key = (
            part
            if key is None
            else key.str.cat(part, sep="|||")
        )

    return key


def build_target_stats_from_reference(
    reference: pd.DataFrame,
    apply_frame: pd.DataFrame,
    group_specs: dict[str, list[str]],
    smoothing: float,
) -> pd.DataFrame:
    if "target_log_price" not in reference.columns:
        raise KeyError(
            "target_stats_reference.parquet must contain "
            "'target_log_price'."
        )

    target_log = pd.to_numeric(
        reference["target_log_price"],
        errors="raise",
    ).astype(float)

    global_mean = float(target_log.mean())
    output = pd.DataFrame(index=apply_frame.index)

    for group_name, group_columns in group_specs.items():
        reference_key = make_group_key(
            reference,
            group_columns,
        )

        apply_key = make_group_key(
            apply_frame,
            group_columns,
        )

        grouped = (
            pd.DataFrame(
                {
                    "key": reference_key.to_numpy(),
                    "target_log": target_log.to_numpy(),
                }
            )
            .groupby("key", sort=False)
            .agg(
                count=("target_log", "size"),
                target_sum=("target_log", "sum"),
            )
        )

        grouped["smooth_mean"] = (
            grouped["target_sum"]
            + smoothing * global_mean
        ) / (
            grouped["count"] + smoothing
        )

        counts = (
            apply_key
            .map(grouped["count"])
            .fillna(0)
            .to_numpy(dtype=float)
        )

        log_means = (
            apply_key
            .map(grouped["smooth_mean"])
            .fillna(global_mean)
            .to_numpy(dtype=float)
        )

        output[f"{group_name}__count"] = np.log1p(counts)
        output[f"{group_name}__log_mean"] = log_means

    return output


def make_vehicle_text(
    frame: pd.DataFrame,
    text_columns: list[str],
) -> pd.Series:
    missing = [
        column
        for column in text_columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Canonical test table lacks text columns: "
            f"{missing}"
        )

    text_parts = [
        frame[column]
        .astype("string")
        .fillna("")
        .str.strip()
        for column in text_columns
    ]

    return pd.concat(text_parts, axis=1).agg(
        " ".join,
        axis=1,
    )


def build_model_frame(
    base_features: pd.DataFrame,
    canonical_features: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """
    Reconstruct a model matrix from prepared V8 features and, where
    necessary, raw canonical columns such as full title and colour.
    """
    output = pd.DataFrame(index=base_features.index)

    missing = []

    for column in feature_columns:
        if column in base_features.columns:
            output[column] = base_features[column]
        elif column in canonical_features.columns:
            output[column] = canonical_features[column]
        else:
            missing.append(column)

    if missing:
        raise KeyError(
            "Cannot reconstruct model matrix. Missing columns: "
            f"{missing}"
        )

    output = output[feature_columns]

    return prepare_catboost_frame(
        output,
        categorical_columns,
    )


def clean_retrieval_category(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .fillna("__MISSING__")
        .str.strip()
        .str.upper()
        .astype(str)
    )


def prepare_retrieval_query(
    X: pd.DataFrame,
    retrieval_schema: dict,
) -> pd.DataFrame:
    """
    Prepare the test matrix used by K=3 retrieval.

    `Пробег_log` is a derived distance feature. It is calculated from
    the source feature `Пробег_число`, so both must be available here.
    """
    numeric_scales = retrieval_schema["numeric_scales"]
    categorical_weights = retrieval_schema["categorical_weights"]

    source_columns = list(
        dict.fromkeys(
            [
                "Бренд",
                "Модель",
                "Пробег_число",
                *[
                    column
                    for column in numeric_scales
                    if column != "Пробег_log"
                ],
                *categorical_weights.keys(),
            ]
        )
    )

    missing = [
        column
        for column in source_columns
        if column not in X.columns
    ]

    if missing:
        raise KeyError(
            "Prepared test features lack retrieval columns: "
            f"{missing}"
        )

    result = X[source_columns].copy().reset_index(drop=True)

    for column in numeric_scales:
        if column == "Пробег_log":
            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).astype(float)

    result["Пробег_log"] = np.log1p(
        pd.to_numeric(
            result["Пробег_число"],
            errors="coerce",
        )
    )

    for column in categorical_weights:
        result[column] = clean_retrieval_category(
            result[column]
        )

    result["Бренд"] = clean_retrieval_category(
        result["Бренд"]
    )

    result["Модель"] = clean_retrieval_category(
        result["Модель"]
    )

    result["brand_model_key"] = (
        result["Бренд"]
        + "|||"
        + result["Модель"]
    )

    result["brand_key"] = result["Бренд"]

    return result


def comparable_distance(
    reference_block: pd.DataFrame,
    query_block: pd.DataFrame,
    numeric_scales: dict[str, float],
    categorical_weights: dict[str, float],
    missing_numeric_penalty: float,
    missing_categorical_penalty: float,
) -> np.ndarray:
    distance = np.zeros(
        (len(reference_block), len(query_block)),
        dtype=float,
    )

    for column, scale in numeric_scales.items():
        reference_values = reference_block[
            column
        ].to_numpy(dtype=float)[:, None]

        query_values = query_block[
            column
        ].to_numpy(dtype=float)[None, :]

        available = (
            np.isfinite(reference_values)
            & np.isfinite(query_values)
        )

        normalized_gap = (
            np.abs(reference_values - query_values)
            / scale
        )

        distance += np.where(
            available,
            normalized_gap ** 2,
            missing_numeric_penalty,
        )

    for column, weight in categorical_weights.items():
        reference_values = reference_block[
            column
        ].to_numpy(dtype=object)[:, None]

        query_values = query_block[
            column
        ].to_numpy(dtype=object)[None, :]

        available = (
            (reference_values != "__MISSING__")
            & (query_values != "__MISSING__")
        )

        mismatch = (
            reference_values != query_values
        ).astype(float)

        distance += weight * np.where(
            available,
            mismatch,
            missing_categorical_penalty,
        )

    return distance


def predict_retrieval(
    reference: pd.DataFrame,
    query: pd.DataFrame,
    retrieval_schema: dict,
) -> np.ndarray:
    required_reference_columns = {
        "brand_model_key",
        "brand_key",
        "target_log_price",
        *retrieval_schema["numeric_scales"].keys(),
        *retrieval_schema["categorical_weights"].keys(),
    }

    missing = [
        column
        for column in required_reference_columns
        if column not in reference.columns
    ]

    if missing:
        raise KeyError(
            "retrieval_reference.parquet lacks columns: "
            f"{missing}"
        )

    reference = reference.reset_index(drop=True)
    query = query.reset_index(drop=True)

    k_neighbors = int(retrieval_schema["k_neighbors"])
    distance_epsilon = float(
        retrieval_schema["distance_epsilon"]
    )

    numeric_scales = retrieval_schema["numeric_scales"]
    categorical_weights = retrieval_schema[
        "categorical_weights"
    ]

    missing_numeric_penalty = float(
        retrieval_schema["missing_numeric_penalty"]
    )

    missing_categorical_penalty = float(
        retrieval_schema["missing_categorical_penalty"]
    )

    prediction_log = np.zeros(
        len(query),
        dtype=float,
    )

    brand_model_groups = reference.groupby(
        "brand_model_key",
        sort=False,
    ).indices

    brand_groups = reference.groupby(
        "brand_key",
        sort=False,
    ).indices

    query_groups = query.groupby(
        "brand_model_key",
        sort=False,
    ).indices

    for group_key, query_positions in query_groups.items():
        candidate_positions = brand_model_groups.get(
            group_key
        )

        if candidate_positions is None:
            brand_key = query.loc[
                query_positions[0],
                "brand_key",
            ]

            candidate_positions = brand_groups.get(
                brand_key
            )

        if candidate_positions is None:
            candidate_positions = np.arange(
                len(reference)
            )

        reference_block = reference.iloc[
            candidate_positions
        ]

        query_block = query.iloc[
            query_positions
        ]

        distances = comparable_distance(
            reference_block=reference_block,
            query_block=query_block,
            numeric_scales=numeric_scales,
            categorical_weights=categorical_weights,
            missing_numeric_penalty=missing_numeric_penalty,
            missing_categorical_penalty=(
                missing_categorical_penalty
            ),
        )

        k_effective = min(
            k_neighbors,
            len(reference_block),
        )

        nearest_positions = np.argpartition(
            distances,
            kth=k_effective - 1,
            axis=0,
        )[:k_effective]

        nearest_distances = np.take_along_axis(
            distances,
            nearest_positions,
            axis=0,
        )

        nearest_log_prices = reference_block[
            "target_log_price"
        ].to_numpy(dtype=float)[nearest_positions]

        similarity_weights = 1 / (
            nearest_distances + distance_epsilon
        ) ** 2

        prediction_log[query_positions] = (
            similarity_weights * nearest_log_prices
        ).sum(axis=0) / similarity_weights.sum(axis=0)

    return np.maximum(
        np.expm1(prediction_log),
        1,
    )


def load_catboost_model(path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(path)
    return model


def catboost_price_prediction(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
) -> np.ndarray:
    return np.maximum(
        np.expm1(model.predict(frame)),
        1,
    )


def get_disagreement_multipliers(
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    cut_points = np.asarray(
        config["disagreement_cut_points"],
        dtype=float,
    )

    raw_multipliers = config[
        "disagreement_multipliers"
    ]

    if {
        "low",
        "medium",
        "high",
    }.issubset(raw_multipliers):
        multipliers = np.asarray(
            [
                raw_multipliers["low"],
                raw_multipliers["medium"],
                raw_multipliers["high"],
            ],
            dtype=float,
        )
    else:
        multipliers = np.asarray(
            [
                raw_multipliers[str(index)]
                for index in range(3)
            ],
            dtype=float,
        )

    if len(cut_points) != 2 or len(multipliers) != 3:
        raise ValueError(
            "Expected two disagreement cut points and "
            "three calibration multipliers."
        )

    return cut_points, multipliers


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    processed_dir = project_root / "data" / "processed"
    model_dir = (
        project_root
        / "models"
        / "final_ensemble_12_66"
    )

    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else (
            project_root
            / "submission"
            / "submission_final_ensemble_reproduced.csv"
        )
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "prepared X_test_v8": (
            processed_dir / "X_test_v8.parquet"
        ),
        "test model input": (
            processed_dir
            / "test_model_input_v8.parquet"
        ),
        "canonical test": (
            processed_dir
            / "X_test_canonical.parquet"
        ),
        "feature schema": (
            processed_dir / "feature_schema.json"
        ),
        "ensemble configuration": (
            model_dir / "ensemble_config.json"
        ),
        "model schemas": (
            model_dir / "model_feature_schemas.json"
        ),
        "Ridge model": (
            model_dir / "ridge_log_target_final.joblib"
        ),
        "Text Ridge model": (
            model_dir / "text_tfidf_ridge_final.joblib"
        ),
        "CatBoost V5": (
            model_dir / "catboost_v5_title_final.cbm"
        ),
        "CatBoost V6": (
            model_dir / "catboost_v6_final.cbm"
        ),
        "CatBoost V8": (
            model_dir / "catboost_v8_final.cbm"
        ),
        "CatBoost stats": (
            model_dir / "catboost_stats_final.cbm"
        ),
        "CatBoost V10": (
            model_dir / "catboost_v10_final.cbm"
        ),
        "retrieval reference": (
            model_dir / "retrieval_reference.parquet"
        ),
        "target-stats reference": (
            model_dir
            / "target_stats_reference.parquet"
        ),
    }

    require_files(paths)

    feature_schema = load_json(
        paths["feature schema"]
    )

    ensemble_config = load_json(
        paths["ensemble configuration"]
    )

    model_schemas = load_json(
        paths["model schemas"]
    )

    if "v6" not in model_schemas:
        raise KeyError(
            "model_feature_schemas.json lacks the V6 schema. "
            "Update and rerun Notebook 3 after adding the V6 "
            "training/export cell."
        )

    id_column = feature_schema["id_column"]
    expected_v8_columns = feature_schema[
        "feature_columns"
    ]

    X_test_v8 = pd.read_parquet(
        paths["prepared X_test_v8"]
    )

    test_model_input = pd.read_parquet(
        paths["test model input"]
    )

    test_canonical = pd.read_parquet(
        paths["canonical test"]
    )

    if id_column not in test_model_input.columns:
        raise KeyError(
            f"'{id_column}' is absent from test_model_input."
        )

    if len(X_test_v8) != len(test_model_input):
        raise ValueError(
            "X_test_v8 and test_model_input have different "
            "row counts."
        )

    missing_v8_features = [
        column
        for column in expected_v8_columns
        if column not in X_test_v8.columns
    ]

    unexpected_v8_features = [
        column
        for column in X_test_v8.columns
        if column not in expected_v8_columns
    ]

    if missing_v8_features or unexpected_v8_features:
        raise ValueError(
            "Prepared V8 feature schema mismatch.\n"
            f"Missing: {missing_v8_features}\n"
            f"Unexpected: {unexpected_v8_features}"
        )

    X_test_v8 = X_test_v8[
        expected_v8_columns
    ].reset_index(drop=True)

    test_ids = test_model_input[
        id_column
    ].astype(str).reset_index(drop=True)

    canonical_test = align_canonical_by_id(
        canonical_frame=test_canonical,
        ids=test_ids,
        id_column=id_column,
    )

    categorical_v8 = model_schemas["v8"][
        "categorical_columns"
    ]

    X_test_cb = prepare_catboost_frame(
        X_test_v8,
        categorical_v8,
    )

    target_stats_reference = pd.read_parquet(
        paths["target-stats reference"]
    )

    target_stats_schema = model_schemas[
        "target_statistics"
    ]

    target_stats = build_target_stats_from_reference(
        reference=target_stats_reference,
        apply_frame=X_test_cb,
        group_specs=target_stats_schema["group_specs"],
        smoothing=float(
            target_stats_schema["smoothing"]
        ),
    )

    X_test_v5 = build_model_frame(
        base_features=X_test_cb,
        canonical_features=canonical_test,
        feature_columns=model_schemas["v5"][
            "feature_columns"
        ],
        categorical_columns=model_schemas["v5"][
            "categorical_columns"
        ],
    )

    X_test_v6 = build_model_frame(
        base_features=X_test_cb,
        canonical_features=canonical_test,
        feature_columns=model_schemas["v6"][
            "feature_columns"
        ],
        categorical_columns=model_schemas["v6"][
            "categorical_columns"
        ],
    )

    stats_schema = model_schemas["stats"]
    X_test_stats = pd.concat(
        [
            X_test_cb[
                stats_schema["base_feature_columns"]
            ].reset_index(drop=True),
            target_stats[
                stats_schema["target_stats_columns"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    X_test_stats = X_test_stats[
        stats_schema["feature_columns"]
    ]

    X_test_stats = prepare_catboost_frame(
        X_test_stats,
        stats_schema["categorical_columns"],
    )

    v10_schema = model_schemas["v10"]
    X_test_v10 = pd.concat(
        [
            X_test_cb[
                v10_schema["base_feature_columns"]
            ].reset_index(drop=True),
            target_stats[
                v10_schema["target_stats_columns"]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    X_test_v10 = X_test_v10[
        v10_schema["feature_columns"]
    ]

    X_test_v10 = prepare_catboost_frame(
        X_test_v10,
        v10_schema["categorical_columns"],
    )

    ridge_model = joblib.load(paths["Ridge model"])
    text_model = joblib.load(paths["Text Ridge model"])

    v5_model = load_catboost_model(
        paths["CatBoost V5"]
    )

    v6_model = load_catboost_model(
        paths["CatBoost V6"]
    )

    v8_model = load_catboost_model(
        paths["CatBoost V8"]
    )

    stats_model = load_catboost_model(
        paths["CatBoost stats"]
    )

    v10_model = load_catboost_model(
        paths["CatBoost V10"]
    )

    text_columns = model_schemas["text_ridge"][
        "text_columns"
    ]

    text_documents = make_vehicle_text(
        canonical_test,
        text_columns,
    )

    retrieval_reference = pd.read_parquet(
        paths["retrieval reference"]
    )

    retrieval_query = prepare_retrieval_query(
        X=X_test_cb,
        retrieval_schema=model_schemas["retrieval"],
    )

    component_predictions = {
        "ridge": np.maximum(
            np.expm1(
                ridge_model.predict(X_test_cb)
            ),
            1,
        ),
        "v5": catboost_price_prediction(
            v5_model,
            X_test_v5,
        ),
        "v6": catboost_price_prediction(
            v6_model,
            X_test_v6,
        ),
        "v8": catboost_price_prediction(
            v8_model,
            X_test_cb,
        ),
        "stats": catboost_price_prediction(
            stats_model,
            X_test_stats,
        ),
        "text": np.maximum(
            np.expm1(
                text_model.predict(text_documents)
            ),
            1,
        ),
        "retrieval_k3_pred": predict_retrieval(
            reference=retrieval_reference,
            query=retrieval_query,
            retrieval_schema=model_schemas[
                "retrieval"
            ],
        ),
        "v10_stats_v8_pred": catboost_price_prediction(
            v10_model,
            X_test_v10,
        ),
    }

    weights = ensemble_config["ensemble_weights"]

    required_components = set(weights)
    found_components = set(component_predictions)

    if required_components != found_components:
        raise ValueError(
            "Ensemble components mismatch.\n"
            f"Expected: {sorted(required_components)}\n"
            f"Found: {sorted(found_components)}"
        )

    prediction_matrix = np.column_stack(
        [
            component_predictions[name]
            for name in weights
        ]
    )

    if not np.isfinite(prediction_matrix).all():
        raise ValueError(
            "At least one component prediction contains "
            "NaN or infinity."
        )

    if not (prediction_matrix > 0).all():
        raise ValueError(
            "At least one component prediction is <= 0."
        )

    weight_vector = np.asarray(
        [
            weights[name]
            for name in weights
        ],
        dtype=float,
    )

    if not np.isclose(
        weight_vector.sum(),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Ensemble weights must sum to 1. "
            f"Current sum: {weight_vector.sum()}"
        )

    raw_prediction = np.maximum(
        prediction_matrix @ weight_vector,
        1,
    )

    disagreement = (
        prediction_matrix.std(axis=1)
        / np.maximum(
            prediction_matrix.mean(axis=1),
            1,
        )
    )

    cut_points, multipliers = (
        get_disagreement_multipliers(
            ensemble_config
        )
    )

    calibration_bins = np.digitize(
        disagreement,
        cut_points,
        right=True,
    )

    calibration_vector = multipliers[
        calibration_bins
    ]

    final_prediction = np.maximum(
        raw_prediction * calibration_vector,
        1,
    )

    submission = pd.DataFrame(
        {
            "Цена": final_prediction,
        }
    )

    if len(submission) != len(X_test_v8):
        raise ValueError(
            "Submission row count does not match test data."
        )

    if submission.isna().any().any():
        raise ValueError(
            "Submission contains missing predictions."
        )

    submission.to_csv(
        output_csv,
        index=False,
    )

    if not args.no_zip:
        output_zip = output_csv.with_suffix(".zip")

        with zipfile.ZipFile(
            output_zip,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.write(
                output_csv,
                arcname="submission.csv",
            )

        with zipfile.ZipFile(
            output_zip,
            mode="r",
        ) as archive:
            if archive.namelist() != ["submission.csv"]:
                raise ValueError(
                    "ZIP must contain only submission.csv."
                )
    else:
        output_zip = None

    bin_counts = pd.Series(
        calibration_bins
    ).value_counts().sort_index()

    print("\n=== INFERENCE COMPLETED ===")
    print("Project root:", project_root)
    print("Rows:", len(submission))
    print("CSV:", output_csv)

    if output_zip is not None:
        print("ZIP:", output_zip)

    print("\n=== CALIBRATION BIN DISTRIBUTION ===")
    for bin_id in range(3):
        count = int(bin_counts.get(bin_id, 0))
        share = count / len(submission) * 100

        print(
            f"Bin {bin_id}: {count} rows "
            f"({share:.3f}%), "
            f"multiplier={multipliers[bin_id]:.3f}"
        )

    print("\n=== PREDICTION SUMMARY ===")
    print(submission["Цена"].describe())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
