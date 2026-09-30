#!/usr/bin/env python3
"""Train and evaluate the first VibeCheck personality baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAITS = ["O", "C", "E", "A", "N"]
FEATURES = [f"TestA_{index}" for index in range(1, 26)]


def load_mix_a(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        member = "Study 1/Data/Study1_data.csv"
        with ZipFile(path) as archive:
            if member not in archive.namelist():
                raise ValueError(f"Archive does not contain {member}")
            with archive.open(member) as source:
                data = pd.read_csv(source)
    else:
        data = pd.read_csv(path)
    required = {"Condition", "Test A OK", *FEATURES, *TRAITS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    eligible = data.loc[
        (data["Condition"] == 3)
        & (data["Test A OK"] == 1)
        & data[TRAITS].notna().all(axis=1),
        FEATURES + TRAITS,
    ].copy()

    if eligible.empty:
        raise ValueError("No complete Mix A participants were found.")

    return eligible[FEATURES], eligible[TRAITS]


def pearson_by_trait(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = {}
    for index, trait in enumerate(TRAITS):
        result[trait] = float(np.corrcoef(actual[:, index], predicted[:, index])[0, 1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, targets = load_mix_a(args.input)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=args.seed,
    )

    estimator = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    RidgeCV(alphas=np.logspace(-3, 3, 25))
                ),
            ),
        ]
    )
    estimator.fit(x_train, y_train)
    predicted = estimator.predict(x_test)

    metrics = {
        "dataset": "Nave et al. Study 1, Mix A",
        "seed": args.seed,
        "train_users": int(len(x_train)),
        "test_users": int(len(x_test)),
        "features": FEATURES,
        "targets": TRAITS,
        "pearson_r": pearson_by_trait(y_test.to_numpy(), predicted),
        "mae": {
            trait: float(mean_absolute_error(y_test.iloc[:, index], predicted[:, index]))
            for index, trait in enumerate(TRAITS)
        },
        "r2": {
            trait: float(r2_score(y_test.iloc[:, index], predicted[:, index]))
            for index, trait in enumerate(TRAITS)
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, args.output_dir / "mix_a_ridge.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
