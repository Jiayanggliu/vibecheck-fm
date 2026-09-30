#!/usr/bin/env python3
"""Compare shorter VibeCheck clip sets with leakage-safe cross-validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline import FEATURES, TRAITS, load_mix_a


CLIP_COUNTS = (5, 8, 10, 15, 25)


def correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.corrcoef(actual, predicted)[0, 1])


def rank_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank clips by mean absolute correlation with the five targets."""
    scores = []
    for feature_index in range(x.shape[1]):
        feature = x[:, feature_index]
        trait_scores = [
            abs(correlation(feature, y[:, trait_index]))
            for trait_index in range(y.shape[1])
        ]
        scores.append(np.mean(trait_scores))
    return np.argsort(scores)[::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/clip-counts.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, targets = load_mix_a(args.input)
    x = features.to_numpy(dtype=float)
    y = targets.to_numpy(dtype=float)
    splitter = RepeatedKFold(n_splits=5, n_repeats=3, random_state=args.seed)

    fold_scores = {count: {trait: [] for trait in TRAITS} for count in CLIP_COUNTS}
    selection_counts = {count: Counter() for count in CLIP_COUNTS if count < len(FEATURES)}

    for train_index, test_index in splitter.split(x):
        ranking = rank_features(x[train_index], y[train_index])
        for count in CLIP_COUNTS:
            selected = ranking[:count]
            if count < len(FEATURES):
                selection_counts[count].update(FEATURES[index] for index in selected)

            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=np.logspace(-3, 3, 25)),
            )
            model.fit(x[train_index][:, selected], y[train_index])
            predicted = model.predict(x[test_index][:, selected])
            for trait_index, trait in enumerate(TRAITS):
                fold_scores[count][trait].append(
                    correlation(y[test_index, trait_index], predicted[:, trait_index])
                )

    results = {
        "dataset": "Nave et al. Study 1, Mix A",
        "users": len(features),
        "method": "5-fold cross-validation repeated 3 times; feature ranking occurs within each training fold",
        "clip_counts": {},
    }
    for count in CLIP_COUNTS:
        results["clip_counts"][str(count)] = {
            "pearson_r_mean": {
                trait: float(np.mean(fold_scores[count][trait])) for trait in TRAITS
            },
            "pearson_r_std": {
                trait: float(np.std(fold_scores[count][trait], ddof=1)) for trait in TRAITS
            },
        }
        if count < len(FEATURES):
            results["clip_counts"][str(count)]["selection_frequency"] = dict(
                selection_counts[count].most_common()
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

