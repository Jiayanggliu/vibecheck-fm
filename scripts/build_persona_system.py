#!/usr/bin/env python3
"""Build deterministic VibeCheck personas from cross-validated model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_baseline import TRAITS, load_mix_a


SELECTED_CLIPS = [
    "TestA_1", "TestA_2", "TestA_3", "TestA_4", "TestA_5",
    "TestA_6", "TestA_8", "TestA_9", "TestA_10", "TestA_12",
    "TestA_13", "TestA_14", "TestA_15", "TestA_23", "TestA_24",
]

# Names describe model-output patterns, not clinical or diagnostic categories.
PERSONAS = {
    0: {
        "name": "Midnight Drifter",
        "description": "Private, emotionally vivid, and drawn to familiar sonic spaces.",
    },
    1: {
        "name": "Warm Curator",
        "description": "Open-minded, considerate, and intentional about what enters the rotation.",
    },
    2: {
        "name": "Grounded Contrarian",
        "description": "Direct, independent, and selective about musical novelty.",
    },
    3: {
        "name": "Restless Auteur",
        "description": "Imaginative, unconventional, and comfortable following a personal compass.",
    },
    4: {
        "name": "Radiant Organizer",
        "description": "Social, steady, and energized by music that brings people together.",
    },
    5: {
        "name": "Electric Explorer",
        "description": "Curious, outgoing, and quick to chase a fresh sound.",
    },
    6: {
        "name": "Soft-Spoken Sentinel",
        "description": "Reserved, caring, and attentive to music's emotional undercurrent.",
    },
    7: {
        "name": "Golden Regular",
        "description": "Reliable, upbeat, and happiest when the vibe feels welcoming.",
    },
}


def make_model():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 25)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features, targets = load_mix_a(args.input)
    x = features[SELECTED_CLIPS]
    y = targets.to_numpy(dtype=float)

    folds = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_predictions = cross_val_predict(make_model(), x, y, cv=folds)

    persona_scaler = StandardScaler().fit(oof_predictions)
    standardized = persona_scaler.transform(oof_predictions)
    clusterer = KMeans(n_clusters=8, random_state=args.seed, n_init=50).fit(standardized)

    counts = np.bincount(clusterer.labels_, minlength=8)
    report = {
        "users": int(len(x)),
        "traits": TRAITS,
        "selected_clips": SELECTED_CLIPS,
        "method": "K-means on standardized five-fold out-of-fold predictions",
        "personas": [],
    }
    for cluster_id in range(8):
        report["personas"].append(
            {
                "cluster_id": cluster_id,
                **PERSONAS[cluster_id],
                "users": int(counts[cluster_id]),
                "share": float(counts[cluster_id] / len(x)),
                "centroid_z": {
                    trait: float(clusterer.cluster_centers_[cluster_id, index])
                    for index, trait in enumerate(TRAITS)
                },
            }
        )

    final_model = make_model().fit(x, y)
    bundle = {
        "prediction_model": final_model,
        "persona_scaler": persona_scaler,
        "clusterer": clusterer,
        "personas": PERSONAS,
        "features": SELECTED_CLIPS,
        "traits": TRAITS,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_dir / "persona_system.joblib")
    (args.output_dir / "persona-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

