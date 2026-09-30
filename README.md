# VibeCheck

VibeCheck explores whether reactions to music can estimate broad personality traits and turn the result into a playful music persona.

## Current milestone

This repository starts with a reproducible baseline using Study 1 from Nave et al. (2018). The model uses ratings for 25 music excerpts to predict continuous Big Five scores.

It does **not** yet predict personality from arbitrary Spotify history. The supplied Study 2 Facebook-like matrix has no artist-to-column mapping, so new users cannot be encoded consistently.

## Data setup

Download the OSF archive and keep it outside Git. The script accepts either the complete OSF ZIP or the extracted training CSV:

```text
<archive>/Study 1/Data/Study1_data.csv
```

The archive also contains music excerpts. Do not commit or redistribute them without confirming the applicable rights.

## Run the baseline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_baseline.py \
  --input "/path/to/nfqb9-osfstorage-archive.zip"
```

The script uses the largest cohort (`Condition == 3`, Mix A), holds out 20% of users, and writes local metrics and a fitted model under `artifacts/`.

### Reproduced baseline

Using random seed 42, the local 80/20 holdout produced these Pearson correlations:

| Trait | Pearson r |
| --- | ---: |
| Openness | 0.251 |
| Extraversion | 0.175 |
| Agreeableness | 0.148 |
| Conscientiousness | 0.118 |
| Neuroticism | 0.102 |

These are weak signals. The experience should present the output as entertainment and avoid claims of psychological assessment.

## Important limitations

- The target is a noisy estimate of broad traits, not a diagnosis.
- The training input is a fixed 25-clip rating task, not listening history.
- Accuracy should be communicated honestly, especially for traits other than Openness.
- Age and gender are deliberately excluded from the baseline.
- Before public or commercial use, confirm the dataset and audio licensing terms.
