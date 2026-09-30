# Persona system

## Decision

VibeCheck uses eight entertainment personas generated from the model's five predicted Big Five scores. The personas are deterministic: the same predicted profile always maps to the same nearest cluster.

They are not diagnoses, validated psychological types, or substitutes for the continuous scores.

## Method

1. Generate five-fold out-of-fold Big Five predictions for all 17,904 eligible Mix A users using the selected 15 clips.
2. Standardize the five predicted traits using the out-of-fold prediction distribution.
3. Fit eight K-means clusters with a fixed random seed and 50 initializations.
4. Name each cluster from its centroid pattern.
5. For a new user, predict five trait scores, standardize them with the saved scaler, and assign the nearest saved cluster.

Using out-of-fold predictions makes the persona distribution closer to what unseen users will receive. It also prevents the cluster design from relying on the true personality labels at inference time.

## Personas and validation distribution

| Persona | Share | Main predicted pattern |
| --- | ---: | --- |
| Soft-Spoken Sentinel | 15.0% | Lower E; higher A and N |
| Warm Curator | 14.2% | Higher O, C, and A |
| Restless Auteur | 14.2% | Higher O; lower C, E, and A |
| Grounded Contrarian | 13.1% | Lower O and A; near-average E |
| Midnight Drifter | 12.6% | Lower O, C, E, and A; higher N |
| Radiant Organizer | 10.6% | Higher C, E, and A; lower N |
| Golden Regular | 10.4% | Higher C, E, and A; lower O |
| Electric Explorer | 10.0% | Higher O and E; lower N |

No persona contains more than 15.0% or fewer than 10.0% of the validation population. This is a useful balance check, not evidence that the types are psychologically real.

## Product rules

- Display the playful persona first and the five continuous predicted scores second.
- Say “music persona,” not “personality diagnosis” or “psychological profile.”
- Explain that results are based on reactions to 15 short music clips.
- Do not infer sensitive traits, mental health, intelligence, politics, or sexuality.
- Do not silently change names, centroids, scaling, or clip order. Version those changes and re-run distribution checks.

