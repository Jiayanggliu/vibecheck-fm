# Dataset notes

## Study 1

- 23,737 rows total
- 22,252 rows with all five personality targets
- 19,135 participants in the mixed-genre condition
- 17,904 mixed-genre participants with complete personality targets
- 25 ratings in Mix A are the baseline inputs
- Targets are Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism

This is suitable for reproducing the paper's fixed-excerpt prediction task. It is not direct evidence that the same model works on streaming histories.

## Study 2

- 21,929 users
- Five personality targets plus age and gender
- 62,035 sparse Facebook music-like features
- The included matrix has no artist names or stable external identifiers for its feature columns

The missing feature mapping prevents a new user from being encoded for inference. Keep Study 2 as research evidence, not the MVP input pipeline.

