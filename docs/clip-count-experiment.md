# Clip-count experiment

## Decision

Use **15 Mix A clips** for the first MVP experiment.

This is a research decision, not a claim that 15 clips provide a clinically meaningful personality assessment. All five targets remain weak predictions.

## Method

- 17,904 users with complete Big Five targets
- Five-fold cross-validation repeated three times
- Clip ranking recalculated inside every training fold to prevent test leakage
- Clips ranked by their mean absolute correlation with the five targets
- Standardized multi-target ridge regression
- Pearson correlation used as the main comparison metric

## Results

| Clips | O | C | E | A | N |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.190 | 0.065 | 0.150 | 0.107 | 0.102 |
| 8 | 0.211 | 0.078 | 0.152 | 0.134 | 0.102 |
| 10 | 0.220 | 0.089 | 0.153 | 0.137 | 0.103 |
| **15** | **0.237** | **0.094** | **0.155** | **0.144** | **0.107** |
| 25 | 0.248 | 0.113 | 0.159 | 0.153 | 0.111 |

Compared with all 25 clips, the 15-clip version retains 95.3% of the Openness correlation, 97.7% of Extraversion, 94.2% of Agreeableness, and 96.5% of Neuroticism. It retains 82.9% for Conscientiousness, which is weak even in the full model.

## Stable 15-clip set

The following clips appeared in the selected set in at least 14 of the 15 cross-validation folds:

| Test column | Artist | Title | Genre |
| --- | --- | --- | --- |
| TestA_1 | Bruce Smith | Children of Spring | Easy Listening |
| TestA_2 | The O'Neill Brothers | Through the Years | Smooth |
| TestA_3 | Walter Rodriguez | Safety | Quiet Storm |
| TestA_4 | Frank Josephs | Mountain Trek | Quiet Storm |
| TestA_5 | Taryn Murphy | Love Along The Way | Soft Rock |
| TestA_6 | Magic Dingus Box | The Way It Goes | Electronica |
| TestA_8 | Robert LaRow | Sexy | Europop |
| TestA_9 | Mykill Miers | Immaculate | Rap/Hip-Hop |
| TestA_10 | Sammy Smash | Get the Party Started | Rap/Hip-Hop |
| TestA_12 | Bruce Smith | Sonata A Major | Classical |
| TestA_13 | Antonio Vivaldi | Concerto in C | Classical |
| TestA_14 | Various Artists | La Trapera | Latin |
| TestA_15 | Paul Serrato & Co. | Who are You? | Traditional Jazz |
| TestA_23 | Laura Hawthorne | Famous Right Where | Mainstream Country |
| TestA_24 | James E. Burns | I’m Already Over You | New Country |

## Product implication

Treat 15 clips as the research default. Before UI development, run a small completion-time usability test. If 15 clips cause unacceptable dropout, use 10 clips and explicitly accept the measured accuracy tradeoff.

Do not publish the supplied audio until its redistribution rights are confirmed.

