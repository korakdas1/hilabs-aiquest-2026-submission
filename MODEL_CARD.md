# TermMap model card

## Scope and release contents

Map an already-extracted diagnosis or condition, with optional context, to one code in each authorized ICD-10-CM and SNOMED CT dictionary. Entity extraction and non-diagnosis categories are outside the submitted scope. This release contains source code, not a fitted model. It does not include proprietary datasets, dictionaries, model weights, training-derived indexes or evaluation predictions.

## Method

Normalize text and stable abbreviations; retrieve candidates independently per system with SQLite FTS5 and character/word/context TF-IDF; rank with separate LightGBM models; restrict selected codes to the supplied dictionaries. Training uses only authorized diagnosis labels. Fitted objects retain fitting examples, so all generated artifacts are private.

## Historical evaluation

The original documentation and recorded demo report the following exact-match results. They were retained for consistency, not rerun during release preparation; private row-level evidence is not distributed.

| Path / grouped split | ICD-10-CM | SNOMED CT |
| --- | ---: | ---: |
| Local / seed 42 | 226/471 (48.0%) | 260/537 (48.4%) |
| Local / seed 43 | 238/451 (52.8%) | 269/546 (49.3%) |
| Optional DeepSeek 1017 / seed 42 | 307/471 (65.2%) | 328/537 (61.1%) |

The split groups normalized entity strings across coding systems with 80/20 fitting/holdout allocation. Development choices used seed 42. Grouping by entity does not establish patient-level or institution-level independence. The hidden challenge evaluation is not represented. Optional neural and hosted paths are experimental; hosted inference is excluded from the default submission command.

## Limitations and safe interpretation

No documented path meets the 95% target. Weaknesses include ambiguous abbreviations, rare or unseen codes, close specificity variants, spelling noise, negation and temporal context. A code can be in the dictionary and still be wrong. Ranking scores are not probabilities. No clinical deployment, calibration, fairness or prospective validation is established. Qualified review is required before clinical or billing use.

## Reproduction and privacy

See README.md for installation, schemas, training and prediction. Authorized reviewers must supply the private challenge inputs and train locally. Optional model downloads and hosted APIs were not exercised during release checks. Keep derived artifacts and diagnostics private. The included tests and examples are synthetic; the video is a historical synthetic demonstration rather than accuracy evidence.
