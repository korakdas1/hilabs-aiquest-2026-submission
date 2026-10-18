# TermMap - HiLabs AI Quest 2026

Local clinical term mapping from extracted diagnoses and conditions to ICD-10-CM and SNOMED CT. The submitted path uses SQLite FTS5, TF-IDF retrieval and separate LightGBM rankers. It runs locally after dependency installation and uses no hosted API by default.

**This is a source-only model release: no pretrained fitted artifacts, training records, proprietary dictionaries, evaluation predictions or credentials are included.** Authorized reviewers must provide the challenge data separately and fit the local model before inference. The package includes synthetic demonstration inputs, tests, a three-page report and a captioned demo recording.

- [Three-page submission report](output/pdf/termmap_submission_report.pdf)
- [Demo recording](output/video/termmap_demo.mp4) and [runbook](DEMO.md)
- [Supporting assets on Google Drive](https://drive.google.com/drive/folders/1ekAAR7IXt-S8xGq9QeSjdZR-iDEuiEUx?usp=sharing)
- [Model card](MODEL_CARD.md), [limitations](DIAGNOSIS.md), [release audit](PRIVACY_REVIEW.md)

## Installation and private inputs

Use Python 3.12 or 3.13 with SQLite FTS5 support. Run from this repository directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Place these authorized input files in `private_data/` (ignored by Git):

| File | Required layout |
| --- | --- |
| `hackathon_train.csv` | UTF-8 CSV with `entity,chunk_text,coding_system,code,display,subconcept` |
| `icd10cm_codes.xlsx` | First worksheet: code in column A, description in column B; first row is a header |
| `SNOMEDCT_US_codes_and_descriptions.xlsx` | First worksheet: code in column A, description in column B; first row is a header |

Training keeps only `DiagnosisCode` rows with coding system `ICD10CM` or `SNOMEDCT_US`. Labels absent from the supplied terminology are excluded. Obtain these files through the authorized challenge channel; this repository does not distribute them. Dependencies are pinned in `requirements.txt`; a fresh pinned installation was not performed during release checks.

## Training and prediction

```bash
.venv/bin/python -m termmap.cli --data-dir private_data prepare
.venv/bin/python -m termmap.cli --data-dir private_data train
.venv/bin/python -m termmap.cli --data-dir private_data predict --input examples/demo_input.csv --output artifacts/demo_predictions.csv
```

For the evaluation file, the submission command is:

```bash
.venv/bin/python -m termmap.cli --data-dir private_data predict --input private_data/evaluation.csv --output artifacts/predictions.csv
```

Global options such as `--data-dir` and `--artifacts-dir` go before the subcommand. Prediction builds missing terminology and fits missing or invalidated rankers automatically; the first run is not pretrained inference. Keep the source files available for cache validation. Training and index creation can take substantial time. All derived artifacts stay private under `artifacts/`; fitted objects retain training examples and must not be published.

Prediction CSV requires `clinical_entity` (alias: `entity`). Optional columns are `chunk_id` and `chunk_text` (aliases: `context` or `text`). Missing IDs become one-based row numbers. Empty entity values are rejected. This example is synthetic:

```csv
chunk_id,clinical_entity,chunk_text
synthetic-1,HTN,Synthetic example of hypertension.
```

The output preserves row order and has exactly:

```text
chunk_id,clinical_entity,icd10cm_code,snomed_code
```

Optional `--details-output artifacts/candidates.jsonl` writes candidate diagnostics; these can contain clinical text and must remain private. The default path is intended to be repeatable for fixed inputs, source code, dependencies and artifacts; cross-platform bitwise equivalence is not claimed.

## Tests and evaluation

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q -rs
```

Lightweight tests use synthetic fixtures and mock hosted requests. Integration tests skip when private training data or fitted artifacts are absent. The FAISS alias test skips if the optional dependency is unavailable. Tests do not require fitting a model or downloading weights.

Authorized evaluators may separately run the following training-intensive grouped validation command (not run during release preparation):

```bash
.venv/bin/python -m termmap.cli --data-dir private_data evaluate --seed 42 --output artifacts/validation.json
```

## Results and limitations

These historical aggregate results are carried forward from the original submission documentation and recording; they were not rerun or independently verified during this release audit. Private row-level evidence and fitted artifacts are excluded.

| Historical path / split | ICD exact | SNOMED exact |
| --- | ---: | ---: |
| Submitted local / seed 42 | 226/471 (48.0%) | 260/537 (48.4%) |
| Local / seed 43 | 238/451 (52.8%) | 269/546 (49.3%) |
| Optional DeepSeek 1017 / seed 42 | 307/471 (65.2%) | 328/537 (61.1%) |

Validation groups normalized entities into an 80/20 fitting/holdout split. Development choices used seed 42, so that result is not an untouched final test. These are not hidden challenge evaluation scores. No documented path reaches the 95% target. Abbreviation ambiguity, spelling noise, negation, history, specificity and unseen codes remain weaknesses. Dictionary membership does not establish clinical correctness or billing validity. Candidate scores are not calibrated probabilities. Outputs need qualified review before clinical or billing use.

## Optional research code

`requirements-neural.txt`, `termmap/neural.py` and `termmap/learned.py` preserve optional SapBERT/BOND retrieval and learned reranking. They require additional dependencies and model downloads; weights and indexes are not included. `--neural`, `--learned`, `prepare-neural` and `train-neural` are excluded from the submitted default path and were not run during release verification.

`--use-openrouter` is explicit opt-in hosted adjudication. It requires `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` and `OPENROUTER_ZERO_RETENTION_CONFIRMED=true` in the local environment. Enabling it sends the entity, nearby context and candidates to an external provider. The code requests zero-retention routing; that setting is not an independent guarantee of provider behavior or authorization to send data. No hosted requests were made during release verification. Credentials, caches and all clinical outputs must stay private.

Optional licensed UMLS/RF2 imports are supported through CLI flags shown by `--help`; licensed releases are not included. The default predictor does not need them.

## Demo and privacy

The 151-second silent recording uses the synthetic `examples/hard_cases.csv`. It shows a historical fitted local model, including an abbreviation failure, and does not mean those fitted artifacts are shipped. Its root-level input-file commands remain supported; the commands above use `private_data/` for clearer separation. See `DEMO.md` for details.

Only reviewed source, dependency specifications, synthetic examples/tests, documentation, the sanitized PDF and the synthetic demo video belong in this release. Do not add original CSV/XLSX files, model objects, caches, prediction files, diagnostic logs, credentials or environment files. See `PRIVACY_REVIEW.md` for the audit scope. Supporting Drive assets are separate from this Git release.
