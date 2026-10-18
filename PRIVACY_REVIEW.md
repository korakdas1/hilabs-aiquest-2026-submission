# Public-release privacy review

## Scope and outcome

Reviewed the complete submission folder against its original ZIP: all packaged files matched before cleanup. No content was imported from a separate research repository. The release retains source, dependency specifications, synthetic tests/examples, documentation, a sanitized three-page PDF and the synthetic demonstration recording.

Private holdout quotations and source-derived clinical examples were removed from README, MODEL_CARD, DIAGNOSIS, the PDF and its generator. Environment-file templates were removed. Documentation now clearly states that fitted artifacts and proprietary dictionaries are absent. Ignore rules cover private datasets, credentials, model objects, caches, predictions and diagnostic outputs. Generic filenames required by the code are documented, but the underlying data is excluded.

The release file inventory and staged text were checked for original datasets, model artifacts, predictions, private diagnostic content, credentials and absolute workstation paths. PDF text and metadata were inspected and all three rendered pages were reviewed. The silent video's metadata, sampled frames and scene transitions were inspected; it contains synthetic demo cases, candidate displays and historical aggregate figures. These demo predictions are not evaluation-set predictions. No private records or secrets were found in the reviewed release files. This is a bounded release audit, not a claim that future generated outputs are safe to publish.

## Verification

- Python syntax parsed successfully for all 12 source, test and tool files.
- CLI help loaded successfully.
- Existing lightweight test suite: **9 passed, 2 skipped**.
- Skips: private training CSV unavailable; fitted-model/dictionary integration artifacts unavailable.
- No optional-dependency test skips on this host; FAISS was available. The suite now explicitly skips its FAISS test when that optional dependency is missing.
- No retraining, model downloads, hosted API calls or new experiments were performed. Neural model inference and learned training were not exercised.
- The test run used existing host dependencies plus small temporary installs of openpyxl, rapidfuzz and reportlab. A fresh installation of the full pinned environment was not performed.
- The PDF has exactly three pages. Historical accuracy figures match the README, model card and demo documentation; they were not independently reproduced during this audit.

## Separate assets

The sanitized PDF and rebuilt ZIP replace the local submission copies. Any older copies uploaded to Drive must be replaced separately. This audit does not inspect or certify the Drive folder's contents or permissions. Keep private datasets and fitted artifacts out of publicly shared supporting assets.
