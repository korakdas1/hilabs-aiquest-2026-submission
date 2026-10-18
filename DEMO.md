# TermMap demo runbook

[Supporting assets on Google Drive](https://drive.google.com/drive/folders/1ekAAR7IXt-S8xGq9QeSjdZR-iDEuiEUx?usp=sharing)

The included `output/video/termmap_demo.mp4` is a 151-second silent, captioned recording using the synthetic `examples/hard_cases.csv`. It shows a historical fitted local model. Fitted artifacts, original challenge data and proprietary dictionaries are not included in this release. The recording's predictions are synthetic demo output, not evaluation predictions or accuracy evidence.

## Reproduce with authorized private inputs

Complete the README installation and provide the three authorized challenge files in `private_data/`. Then run:

```bash
.venv/bin/python -m termmap.cli --data-dir private_data train
.venv/bin/python -m termmap.cli --data-dir private_data predict --input examples/hard_cases.csv --output artifacts/demo_predictions.csv
.venv/bin/python -m termmap.cli --data-dir private_data demo --term "MS" --context "Neurology follow-up for multiple sclerosis with gait difficulty."
```

The recording uses files at the repository root; that equivalent layout remains supported. The caption tool `tools/live_demo.py --auto` uses that original root-level layout and an already prepared `.venv`. Run it only in a private workspace after fitting, with Tk available. Do not expose input workbooks, training rows, credentials or unrelated applications in a new recording.

## Sequence

1. Explain normalization, independent retrieval, two local rankers, legal-code checks and the four-column CSV.
2. Show the synthetic abbreviation, spelling-noise, laterality and specificity cases. Preserve input IDs and order.
3. Inspect the recorded `MS` failure: the model selected an altered-mental-status code despite the synthetic neurological context. The displayed scores are rankings, not probabilities. New fits may produce different outputs.
4. Explain the historical results consistently: local seed 42 is 48.0% ICD / 48.4% SNOMED; local seed 43 is 52.8% / 49.3%; optional DeepSeek 1017 seed 42 is 65.2% / 61.1%. These figures were not rerun for the release and no documented path reaches 95%.
5. Show the README prediction command and state that private inputs and local training are required. Hosted and neural options are excluded from the default submission path.

The three-page report has been sanitized for public release. Replace any earlier report on Drive with `output/pdf/termmap_submission_report.pdf`. The release audit does not certify uninspected files or permissions on Drive.
