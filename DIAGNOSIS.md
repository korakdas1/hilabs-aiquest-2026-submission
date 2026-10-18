# Failure analysis - public release

Private holdout quotations, row-level diagnoses, predictions and source-derived examples have been removed. This document summarizes failure categories only; it is not a distributed diagnostic dataset.

- Retrieval can miss rare or unseen concepts, leaving the ranker without the correct candidate.
- Ranking can confuse related concepts and different levels of specificity even when the correct candidate is present.
- Ambiguous abbreviations can outweigh nearby context. The synthetic `MS` demonstration in the video is an illustration, not a held-out clinical record.
- Negation, personal/family history, laterality and temporal qualifiers require more reliable context handling.
- Optional neural or hosted stages can improve some choices while changing correct local answers to incorrect ones.

Historical aggregate results and their limitations are recorded in MODEL_CARD.md. They remain below the challenge target. Future work would improve candidate recall, validate ranking on fresh groups, and use qualified review for ambiguous labels. These are proposed steps, not new experiments or performance claims.
