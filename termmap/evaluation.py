"""Reproducible group holdout evaluation with no test terms in the fit."""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import joblib

from sklearn.model_selection import GroupShuffleSplit

from .data import Example, SYSTEMS, normalize
from .model import TermMapper
from .openrouter import OpenRouterAdjudicator
from .terminology import Terminology


def evaluate(examples: list[Example], terminology_path: str | Path,
             output: str | Path | None = None, test_size: float = 0.2,
             seed: int = 42, adjudicator: OpenRouterAdjudicator | None = None,
             api_limit: int = 25, neural_dir: str | Path | None = None,
             learned: bool = False) -> dict[str, object]:
    groups = [normalize(x.entity) for x in examples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(examples, groups=groups))
    train = [examples[i] for i in train_idx]
    test = [examples[i] for i in test_idx]
    mapper, training = TermMapper.train(train, terminology_path, verbose=True)
    neural = None
    if neural_dir is not None:
        from .neural import NeuralMapper, SAPBERT_MODEL, BOND_MODEL
        learned_dir = None
        if learned:
            from .learned import train_learned
            learned_dir = Path(neural_dir).parent / f"learned_holdout_seed{seed}"
            with Terminology(terminology_path) as terminology:
                train_learned(train, mapper, terminology, neural_dir, learned_dir)
        neural = NeuralMapper(mapper, neural_dir, learned_dir=learned_dir)
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": mapper.models, "terminology_path": str(terminology_path)},
                    output_path.with_suffix(".holdout.joblib"), compress=3)
    report: dict[str, object] = {"split": "grouped by normalized entity", "seed": seed,
                                  "training": training, "systems": {},
                                  "neural_models": ([SAPBERT_MODEL, "trained SapBERT cross-encoder"] if learned
                                                    else [SAPBERT_MODEL, BOND_MODEL] if neural is not None else [])}
    if adjudicator is not None:
        report["openrouter_model"] = adjudicator.model
    error_rows: list[dict[str, object]] = []
    with Terminology(terminology_path) as terminology:
        for system in SYSTEMS:
            model = mapper.models[system]
            samples = [x for x in test if x.system == system and terminology.contains(system, x.code)]
            api_indices = (set(random.Random(seed + SYSTEMS.index(system)).sample(
                range(len(samples)), min(api_limit, len(samples)))) if adjudicator else set())
            api_stats_before = dict(adjudicator.stats) if adjudicator else {}
            correct = top5 = top25 = retrieved = baseline_correct = local_ranker_correct = seen_correct = unseen_correct = 0
            seen_total = unseen_total = 0
            errors = Counter()
            api_success = api_local_correct = api_remote_correct = api_failures = 0
            api_improved = api_regressed = 0
            api_failure_reasons = Counter()
            api_remote_latencies: list[float] = []
            for i, sample in enumerate(samples):
                candidates = model.candidates(sample.entity, sample.context, terminology)
                baseline_correct += int(bool(candidates) and candidates[0].code == sample.code)
                local_ranking = model.rank_candidates(candidates, top_k=5)
                local_ranker_correct += int(local_ranking[0]["code"] == sample.code)
                if neural is not None:
                    vector = neural.dense.encode(sample.entity)
                    ranking, candidate_codes = neural.rank_system(system, sample.entity, sample.context,
                                                                  terminology, vector,
                                                                  top_k=25 if learned else 5)
                else:
                    ranking = local_ranking
                    candidate_codes = {c.code for c in candidates}
                retrieved_here = sample.code in candidate_codes
                retrieved += int(retrieved_here)
                prediction = str(ranking[0]["code"])
                correct_here = prediction == sample.code
                correct += int(correct_here)
                top5 += int(sample.code in {str(x["code"]) for x in ranking[:5]})
                if learned:
                    top25 += int(sample.code in {str(x["code"]) for x in ranking[:25]})
                seen = sample.code in model.code_counts
                if seen:
                    seen_total += 1
                    seen_correct += int(correct_here)
                else:
                    unseen_total += 1
                    unseen_correct += int(correct_here)
                if not correct_here:
                    errors["retrieval_miss" if not retrieved_here else "ranking_error"] += 1
                    error_rows.append({"system": system, "entity": sample.entity,
                                       "gold_code": sample.code, "gold_display": sample.display,
                                       "predicted_code": prediction, "retrieved": retrieved_here,
                                       "baseline_code": candidates[0].code if candidates else "",
                                       "top5": [str(x["code"]) for x in ranking[:5]]})
                if adjudicator is not None and i in api_indices:
                    try:
                        runtime = neural or mapper
                        local_pair, details = runtime.predict_with_db(sample.entity, sample.context,
                                                                      terminology, top_k=50)
                        remote_calls_before = adjudicator.stats["remote_calls"]
                        started = time.perf_counter()
                        remote_pair = adjudicator.adjudicate(sample.entity, sample.context,
                                                              local_pair, details, terminology)
                        if adjudicator.stats["remote_calls"] > remote_calls_before:
                            api_remote_latencies.append(time.perf_counter() - started)
                        api_success += 1
                        local_here = local_pair[system] == sample.code
                        remote_here = remote_pair[system] == sample.code
                        api_local_correct += int(local_here)
                        api_remote_correct += int(remote_here)
                        api_improved += int(not local_here and remote_here)
                        api_regressed += int(local_here and not remote_here)
                    except (OSError, ValueError, KeyError, IndexError) as exc:
                        api_failures += 1
                        reason = type(exc).__name__
                        if getattr(exc, "code", None) is not None:
                            reason += f"_{exc.code}"
                        api_failure_reasons[reason] += 1
                if (i + 1) % 100 == 0:
                    api_progress = (f", API successful {api_success}, failed {api_failures}"
                                    if adjudicator is not None else "")
                    print(f"{system}: evaluated {i + 1}/{len(samples)}{api_progress}", flush=True)
            total = max(1, len(samples))
            report["systems"][system] = {
                "holdout_rows": len(samples), "exact_accuracy": correct / total,
                "baseline_accuracy": baseline_correct / total,
                "local_ranker_accuracy_same_rows": local_ranker_correct / total,
                "top5_accuracy": top5 / total, "candidate_recall": retrieved / total,
                "seen_code_rows": seen_total, "seen_code_accuracy": seen_correct / max(1, seen_total),
                "unseen_code_rows": unseen_total, "unseen_code_accuracy": unseen_correct / max(1, unseen_total),
                "error_counts": dict(errors),
            }
            if learned:
                report["systems"][system]["top25_recall"] = top25 / total
            if adjudicator is not None:
                report["systems"][system]["openrouter_sample"] = {
                    "attempted": len(api_indices), "successful": api_success,
                    "failed": api_failures,
                    "failure_reasons": dict(api_failure_reasons),
                    "local_accuracy_same_rows": api_local_correct / max(1, api_success),
                    "openrouter_accuracy_same_rows": api_remote_correct / max(1, api_success),
                    "improved_rows": api_improved, "regressed_rows": api_regressed,
                    "remote_latency_median_seconds": (round(statistics.median(api_remote_latencies), 3)
                                                      if api_remote_latencies else None),
                    "remote_latency_p95_seconds": (round(sorted(api_remote_latencies)[
                        int(0.95 * (len(api_remote_latencies) - 1))], 3)
                        if api_remote_latencies else None),
                    "usage": {key: round(adjudicator.stats[key] - api_stats_before[key], 8)
                              for key in adjudicator.stats},
                }
    report["headline_lower_exact_accuracy"] = min(
        float(report["systems"][system]["exact_accuracy"]) for system in SYSTEMS)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with output.with_suffix(".errors.jsonl").open("w", encoding="utf-8") as handle:
            for row in error_rows:
                handle.write(json.dumps(row) + "\n")
    return report
