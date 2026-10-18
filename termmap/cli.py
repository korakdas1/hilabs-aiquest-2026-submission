"""Command-line entry point for indexing, fitting, evaluation and inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from . import __version__
from .data import iter_input, load_training
from .evaluation import evaluate
from .model import TermMapper
from .openrouter import OpenRouterAdjudicator
from .terminology import Terminology, build_terminology


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(data_dir: Path, terminology_path: Path) -> dict[str, object]:
    with Terminology(terminology_path) as terminology:
        signatures = terminology.source_signatures()
    code_hashes = {name: _hash(Path(__file__).parent / name)
                   for name in ("data.py", "terminology.py", "model.py")}
    return {"engine_version": __version__, "training_sha256": _hash(data_dir / "hackathon_train.csv"),
            "terminology_sources": signatures, "implementation_sha256": code_hashes}


def _ensure_mapper(data_dir: Path, terminology_path: Path, model_path: Path,
                   force: bool = False, licensed_sources: dict | None = None) -> TermMapper:
    build_terminology(data_dir, terminology_path, **(licensed_sources or {}))
    manifest_path = model_path.with_suffix(".manifest.json")
    expected = _manifest(data_dir, terminology_path)
    if not force and model_path.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) == expected:
            return TermMapper.load(model_path, terminology_path)
    examples = load_training(data_dir / "hackathon_train.csv")
    print(f"Fitting on {len(examples):,} diagnosis labels", flush=True)
    mapper, metrics = TermMapper.train(examples, terminology_path, output=model_path)
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return mapper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local clinical term mapper for ICD-10-CM and SNOMED CT")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing the supplied CSV and XLSX files")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Local cache directory")
    parser.add_argument("--umls-mrconso", type=Path, help="Locally licensed UMLS MRCONSO.RRF")
    parser.add_argument("--snomed-rf2-descriptions", type=Path, help="Locally licensed SNOMED RF2 description Snapshot")
    parser.add_argument("--snomed-rf2-relationships", type=Path, help="Locally licensed SNOMED RF2 relationship Snapshot")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build the terminology search index")
    prep.add_argument("--force", action="store_true")
    prep_neural = sub.add_parser("prepare-neural", help="Build local SapBERT FAISS indexes and download licensed model weights")
    prep_neural.add_argument("--limit-codes", type=int, help="Smoke-test only; omitted for the complete index")
    train = sub.add_parser("train", help="Fit both local ranking models")
    train.add_argument("--force", action="store_true")
    sub.add_parser("train-neural", help="Fit the local query adapter and biomedical cross-encoder")
    check = sub.add_parser("evaluate", help="Run grouped held-out validation")
    check.add_argument("--output", type=Path, default=Path("artifacts/validation.json"))
    check.add_argument("--seed", type=int, default=42)
    check.add_argument("--use-openrouter", action="store_true", help="Compare optional OpenRouter on a bounded holdout sample")
    check.add_argument("--api-limit", type=int, default=25, help="Maximum API adjudications per coding system")
    check.add_argument("--neural", action="store_true", help="Evaluate local SapBERT and BOND reranking")
    check.add_argument("--learned", action="store_true", help="Train and evaluate the supervised neural path on this split")
    predict = sub.add_parser("predict", help="Predict both codes for every input row")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--details-output", type=Path, help="Optional local JSONL of top candidates")
    predict.add_argument("--use-openrouter", action="store_true", help="Opt-in zero-retention LLM adjudication")
    predict.add_argument("--neural", action="store_true", help="Use local SapBERT FAISS retrieval and BOND reranking")
    predict.add_argument("--learned", action="store_true", help="Use the fitted query adapter and cross-encoder")
    demo = sub.add_parser("demo", help="Map a term and show ranked candidates")
    demo.add_argument("--term", required=True)
    demo.add_argument("--context", default="")
    demo.add_argument("--use-openrouter", action="store_true", help="Opt-in zero-retention LLM adjudication")
    demo.add_argument("--neural", action="store_true", help="Use local SapBERT FAISS retrieval and BOND reranking")
    demo.add_argument("--learned", action="store_true", help="Use the fitted query adapter and cross-encoder")
    args = parser.parse_args(argv)

    data_dir = args.data_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    terminology_path = artifacts_dir / "terminology.sqlite"
    model_path = artifacts_dir / "model.joblib"
    licensed_sources = {"umls_mrconso": args.umls_mrconso,
                        "snomed_rf2_descriptions": args.snomed_rf2_descriptions,
                        "snomed_rf2_relationships": args.snomed_rf2_relationships}
    if args.command == "prepare":
        print(build_terminology(data_dir, terminology_path, force=args.force, **licensed_sources))
        return 0
    if args.command == "prepare-neural":
        from .neural import build_dense_index
        build_terminology(data_dir, terminology_path, **licensed_sources)
        print(build_dense_index(terminology_path, artifacts_dir / "neural", limit_codes=args.limit_codes))
        return 0
    if args.command == "evaluate":
        if args.api_limit < 1:
            parser.error("--api-limit must be positive")
        adjudicator = (OpenRouterAdjudicator(artifacts_dir / "openrouter_cache.jsonl", data_dir / ".env")
                       if args.use_openrouter else None)
        build_terminology(data_dir, terminology_path, **licensed_sources)
        examples = load_training(data_dir / "hackathon_train.csv")
        neural_dir = None
        if args.neural or args.learned:
            from .neural import build_dense_index
            neural_dir = build_dense_index(terminology_path, artifacts_dir / "neural")
        report = evaluate(examples, terminology_path, args.output, seed=args.seed,
                          adjudicator=adjudicator, api_limit=args.api_limit,
                          neural_dir=neural_dir, learned=args.learned)
        print(json.dumps(report, indent=2))
        return 0
    adjudicator = (OpenRouterAdjudicator(artifacts_dir / "openrouter_cache.jsonl", data_dir / ".env")
                   if getattr(args, "use_openrouter", False) else None)
    mapper = _ensure_mapper(data_dir, terminology_path, model_path,
                            force=getattr(args, "force", False), licensed_sources=licensed_sources)
    runtime = mapper
    if getattr(args, "neural", False) or getattr(args, "learned", False) or args.command == "train-neural":
        from .neural import NeuralMapper, build_dense_index
        neural_dir = build_dense_index(terminology_path, artifacts_dir / "neural")
        learned_dir = None
        if getattr(args, "learned", False) or args.command == "train-neural":
            from .learned import train_learned
            with Terminology(terminology_path) as terminology:
                learned_dir = train_learned(load_training(data_dir / "hackathon_train.csv"),
                                            mapper, terminology, neural_dir,
                                            artifacts_dir / "learned_full")
        if args.command != "train-neural":
            runtime = NeuralMapper(mapper, neural_dir, learned_dir=learned_dir)
    if args.command == "train":
        print(model_path)
        return 0
    if args.command == "train-neural":
        print(artifacts_dir / "learned_full")
        return 0
    if args.command == "demo":
        with Terminology(terminology_path) as terminology:
            prediction, details = runtime.predict_with_db(args.term, args.context, terminology,
                                                         top_k=50 if adjudicator else 5)
            if adjudicator:
                prediction = adjudicator.adjudicate(args.term, args.context, prediction, details, terminology)
        print(json.dumps({"prediction": prediction, "candidates": details}, indent=2))
        return 0
    if args.command == "predict":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = args.output.with_name(args.output.name + ".partial")
        details_handle = None
        if args.details_output:
            args.details_output.parent.mkdir(parents=True, exist_ok=True)
            details_handle = args.details_output.open("w", encoding="utf-8")
        count = 0
        try:
            with Terminology(terminology_path) as terminology, temp_output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"])
                writer.writeheader()
                for count, row in enumerate(iter_input(args.input), 1):
                    prediction, details = runtime.predict_with_db(row.entity, row.context, terminology,
                                                                  top_k=50 if adjudicator else (5 if details_handle else 1))
                    if adjudicator:
                        try:
                            prediction = adjudicator.adjudicate(row.entity, row.context, prediction,
                                                                details, terminology)
                        except (OSError, ValueError, KeyError, IndexError) as exc:
                            print(f"OpenRouter failed on input row {count}: {type(exc).__name__}; using local codes", flush=True)
                    writer.writerow({"chunk_id": row.chunk_id, "clinical_entity": row.entity,
                                     "icd10cm_code": prediction["ICD10CM"],
                                     "snomed_code": prediction["SNOMEDCT_US"]})
                    if details_handle:
                        details_handle.write(json.dumps({"chunk_id": row.chunk_id, "clinical_entity": row.entity,
                                                         "candidates": details}) + "\n")
                    if count % 100 == 0:
                        print(f"Predicted {count:,} rows", flush=True)
            os.replace(temp_output, args.output)
        except BaseException:
            temp_output.unlink(missing_ok=True)
            raise
        finally:
            if details_handle:
                details_handle.close()
        print(f"Wrote {count:,} predictions to {args.output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
