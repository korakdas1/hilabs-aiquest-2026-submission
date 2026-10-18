"""Training-only adaptation of SapBERT retrieval and a clinical cross-encoder.

All supervision comes from the caller's fitting examples.  The fixed SapBERT
code index remains usable: only a small query-side residual is trained.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from .data import Example, normalize
from .model import TermMapper, expanded, local_context
from .neural import SAPBERT_MODEL, SAPBERT_REVISION, DenseIndex
from .terminology import Terminology

SEED = 42
NEGATIVES = 7
MAX_LENGTH = 128


def _seed() -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _query(system: str, entity: str, context: str) -> str:
    near = normalize(local_context(entity, context, radius=90)[0])[:180] if context else ""
    label = "ICD-10-CM" if system == "ICD10CM" else "SNOMED CT"
    return f"{label} diagnosis: {expanded(entity)}" + (f". Chart: {near}" if near else "")


def _description(db: Terminology, system: str, code: str, entity: str,
                 cache: dict[tuple[str, str], list[str]], indexed_only: bool = False) -> str:
    key = (system, code)
    if key not in cache:
        cache[key] = db.descriptions(system, [code], limit_per_code=30)[code] or [code]
    choices = cache[key]
    if indexed_only:
        indexed: list[str] = []
        seen: set[str] = set()
        for desc in choices:
            norm = normalize(desc)
            if norm not in seen:
                indexed.append(desc)
                seen.add(norm)
            if len(indexed) == 2:
                break
        choices = indexed
    return max(choices, key=lambda desc: (fuzz.WRatio(expanded(entity), normalize(desc)), -len(desc)))


def _fingerprint(examples: list[Example], db: Terminology) -> str:
    digest = hashlib.sha256()
    for row in examples:
        digest.update(json.dumps([row.entity, row.context, row.system, row.code],
                                 ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    digest.update(json.dumps(db.source_signatures(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _mine_groups(examples: list[Example], mapper: TermMapper, db: Terminology,
                 dense: DenseIndex) -> tuple[list[tuple[str, list[str], list[str]]], np.ndarray]:
    """Mine distinct same-system wrong codes; insert the gold only for loss."""
    valid = [x for x in examples if db.contains(x.system, x.code)]
    vectors = dense.encoder.encode([expanded(x.entity) for x in valid])
    groups: list[tuple[str, list[str], list[str]]] = []
    desc_cache: dict[tuple[str, str], list[str]] = {}
    kept_vectors: list[np.ndarray] = []
    model_positions = {system: 0 for system in mapper.models}
    for i, row in enumerate(valid):
        model = mapper.models[row.system]
        local = model.candidates(row.entity, row.context, db,
                                 exclude_example=model_positions[row.system])
        model_positions[row.system] += 1
        local_codes = [str(item["code"]) for item in model.rank_candidates(local, top_k=18)]
        dense_codes = list(dense.search_vector(row.system, vectors[i:i + 1], limit=24))
        negatives = list(dict.fromkeys(local_codes[:10] + dense_codes[:15] + local_codes[10:]))
        negatives = [code for code in negatives if code != row.code][:NEGATIVES]
        if len(negatives) < NEGATIVES:
            continue
        codes = [row.code] + negatives
        descriptions = [_description(db, row.system, code, row.entity, desc_cache)
                        for code in codes]
        indexed_descriptions = [_description(db, row.system, code, row.entity, desc_cache,
                                             indexed_only=True) for code in codes]
        groups.append((_query(row.system, row.entity, row.context), descriptions,
                       indexed_descriptions))
        kept_vectors.append(vectors[i])
        if len(groups) % 500 == 0:
            print(f"Mined {len(groups):,} training groups", flush=True)
    return groups, np.stack(kept_vectors).astype(np.float32)


class QueryAdapter:
    """A residual projection on frozen SapBERT mention vectors."""

    def __init__(self, path: str | Path):
        weights = np.load(path, allow_pickle=False)
        self.weight = weights["weight"].astype(np.float32)
        self.bias = weights["bias"].astype(np.float32)

    def transform(self, vector: np.ndarray) -> np.ndarray:
        transformed = vector + vector @ self.weight.T + self.bias
        transformed /= np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-8)
        return np.ascontiguousarray(transformed, dtype=np.float32)


def _train_adapter(groups: list[tuple[str, list[str], list[str]]], query_vectors: np.ndarray,
                   dense: DenseIndex, output: Path, epochs: int = 5) -> None:
    import torch
    from torch import nn

    descriptions = list(dict.fromkeys(desc for _, _, candidates in groups for desc in candidates))
    desc_vectors = dense.encoder.encode(descriptions)
    desc_position = {desc: i for i, desc in enumerate(descriptions)}
    candidate_vectors = np.stack([desc_vectors[[desc_position[d] for d in choices]]
                                  for _, _, choices in groups]).astype(np.float32)
    device = dense.encoder.device
    layer = nn.Linear(query_vectors.shape[1], query_vectors.shape[1]).to(device)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=2e-4, weight_decay=0.02)
    order = np.arange(len(groups))
    for epoch in range(epochs):
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            q = torch.from_numpy(query_vectors[batch]).to(device)
            c = torch.from_numpy(candidate_vectors[batch]).to(device)
            adapted = torch.nn.functional.normalize(q + layer(q), dim=1)
            logits = torch.bmm(c, adapted.unsqueeze(-1)).squeeze(-1) / 0.07
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch),
                                                       dtype=torch.long, device=device))
            loss += 0.05 * layer(q).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        print(f"Query adapter epoch {epoch + 1}/{epochs}: loss={total / len(groups):.4f}", flush=True)
    np.savez(output, weight=layer.weight.detach().cpu().numpy(),
             bias=layer.bias.detach().cpu().numpy())


class TaskCrossEncoder:
    """Locally fitted SapBERT-base pair classifier used as a reranker."""

    def __init__(self, path: str | Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, trust_remote_code=False, use_safetensors=True).to(self.device).eval()

    def score(self, query: str, descriptions: list[str], batch_size: int | None = None) -> np.ndarray:
        import torch

        if not descriptions:
            return np.empty(0, dtype=np.float32)
        batch_size = batch_size or (64 if self.device.type == "cuda" else 8)
        parts = []
        with torch.inference_mode():
            for start in range(0, len(descriptions), batch_size):
                chunk = descriptions[start:start + batch_size]
                tokens = self.tokenizer([query] * len(chunk), chunk, padding=True,
                                        truncation=True, max_length=MAX_LENGTH,
                                        return_tensors="pt").to(self.device)
                parts.append(self.model(**tokens).logits[:, 0].float().cpu().numpy())
        return np.concatenate(parts).astype(np.float32)


def _train_cross(groups: list[tuple[str, list[str], list[str]]], cache_dir: Path,
                 output: Path, epochs: int = 2) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL, revision=SAPBERT_REVISION,
                                               cache_dir=str(cache_dir), trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        SAPBERT_MODEL, revision=SAPBERT_REVISION, cache_dir=str(cache_dir),
        num_labels=1, ignore_mismatched_sizes=True, trust_remote_code=False,
        use_safetensors=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    batch_groups = 4 if torch.cuda.is_available() else 1
    steps = epochs * ((len(groups) + batch_groups - 1) // batch_groups)
    schedule = get_linear_schedule_with_warmup(optimizer, max(1, steps // 10), steps)
    for epoch in range(epochs):
        order = np.random.default_rng(SEED + epoch).permutation(len(groups))
        total = 0.0
        for step, start in enumerate(range(0, len(order), batch_groups), 1):
            indices = order[start:start + batch_groups]
            batch = [groups[int(i)] for i in indices]
            widths = [len(descriptions) for _, descriptions, _ in batch]
            queries = [query for query, descriptions, _ in batch for _ in descriptions]
            descriptions = [desc for _, choices, _ in batch for desc in choices]
            tokens = tokenizer(queries, descriptions, padding=True, truncation=True,
                               max_length=MAX_LENGTH, return_tensors="pt").to(model.device)
            logits = model(**tokens).logits[:, 0]
            losses = []
            offset = 0
            for width in widths:
                losses.append(torch.nn.functional.cross_entropy(
                    logits[offset:offset + width].unsqueeze(0),
                    torch.zeros(1, dtype=torch.long, device=model.device)))
                offset += width
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss.detach()) * len(batch)
            if step % 150 == 0:
                print(f"Cross-encoder epoch {epoch + 1}/{epochs}: {start + len(batch)}/{len(groups)}", flush=True)
        print(f"Cross-encoder epoch {epoch + 1}/{epochs}: loss={total / len(groups):.4f}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)


def train_learned(examples: list[Example], mapper: TermMapper, db: Terminology,
                  neural_dir: str | Path, output_dir: str | Path,
                  adapter_epochs: int = 5, cross_epochs: int = 2) -> Path:
    """Fit only on supplied training rows; reuse iff source and split match."""
    output_dir = Path(output_dir)
    expected = {"training_sha256": _fingerprint(examples, db), "base_model": SAPBERT_MODEL,
                "base_revision": SAPBERT_REVISION, "seed": SEED,
                "negatives": NEGATIVES, "max_length": MAX_LENGTH,
                "adapter_epochs": adapter_epochs, "cross_epochs": cross_epochs,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest = output_dir / "manifest.json"
    if (manifest.exists() and (output_dir / "adapter.npz").exists()
            and (output_dir / "cross" / "model.safetensors").exists()
            and json.loads(manifest.read_text(encoding="utf-8")) == expected):
        return output_dir
    _seed()
    dense = DenseIndex(neural_dir)
    groups, vectors = _mine_groups(examples, mapper, db, dense)
    if not groups:
        raise ValueError("No valid training groups for learned neural models")
    print(f"Training on {len(groups):,} mined groups", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _train_adapter(groups, vectors, dense, output_dir / "adapter.npz", epochs=adapter_epochs)
    del dense
    if __import__("torch").cuda.is_available():
        __import__("torch").cuda.empty_cache()
    _train_cross(groups, Path(neural_dir) / "model-cache", output_dir / "cross",
                 epochs=cross_epochs)
    manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return output_dir
