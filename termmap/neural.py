"""Local biomedical dense retrieval and ontology cross-encoder reranking.

Only model weights are downloaded. Clinical text is encoded on this machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .data import SYSTEMS, normalize
from .model import Candidate, TermMapper, expanded, local_context
from .terminology import Terminology

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAPBERT_REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
BOND_MODEL = "AronowLab/BOND-reranker"
BOND_REVISION = "7d77e45dbcec25043f7a01e6d3d689284f6c459d"
MAX_DENSE_DESCRIPTIONS_PER_CODE = 2


def _torch_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SapBERTEncoder:
    """SapBERT's documented CLS embedding with unit normalization."""

    def __init__(self, cache_dir: str | Path):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.device = _torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            SAPBERT_MODEL, revision=SAPBERT_REVISION, cache_dir=str(cache_dir), trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            SAPBERT_MODEL, revision=SAPBERT_REVISION, cache_dir=str(cache_dir),
            trust_remote_code=False, use_safetensors=True).to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        import torch

        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        batch_size = batch_size or (256 if self.device.type == "cuda" else 32)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(texts[start:start + batch_size], padding=True,
                                        truncation=True, max_length=64,
                                        return_tensors="pt").to(self.device)
                cls = self.model(**inputs).last_hidden_state[:, 0, :]
                cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
                chunks.append(cls.cpu().numpy())
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


class BiomedicalCrossEncoder:
    """BOND's local query-candidate relevance score."""

    def __init__(self, cache_dir: str | Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(42)
        self.device = _torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            BOND_MODEL, revision=BOND_REVISION, cache_dir=str(cache_dir), trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BOND_MODEL, revision=BOND_REVISION, cache_dir=str(cache_dir),
            trust_remote_code=False, use_safetensors=True).to(self.device).eval()

    def score(self, query: str, descriptions: list[str], batch_size: int | None = None) -> np.ndarray:
        import torch

        if not descriptions:
            return np.empty(0, dtype=np.float32)
        batch_size = batch_size or (48 if self.device.type == "cuda" else 8)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(descriptions), batch_size):
                part = descriptions[start:start + batch_size]
                inputs = self.tokenizer([query] * len(part), part, padding=True,
                                        truncation=True, max_length=160,
                                        return_tensors="pt").to(self.device)
                logits = self.model(**inputs).logits.float()
                chunks.append(logits[:, -1].cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)


def build_dense_index(terminology_path: str | Path, output_dir: str | Path,
                      limit_codes: int | None = None, batch_size: int = 256) -> Path:
    """Build deterministic FAISS HNSW indexes over legal code descriptions."""
    import faiss

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Terminology(terminology_path) as db:
        source_signatures = db.source_signatures()
    expected = {"terminology_sources": source_signatures, "encoder": SAPBERT_MODEL,
                "encoder_revision": SAPBERT_REVISION, "index": "HNSW32-IP",
                "description_selection": "first_distinct_normalized",
                "max_descriptions_per_code": MAX_DENSE_DESCRIPTIONS_PER_CODE,
                "limit_codes": limit_codes}
    manifest = output_dir / "manifest.json"
    if manifest.exists() and all((output_dir / f"{system}.faiss").exists() and
                                 (output_dir / f"{system}.codes.json").exists() for system in SYSTEMS):
        if json.loads(manifest.read_text(encoding="utf-8")) == expected:
            return output_dir

    encoder = SapBERTEncoder(output_dir / "model-cache")
    faiss.omp_set_num_threads(1)
    with Terminology(terminology_path) as db:
        for system in SYSTEMS:
            codes: list[str] = []
            descriptions: list[str] = []
            index = faiss.IndexHNSWFlat(encoder.model.config.hidden_size, 32,
                                       faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 128
            def add_batch() -> None:
                if not descriptions:
                    return
                vectors = encoder.encode(descriptions, batch_size=batch_size)
                index.add(vectors)
                descriptions.clear()

            for code, description in db.iter_dense_descriptions(
                    system, max_per_code=MAX_DENSE_DESCRIPTIONS_PER_CODE,
                    limit_codes=limit_codes):
                codes.append(code)
                descriptions.append(description)
                if len(descriptions) >= batch_size:
                    add_batch()
                    if len(codes) % 10_000 < batch_size:
                        print(f"Dense {system}: {len(codes):,} descriptions", flush=True)
            add_batch()
            index_path = output_dir / f"{system}.faiss"
            codes_path = output_dir / f"{system}.codes.json"
            faiss.write_index(index, str(index_path.with_suffix(".faiss.partial")))
            codes_path.with_suffix(".json.partial").write_text(
                json.dumps(codes, separators=(",", ":")), encoding="utf-8")
            os.replace(index_path.with_suffix(".faiss.partial"), index_path)
            os.replace(codes_path.with_suffix(".json.partial"), codes_path)
            print(f"Built dense {system} index with {len(codes):,} descriptions "
                  f"for {len(set(codes)):,} legal codes", flush=True)
    manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return output_dir


class DenseIndex:
    def __init__(self, output_dir: str | Path):
        import faiss

        output_dir = Path(output_dir)
        self.encoder = SapBERTEncoder(output_dir / "model-cache")
        faiss.omp_set_num_threads(1)
        self.indices = {system: faiss.read_index(str(output_dir / f"{system}.faiss"))
                        for system in SYSTEMS}
        self.codes = {system: json.loads((output_dir / f"{system}.codes.json").read_text(encoding="utf-8"))
                      for system in SYSTEMS}
        for index in self.indices.values():
            index.hnsw.efSearch = 128

    def encode(self, entity: str) -> np.ndarray:
        return self.encoder.encode([expanded(entity)])

    def search_vector(self, system: str, vector: np.ndarray, limit: int = 70) -> dict[str, float]:
        index = self.indices[system]
        scores, positions = index.search(vector, min(index.ntotal, limit * 3))
        return _unique_code_hits(self.codes[system], scores[0], positions[0], limit)


def _unique_code_hits(codes: list[str], scores: np.ndarray,
                      positions: np.ndarray, limit: int) -> dict[str, float]:
    """Collapse multiple matching descriptions to their best score per code."""
    best: dict[str, float] = {}
    for score, position in zip(scores, positions, strict=True):
        if position < 0:
            continue
        code = codes[int(position)]
        best[code] = max(best.get(code, float("-inf")), float(score))
    return dict(sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit])


class TrainingAliasIndex:
    """Small exact SapBERT index built only from this mapper's fitting rows."""

    def __init__(self, local_mapper: TermMapper, encoder: SapBERTEncoder):
        import faiss

        self.indices = {}
        self.codes = {}
        for system in SYSTEMS:
            pairs = sorted({(expanded(row.entity), row.code)
                            for row in local_mapper.models[system].examples
                            if expanded(row.entity)})
            self.codes[system] = [code for _, code in pairs]
            index = faiss.IndexFlatIP(encoder.model.config.hidden_size)
            if pairs:
                vectors = encoder.encode([alias for alias, _ in pairs])
                index.add(vectors)
            self.indices[system] = index

    def search_vector(self, system: str, vector: np.ndarray,
                      limit: int = 30) -> dict[str, float]:
        index = self.indices[system]
        if not index.ntotal:
            return {}
        scores, positions = index.search(vector, min(index.ntotal, limit * 8))
        return _unique_code_hits(self.codes[system], scores[0], positions[0], limit)


def _diversified_shortlist(anchor: str, local_codes: list[str],
                           fused_codes: list[str], dense_codes: list[str],
                           fallback_codes: list[str], limit: int) -> list[str]:
    """Keep the local decision while exposing candidates from each retrieval path.

    The fixed 2:1:1 schedule is for the adjudicator's candidate list, not its
    final decision. Each source advances past codes already selected, so an
    overlapping result does not consume a shortlist slot.
    """
    ordered = [anchor]
    seen = {anchor}
    sources = {"local": local_codes, "fused": fused_codes, "dense": dense_codes}
    schedule = ("local", "local", "fused", "dense")
    positions = {name: 0 for name in sources}
    while len(ordered) < limit:
        added = False
        for name in schedule:
            source = sources[name]
            while positions[name] < len(source) and source[positions[name]] in seen:
                positions[name] += 1
            if positions[name] >= len(source):
                continue
            code = source[positions[name]]
            positions[name] += 1
            ordered.append(code)
            seen.add(code)
            added = True
            if len(ordered) == limit:
                break
        if not added:
            break
    for code in fallback_codes:
        if len(ordered) == limit:
            break
        if code not in seen:
            ordered.append(code)
            seen.add(code)
    return ordered


class NeuralMapper:
    """Fuse local ranking, SapBERT retrieval, optional hierarchy and reranking."""

    def __init__(self, local_mapper: TermMapper, neural_dir: str | Path,
                 learned_dir: str | Path | None = None):
        self.local_mapper = local_mapper
        self.dense = DenseIndex(neural_dir)
        self.learned_dir = Path(learned_dir) if learned_dir else None
        self.adapter = None
        if self.learned_dir is not None:
            from .learned import QueryAdapter, TaskCrossEncoder
            self.adapter = QueryAdapter(self.learned_dir / "adapter.npz")
            self.cross = TaskCrossEncoder(self.learned_dir / "cross")
        else:
            self.cross = BiomedicalCrossEncoder(Path(neural_dir) / "model-cache")
        self.training_aliases = TrainingAliasIndex(local_mapper, self.dense.encoder)

    def rank_system(self, system: str, entity: str, context: str, terminology: Terminology,
                    vector: np.ndarray, top_k: int = 5) -> tuple[list[dict[str, object]], set[str]]:
        terminology_hits = self.dense.search_vector(system, vector)
        adapter = getattr(self, "adapter", None)
        learned_dir = getattr(self, "learned_dir", None)
        if adapter is not None:
            adapted_hits = self.dense.search_vector(system, adapter.transform(vector), limit=70)
            for code, score in adapted_hits.items():
                terminology_hits[code] = max(terminology_hits.get(code, float("-inf")), score)
        training_aliases = getattr(self, "training_aliases", None)
        alias_hits = (training_aliases.search_vector(system, vector)
                      if training_aliases is not None else {})
        combined_hits = dict(terminology_hits)
        for code, score in alias_hits.items():
            combined_hits[code] = max(combined_hits.get(code, float("-inf")), score)
        dense_hits = dict(sorted(combined_hits.items(),
                                 key=lambda item: (-item[1], item[0]))[:100])
        neighbors = (terminology.snomed_neighbors(list(dense_hits)[:15])
                     if system == "SNOMEDCT_US" else {})
        hierarchy_codes = [code for group in neighbors.values() for code in group]
        local_model = self.local_mapper.models[system]
        # The supervised ranker was fitted without dense candidates. Keep its
        # original decision as an anchor instead of treating every BOND score
        # as calibrated to the supplied ICD/SNOMED answer space.
        base_candidates = local_model.candidates(entity, context, terminology)
        base_ranking = local_model.rank_candidates(base_candidates, top_k=max(top_k, 2))
        base_top = str(base_ranking[0]["code"])
        base_margin = float(base_ranking[0]["score"] - base_ranking[1]["score"])
        # A strong local decision can skip neural scoring for ordinary local
        # predictions. The adjudicator requests 25 candidates, so it still
        # needs the dense and reranked alternatives in its shortlist.
        if base_margin > 3.0 and top_k <= 5 and learned_dir is None:
            ranking = [dict(row, score=1.0 / (rank + 1), local_score=row["score"],
                            decision_source="local")
                       for rank, row in enumerate(base_ranking[:top_k])]
            return ranking, {candidate.code for candidate in base_candidates}
        candidates = local_model.candidates(entity, context, terminology,
                                            force_code=base_top, dense_hits=dense_hits,
                                            extra_codes=hierarchy_codes,
                                            max_candidates=180)
        local_ranking = local_model.rank_candidates(candidates, top_k=len(candidates))
        local_rank = {row["code"]: i + 1 for i, row in enumerate(local_ranking)}
        dense_rank = {code: i + 1 for i, code in enumerate(dense_hits)}
        candidate_by_code = {candidate.code: candidate for candidate in candidates}
        pool_limit = 75 if learned_dir is not None else 60
        pool_codes = list(dict.fromkeys(
            [base_top] + [row["code"] for row in local_ranking[:pool_limit]] +
            [code for code in list(dense_hits)[:pool_limit] if code in candidate_by_code] +
            [code for code in hierarchy_codes[:15] if code in candidate_by_code]))
        pool: list[Candidate] = [candidate_by_code[code] for code in pool_codes]
        nearby = normalize(local_context(entity, context, radius=100)[0])[:240]
        if learned_dir is not None:
            from .learned import _query
            query = _query(system, entity, context)
        else:
            query = expanded(entity) + (". Context: " + nearby if nearby and context else "")
        cross_scores = self.cross.score(query, [c.matched_description or c.code for c in pool])
        cross_order = sorted(range(len(pool)), key=lambda i: (-float(cross_scores[i]), pool[i].code))
        cross_rank = {pool[i].code: rank + 1 for rank, i in enumerate(cross_order)}
        scored = []
        for i, candidate in enumerate(pool):
            code = candidate.code
            fused = ((1.0 if learned_dir is not None else 2.0) / (20 + local_rank[code])
                     + (4.0 if learned_dir is not None else 2.0) / (20 + cross_rank[code]))
            if code in dense_rank:
                fused += 1.0 / (20 + dense_rank[code])
            scored.append((candidate, fused, float(cross_scores[i])))
        scored.sort(key=lambda x: (-x[1], -x[0].baseline, x[0].code))
        # A high SapBERT cosine similarity plus BOND agreement is useful for
        # rescuing a weak local decision. The fixed gate was selected on the
        # development split; it protects strong lexical matches from a cross
        # encoder trained on a different ontology ranking distribution.
        proposed = scored[0]
        base_cross_score = float(cross_scores[pool_codes.index(base_top)])
        accept_neural = (proposed[0].code != base_top if learned_dir is not None else
                         (proposed[0].code != base_top and base_margin <= 3.0
                          and proposed[0].dense_score >= 0.90
                          and proposed[2] >= base_cross_score))
        fused_by_code = {candidate.code: (score, cross_score)
                         for candidate, score, cross_score in scored}
        all_candidates = {candidate.code: candidate for candidate in base_candidates}
        all_candidates.update(candidate_by_code)
        base_scores = {str(row["code"]): float(row["score"]) for row in base_ranking}
        decision = proposed[0].code if accept_neural else base_top
        if top_k > 5:
            ordered = _diversified_shortlist(
                decision,
                [str(row["code"]) for row in base_ranking],
                [candidate.code for candidate, _, _ in scored],
                [code for code in dense_hits if code in candidate_by_code],
                [str(row["code"]) for row in local_ranking],
                top_k,
            )
        else:
            ordered = ([decision] if accept_neural else [])
            ordered += [str(row["code"]) for row in base_ranking]
            ordered += [candidate.code for candidate, _, _ in scored]
        ranking = []
        for code in list(dict.fromkeys(ordered))[:top_k]:
            candidate = all_candidates[code]
            fused_score, cross_score = fused_by_code.get(code, (None, None))
            ranking.append({"code": code, "score": 1.0 / (len(ranking) + 1),
                            "description": candidate.matched_description,
                            "baseline": candidate.baseline,
                            "local_score": base_scores.get(code), "fusion_score": fused_score,
                            "dense_score": candidate.dense_score, "cross_score": cross_score,
                            "decision_source": ("neural" if accept_neural and not ranking
                                                else "local" if code in base_scores else "retrieval")})
        return ranking, {c.code for c in base_candidates} | {c.code for c in candidates}

    def predict_with_db(self, entity: str, context: str, terminology: Terminology,
                        top_k: int = 5) -> tuple[dict[str, str], dict[str, list[dict[str, object]]]]:
        vector = self.dense.encode(entity)
        details = {system: self.rank_system(system, entity, context, terminology, vector,
                                            top_k=top_k)[0] for system in SYSTEMS}
        return {system: str(details[system][0]["code"]) for system in SYSTEMS}, details
