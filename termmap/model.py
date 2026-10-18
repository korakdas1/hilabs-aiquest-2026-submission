"""Hybrid retrieval and supervised ranking for each coding system."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Example, SYSTEMS, normalize
from .terminology import Terminology

ABBREVIATIONS = {
    "htn": "hypertension", "t2dm": "type 2 diabetes mellitus",
    "dm": "diabetes mellitus", "dmii": "type 2 diabetes mellitus",
    "dm2": "type 2 diabetes mellitus", "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease", "chf": "congestive heart failure",
    "cad": "coronary artery disease", "gerd": "gastroesophageal reflux disease",
    "uti": "urinary tract infection", "pna": "pneumonia", "sob": "shortness of breath",
    "afib": "atrial fibrillation", "osa": "obstructive sleep apnea",
    "aki": "acute kidney injury", "cva": "cerebrovascular accident",
    "hhs": "hyperosmolar hyperglycemic syndrome", "cki": "chronic kidney insufficiency",
    "rle": "right lower extremity", "lle": "left lower extremity",
}
QUALIFIERS = ("left", "right", "bilateral", "acute", "chronic", "family", "history",
              "personal", "initial", "subsequent", "unspecified", "malignant", "benign",
              "secondary", "primary", "severe", "mild", "recurrent", "traumatic")
OPPOSITES = (("left", "right"), ("acute", "chronic"), ("benign", "malignant"),
             ("initial", "subsequent"), ("primary", "secondary"))
SEMANTIC_TAGS = ("disorder", "finding", "situation", "procedure", "observable entity",
                 "body structure", "substance", "qualifier value", "event")


def expanded(text: str) -> str:
    words = normalize(text).split()
    return " ".join(ABBREVIATIONS.get(word, word) for word in words)


def local_context(entity: str, context: str, radius: int = 120) -> tuple[str, str, str]:
    """Return nearby text, preceding text and following text around the mention."""
    index = context.casefold().find(entity.casefold()) if entity else -1
    if index < 0:
        return context[: radius * 2], "", ""
    before = context[max(0, index - radius):index]
    after = context[index + len(entity):index + len(entity) + radius]
    return before + " " + entity + " " + after, before, after


def _top_indices(values: np.ndarray, count: int, exclude: int | None = None) -> list[int]:
    if exclude is not None and exclude < len(values):
        values = values.copy()
        values[exclude] = -1.0
    count = min(count, len(values))
    if not count:
        return []
    indices = np.argpartition(values, -count)[-count:]
    return sorted((int(i) for i in indices if values[i] > 0), key=lambda i: (-values[i], i))


def _token_features(query: str, descriptions: list[str]) -> tuple[float, float, float, float, float, float, str]:
    q = normalize(query)
    q_tokens = set(q.split())
    best_ratio = best_set = best_sort = best_overlap = best_exact = best_containment = 0.0
    best_desc = ""
    for desc in descriptions:
        d = normalize(desc)
        if not d:
            continue
        ratio = fuzz.WRatio(q, d) / 100.0
        token_set = fuzz.token_set_ratio(q, d) / 100.0
        token_sort = fuzz.token_sort_ratio(q, d) / 100.0
        d_tokens = set(d.split())
        overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens | d_tokens))
        exact = float(q == d)
        contained = float(q in d or d in q)
        if ratio > best_ratio:
            best_ratio, best_desc = ratio, desc
        best_set = max(best_set, token_set)
        best_sort = max(best_sort, token_sort)
        best_overlap = max(best_overlap, overlap)
        best_exact = max(best_exact, exact)
        best_containment = max(best_containment, contained)
    return best_ratio, best_set, best_sort, best_overlap, best_exact, best_containment, best_desc


@dataclass
class Candidate:
    code: str
    features: list[float]
    baseline: float
    matched_description: str
    dense_score: float = 0.0


class SystemModel:
    def __init__(self, system: str, examples: list[Example]):
        if system not in SYSTEMS:
            raise ValueError(f"Unknown coding system: {system}")
        self.system = system
        self.examples = [x for x in examples if x.system == system]
        self.codes = [x.code for x in self.examples]
        self.code_counts = Counter(self.codes)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, dtype=np.float32)
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, dtype=np.float32)
        self.context_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=40000, stop_words="english", sublinear_tf=True, dtype=np.float32)
        self.near_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words="english", sublinear_tf=True, dtype=np.float32)
        entities = [expanded(x.entity) for x in self.examples]
        contexts = [normalize(x.context[:4000]) or "emptycontext" for x in self.examples]
        near_contexts = [normalize(local_context(x.entity, x.context)[0]) or "emptycontext" for x in self.examples]
        self.char_matrix = self.char_vectorizer.fit_transform(entities)
        self.word_matrix = self.word_vectorizer.fit_transform(entities)
        self.context_matrix = self.context_vectorizer.fit_transform(contexts)
        self.near_matrix = self.near_vectorizer.fit_transform(near_contexts)
        self.ranker: lgb.LGBMRanker | None = None

    def _similarities(self, entity: str, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        query = expanded(entity)
        c = (self.char_matrix @ self.char_vectorizer.transform([query]).T).toarray().ravel()
        w = (self.word_matrix @ self.word_vectorizer.transform([query]).T).toarray().ravel()
        context_norm = normalize(context[:4000]) or "emptycontext"
        x = (self.context_matrix @ self.context_vectorizer.transform([context_norm]).T).toarray().ravel()
        near_norm = normalize(local_context(entity, context)[0]) or "emptycontext"
        n = (self.near_matrix @ self.near_vectorizer.transform([near_norm]).T).toarray().ravel()
        return c, w, x, n

    def candidates(self, entity: str, context: str, terminology: Terminology,
                   exclude_example: int | None = None, force_code: str | None = None,
                   max_candidates: int = 120,
                   dense_hits: dict[str, float] | None = None,
                   extra_codes: list[str] | None = None) -> list[Candidate]:
        char_sim, word_sim, context_sim, near_sim = self._similarities(entity, context)
        selected: dict[str, dict[str, float | str]] = {}
        def new_item() -> dict[str, float | str]:
            return {"char": 0.0, "word": 0.0, "context": 0.0, "near": 0.0,
                    "train_rank": 0.0, "fts": 0.0, "fts_desc": "", "dense": 0.0}
        for source, values, k in (("char", char_sim, 50), ("word", word_sim, 35),
                                  ("context", context_sim, 15), ("near", near_sim, 20)):
            for rank, i in enumerate(_top_indices(values, k, exclude_example)):
                code = self.codes[i]
                item = selected.setdefault(code, new_item())
                item[source] = max(float(item[source]), float(values[i]))
                item["train_rank"] = max(float(item["train_rank"]), 1.0 / (1.0 + rank / 10.0))
        base_query = expanded(entity)
        _, preceding, _ = local_context(entity, context)
        preceding = normalize(preceding[-80:])
        queries = [(base_query, 1.0)]
        if "family history" in preceding or "fhx" in preceding:
            queries.append(("family history of " + base_query, 0.95))
        elif "history of" in preceding or "hx of" in preceding:
            queries.append(("history of " + base_query, 0.9))
        if any(word in preceding.split()[-6:] for word in ("no", "denies")):
            queries.append(("no " + base_query, 0.85))
        for search_query, query_weight in queries:
            for code, (score, desc) in terminology.search(self.system, search_query, limit=100).items():
                score *= query_weight
                item = selected.setdefault(code, new_item())
                if score > float(item["fts"]):
                    item["fts"] = score
                    item["fts_desc"] = desc
        if force_code and terminology.contains(self.system, force_code):
            selected.setdefault(force_code, new_item())
        if dense_hits:
            for code, score in dense_hits.items():
                if terminology.contains(self.system, code):
                    selected.setdefault(code, new_item())["dense"] = float(score)
        if extra_codes:
            for code in extra_codes:
                if terminology.contains(self.system, code):
                    selected.setdefault(code, new_item())

        def retrieval_priority(pair: tuple[str, dict[str, float | str]]) -> float:
            _, values = pair
            return max(float(values["char"]) * 0.9, float(values["word"]) * 0.9,
                       float(values["fts"]), float(values["context"]) * 0.35,
                       float(values["near"]) * 0.45, float(values["dense"]) * 0.9)

        selected = dict(sorted(selected.items(), key=lambda pair: (-retrieval_priority(pair), pair[0]))[:max_candidates])
        if extra_codes:
            for code in extra_codes[:40]:
                if code not in selected and terminology.contains(self.system, code):
                    selected[code] = new_item()
        if force_code and terminology.contains(self.system, force_code) and force_code not in selected:
            selected[force_code] = new_item()
        descriptions = terminology.descriptions(self.system, selected)
        query = expanded(entity)
        query_tokens = set(query.split())
        context_tokens = set(normalize(context[:4000]).split())
        near, before, after = local_context(entity, context)
        near_tokens = set(normalize(near).split())
        before_norm = normalize(before[-80:])
        after_norm = normalize(after[:80])
        family_history = float("family history" in before_norm or "fhx" in before_norm)
        personal_history = float("history of" in before_norm or "hx of" in before_norm or "h o" in before_norm)
        negation = float(any(x in before_norm.split()[-8:] for x in ("no", "denies", "without")) or
                         "negative for" in before_norm)
        resolved = float("resolved" in after_norm or "resolved" in before_norm)
        output: list[Candidate] = []
        for code, values in selected.items():
            aliases = descriptions.get(code, [])
            if values["fts_desc"] and values["fts_desc"] not in aliases:
                aliases.append(str(values["fts_desc"]))
            if not aliases:
                aliases = [code]
            ratio, token_set, token_sort, overlap, exact, contains, best_desc = _token_features(query, aliases)
            alias_tokens = set(normalize(" ".join(aliases)).split())
            qualifier_match = sum(q in query_tokens and q in alias_tokens for q in QUALIFIERS)
            qualifier_missing = sum(q in query_tokens and q not in alias_tokens for q in QUALIFIERS)
            qualifier_extra = sum(q not in query_tokens and q in alias_tokens for q in QUALIFIERS)
            contradiction = sum((a in query_tokens and b in alias_tokens) or
                                (b in query_tokens and a in alias_tokens) for a, b in OPPOSITES)
            context_qual_match = sum(q in context_tokens and q in alias_tokens for q in ("family", "history", "acute", "chronic", "left", "right"))
            near_qual_match = sum(q in near_tokens and q in alias_tokens for q in QUALIFIERS)
            # Tags appear in source descriptions with parentheses. Feature values are learned, never used as a hard filter.
            raw_alias = " ".join(aliases).casefold()
            semantic = [float(f"({tag})" in raw_alias) for tag in SEMANTIC_TAGS]
            code_length = len(code.replace(".", "")) if self.system == "ICD10CM" else len(code)
            frequency = math.log1p(self.code_counts.get(code, 0))
            features = [float(values["char"]), float(values["word"]), float(values["context"]),
                        float(values["near"]), float(values["train_rank"]), float(values["fts"]), ratio, token_set,
                        token_sort, overlap, exact, contains, frequency,
                        float(qualifier_match), float(qualifier_missing), float(qualifier_extra),
                        float(contradiction), float(context_qual_match), float(near_qual_match),
                        float("unspecified" in alias_tokens), float("history" in alias_tokens),
                        float("no" in alias_tokens or "without" in alias_tokens),
                        float("history" in context_tokens), float("family" in context_tokens),
                        float("no" in context_tokens or "denies" in context_tokens),
                        float(len(query_tokens)), float(len(normalize(best_desc).split())),
                        float(code_length), family_history, personal_history, negation, resolved,
                        family_history * float("family" in alias_tokens),
                        personal_history * float("history" in alias_tokens),
                        negation * float("no" in alias_tokens or "without" in alias_tokens),
                        *semantic]
            baseline = (1.2 * exact + 0.8 * float(values["char"]) + 0.5 * float(values["word"])
                        + 0.7 * float(values["fts"]) + 0.6 * ratio + 0.3 * overlap
                        + 0.02 * frequency - 0.25 * contradiction
                        + 0.4 * float(values["dense"]))
            output.append(Candidate(code, features, baseline, best_desc, float(values["dense"])))
        return sorted(output, key=lambda c: (-c.baseline, c.code))

    def fit_ranker(self, terminology: Terminology, verbose: bool = True) -> dict[str, float]:
        feature_rows: list[list[float]] = []
        labels: list[int] = []
        groups: list[int] = []
        retrieved = 0
        for i, example in enumerate(self.examples):
            candidates = self.candidates(example.entity, example.context, terminology,
                                         exclude_example=i)
            if any(c.code == example.code for c in candidates):
                retrieved += 1
            else:
                candidates = self.candidates(example.entity, example.context, terminology,
                                             exclude_example=i, force_code=example.code)
            # Keep challenging negatives and a positive, using source retrieval order.
            candidates = candidates[:35] + [c for c in candidates[35:] if c.code == example.code]
            if len(candidates) < 2:
                continue
            for candidate in candidates:
                features = candidate.features.copy()
                if candidate.code == example.code:
                    # The query itself is not available at inference. A singleton
                    # gold code must therefore look like an unseen code during fit.
                    features[12] = math.log1p(max(0, self.code_counts[candidate.code] - 1))
                feature_rows.append(features)
            labels.extend(int(c.code == example.code) for c in candidates)
            groups.append(len(candidates))
            if verbose and (i + 1) % 500 == 0:
                print(f"{self.system}: built training candidates {i + 1}/{len(self.examples)}", flush=True)
        model = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=180,
                               learning_rate=0.05, num_leaves=15, min_child_samples=30,
                               max_bin=127, random_state=42, deterministic=True,
                               force_col_wise=True, n_jobs=8, verbosity=-1)
        model.fit(np.asarray(feature_rows, dtype=np.float32), np.asarray(labels, dtype=np.int8), group=groups)
        self.ranker = model
        return {"training_queries": len(groups), "training_candidate_recall": retrieved / max(1, len(self.examples))}

    def rank(self, entity: str, context: str, terminology: Terminology, top_k: int = 5) -> list[dict[str, object]]:
        candidates = self.candidates(entity, context, terminology)
        return self.rank_candidates(candidates, top_k)

    def rank_candidates(self, candidates: list[Candidate], top_k: int = 5) -> list[dict[str, object]]:
        if not candidates:
            raise RuntimeError(f"No {self.system} candidates")
        if self.ranker is None:
            scores = [c.baseline for c in candidates]
        else:
            raw = self.ranker.predict(np.asarray([c.features for c in candidates], dtype=np.float32))
            # The lexical score protects rare concepts from the learned model's
            # prior for frequent training codes. This weight was chosen on a
            # development split and is fixed for subsequent evaluation.
            scores = [float(score) + 2.5 * c.baseline for c, score in zip(candidates, raw, strict=True)]
        ranked = sorted(zip(candidates, scores), key=lambda p: (-float(p[1]), -p[0].baseline, p[0].code))
        return [{"code": c.code, "score": float(score), "description": c.matched_description,
                 "baseline": c.baseline} for c, score in ranked[:top_k]]


class TermMapper:
    def __init__(self, models: dict[str, SystemModel], terminology_path: str | Path):
        self.models = models
        self.terminology_path = Path(terminology_path)

    @classmethod
    def train(cls, examples: list[Example], terminology_path: str | Path,
              output: str | Path | None = None, verbose: bool = True) -> tuple["TermMapper", dict[str, object]]:
        with Terminology(terminology_path) as terminology:
            valid = [x for x in examples if terminology.contains(x.system, x.code)]
            rejected = [x.code for x in examples if not terminology.contains(x.system, x.code)]
            models = {}
            metrics: dict[str, object] = {"excluded_invalid_training_codes": sorted(set(rejected))}
            for system in SYSTEMS:
                model = SystemModel(system, valid)
                metrics[system] = model.fit_ranker(terminology, verbose=verbose)
                models[system] = model
        mapper = cls(models, terminology_path)
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"models": models, "terminology_path": str(terminology_path)}, path, compress=3)
        return mapper, metrics

    @classmethod
    def load(cls, path: str | Path, terminology_path: str | Path | None = None) -> "TermMapper":
        data = joblib.load(path)
        return cls(data["models"], terminology_path or data["terminology_path"])

    def predict(self, entity: str, context: str = "", top_k: int = 5) -> tuple[dict[str, str], dict[str, list[dict[str, object]]]]:
        with Terminology(self.terminology_path) as terminology:
            return self.predict_with_db(entity, context, terminology, top_k)

    def predict_with_db(self, entity: str, context: str, terminology: Terminology,
                        top_k: int = 5) -> tuple[dict[str, str], dict[str, list[dict[str, object]]]]:
        details = {system: self.models[system].rank(entity, context, terminology, top_k=top_k)
                   for system in SYSTEMS}
        return {system: str(details[system][0]["code"]) for system in SYSTEMS}, details
