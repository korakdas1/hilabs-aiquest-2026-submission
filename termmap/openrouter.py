"""Opt-in zero-retention LLM adjudication. Never called by the local pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from .data import SYSTEMS
from .model import local_context
from .terminology import Terminology

URL = "https://openrouter.ai/api/v1/chat/completions"

LATERALITY_PAIRS = (("right", "left"), ("left", "right"))
ACUITY_PAIRS = (("acute", "chronic"), ("chronic", "acute"))

# Minimum top-1 vs top-2 baseline margin (lexical retrieval score) at which the
# local prediction is considered strong enough to override hosted adjudication.
# The threshold was chosen on the seed-42 holdout: it keeps near-exact matches
# where the API currently regresses more often than it helps.
CONFIDENCE_MARGIN = 0.75


def _baseline_margin(items: list[dict[str, object]]) -> float:
    if len(items) < 2:
        return float("inf")
    try:
        top = float(items[0].get("baseline") or 0.0)
        second = float(items[1].get("baseline") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return top - second


def _filter_contradictions(entity_norm: str, candidates: list[dict[str, object]],
                           pairs: tuple[tuple[str, str], ...]) -> list[dict[str, object]]:
    """Drop candidates whose description contradicts an explicit entity qualifier.

    Only applies when the entity contains the qualifier as a standalone token; the
    anchor (local prediction) is always kept so a wrong filter cannot delete every
    option. Preserves order.
    """
    from .data import normalize
    tokens = set(entity_norm.split())
    for mine, opposite in pairs:
        if mine not in tokens:
            continue
        kept: list[dict[str, object]] = []
        for i, candidate in enumerate(candidates):
            desc_tokens = set(normalize(str(candidate.get("description", ""))).split())
            if i == 0:
                kept.append(candidate)
                continue
            if opposite in desc_tokens and mine not in desc_tokens:
                continue
            kept.append(candidate)
        candidates = kept
    return candidates


def _read_dotenv(path: str | Path = ".env") -> None:
    file = Path(path)
    if not file.exists():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"OPENROUTER_API_KEY", "OPENROUTER_MODEL", "OPENROUTER_ZERO_RETENTION_CONFIRMED"}:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


class OpenRouterAdjudicator:
    def __init__(self, cache_path: str | Path, dotenv_path: str | Path = ".env"):
        _read_dotenv(dotenv_path)
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        self.model = os.environ.get("OPENROUTER_MODEL", "")
        confirmed = os.environ.get("OPENROUTER_ZERO_RETENTION_CONFIRMED", "").lower()
        if confirmed not in {"1", "true", "yes"}:
            raise ValueError("OpenRouter requires OPENROUTER_ZERO_RETENTION_CONFIRMED=true")
        if not self.key or not self.model:
            raise ValueError("OpenRouter requires OPENROUTER_API_KEY and OPENROUTER_MODEL")
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {"remote_calls": 0, "cache_hits": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "cost_usd": 0.0, "gated_skips": 0}
        self.cache: dict[str, dict[str, str]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    self.cache[item["key"]] = item["prediction"]
                except (ValueError, KeyError, TypeError):
                    continue

    def adjudicate(self, entity: str, context: str, local_prediction: dict[str, str],
                   candidates: dict[str, list[dict[str, object]]], terminology: Terminology) -> dict[str, str]:
        from .data import normalize
        nearby = local_context(entity, context, radius=600)[0]
        entity_norm = normalize(entity)
        # Confidence gate: when the local lexical margin is wide, the local pick
        # is almost certainly the right one. Skipping the API in that case cuts
        # the regression rate (correct local answers that get flipped wrong).
        icd_confident = _baseline_margin(candidates["ICD10CM"]) >= CONFIDENCE_MARGIN
        snomed_confident = _baseline_margin(candidates["SNOMEDCT_US"]) >= CONFIDENCE_MARGIN
        if icd_confident and snomed_confident:
            self.stats["gated_skips"] += 1
            return dict(local_prediction)
        filtered_icd = _filter_contradictions(entity_norm, list(candidates["ICD10CM"][:50]), LATERALITY_PAIRS)
        filtered_icd = _filter_contradictions(entity_norm, filtered_icd, ACUITY_PAIRS)
        filtered_snomed = _filter_contradictions(entity_norm, list(candidates["SNOMEDCT_US"][:50]), LATERALITY_PAIRS)
        filtered_snomed = _filter_contradictions(entity_norm, filtered_snomed, ACUITY_PAIRS)
        gemini = self.model.startswith("google/gemini-")
        provider = {"zdr": True, "data_collection": "deny", "require_parameters": True}
        if self.model in {"qwen/qwen3-235b-a22b-2507", "deepseek/deepseek-v4.1-flash",
                          "deepseek/deepseek-v4.1-flash-1017"}:
            # Keep benchmark requests on DeepInfra's zero-retention endpoints.
            provider["only"] = ["deepinfra"]
        payload = {
            "model": self.model,
            "temperature": 0,
            "seed": 42,
            # Gemini reasoning tokens count toward the completion limit. Qwen
            # Instruct is a non-reasoning model and needs far fewer output tokens.
            "max_tokens": 1024 if gemini else 384,
            "response_format": {"type": "json_object"},
            "provider": provider,
            "messages": [
                {"role": "system", "content": (
                    "You are a clinical coding adjudicator. Map one extracted diagnosis or condition "
                    "to one ICD-10-CM code and one SNOMED CT concept, using only evidence in the "
                    "entity and nearby chart text. Follow these rules strictly:\n"
                    "1. Prefer the 'unspecified' variant when the text does not state a qualifier "
                    "(laterality, acuity, severity, encounter type, episode). Over-specifying is "
                    "an error.\n"
                    "2. Match laterality exactly. If the text says 'right', do NOT pick a 'left' "
                    "or 'bilateral' code. If no side is stated, use the unspecified-side code.\n"
                    "3. Match acuity. 'acute' and 'chronic' are opposites; never pick one when the "
                    "text states the other.\n"
                    "4. Honor negation and history. 'no X', 'denies X', 'without X', 'ruled out' "
                    "means the condition is absent — pick a Z-code or 'ruled out' observation code "
                    "rather than the diagnosis itself. 'history of', 'h/o', 'personal history' "
                    "means past; pick a Z-code personal-history category when available.\n"
                    "5. Family history ('fhx', 'family history of') means a Z-code family-history "
                    "concept, not the disease code.\n"
                    "6. Candidate lists are strong suggestions; you may choose any legal code if "
                    "certain, but prefer codes from the lists when they fit.\n"
                    "Return ONLY a JSON object with keys icd10cm_code and snomed_code (both strings).")},
                {"role": "user", "content": json.dumps({
                    "entity": entity, "nearby_chart_text": nearby,
                    "local_prediction": local_prediction,
                    "icd10cm_candidates": [{"code": x["code"], "description": x["description"]}
                                            for x in filtered_icd],
                    "snomed_candidates": [{"code": x["code"], "description": x["description"]}
                                          for x in filtered_snomed],
                }, ensure_ascii=False)},
            ],
        }
        if gemini:
            payload["reasoning"] = {"effort": "low"}
        request_data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        key = hashlib.sha256(request_data).hexdigest()
        if key in self.cache:
            cached = self.cache[key]
            if all(terminology.contains(system, cached.get(system, "")) for system in SYSTEMS):
                self.stats["cache_hits"] += 1
                return {
                    "ICD10CM": local_prediction["ICD10CM"] if icd_confident else cached["ICD10CM"],
                    "SNOMEDCT_US": local_prediction["SNOMEDCT_US"] if snomed_confident else cached["SNOMEDCT_US"],
                }
        request = urllib.request.Request(URL, data=request_data,
                                         headers={"Authorization": f"Bearer {self.key}",
                                                  "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.load(response)
        self.stats["remote_calls"] += 1
        usage = body.get("usage") or {}
        self.stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.stats["cost_usd"] += float(usage.get("cost") or 0)
        choice = body["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("OpenRouter completion truncated")
        message = choice["message"]["content"].strip()
        if message.startswith("```"):
            message = message.strip("`").removeprefix("json").strip()
        parsed = json.loads(message)
        proposed = {"ICD10CM": str(parsed["icd10cm_code"]).strip(),
                    "SNOMEDCT_US": str(parsed["snomed_code"]).strip()}
        api_result = {system: proposed[system] if terminology.contains(system, proposed[system])
                      else local_prediction[system] for system in SYSTEMS}
        # Partial-override logic: when exactly one system was already locally
        # confident, keep that system's local pick and only accept the hosted
        # answer for the ambiguous system.
        final = {
            "ICD10CM": local_prediction["ICD10CM"] if icd_confident else api_result["ICD10CM"],
            "SNOMEDCT_US": local_prediction["SNOMEDCT_US"] if snomed_confident else api_result["SNOMEDCT_US"],
        }
        self.cache[key] = api_result
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "prediction": api_result}, sort_keys=True) + "\n")
        return final
