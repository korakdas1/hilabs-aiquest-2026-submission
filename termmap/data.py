"""Input schemas, normalization and local CSV parsing."""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

SYSTEMS = ("ICD10CM", "SNOMEDCT_US")


@dataclass(frozen=True)
class Example:
    entity: str
    context: str
    system: str
    code: str
    display: str


@dataclass(frozen=True)
class InputRow:
    chunk_id: str
    entity: str
    context: str


def normalize(text: str) -> str:
    text = html.unescape(unicodedata.normalize("NFKC", text or "")).casefold()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\bh\s*/\s*o\b", " history of ", text)
    text = re.sub(r"\bw\s*/\s*o\b", " without ", text)
    text = re.sub(r"\bw\s*/\s*(?=\w)", " with ", text)
    text = re.sub(r"\bs\s*/\s*p\b", " status post ", text)
    text = re.sub(r"\bc\s*\.?\s*diff\b", " clostridioides difficile ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"(?<=\w)[/](?=\w)", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def load_training(path: str | Path, diagnosis_only: bool = True) -> list[Example]:
    rows: list[Example] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"entity", "chunk_text", "coding_system", "code", "display", "subconcept"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Training CSV requires columns: {sorted(required)}")
        for row in reader:
            if diagnosis_only and row["subconcept"] != "DiagnosisCode":
                continue
            system = row["coding_system"].strip()
            code = row["code"].strip()
            entity = row["entity"].strip()
            if system not in SYSTEMS or not code or not entity:
                continue
            rows.append(Example(entity, row["chunk_text"] or "", system, code, row["display"] or ""))
    return rows


def iter_input(path: str | Path):
    """Yield input rows without holding a chart-volume file in memory."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        entity_key = "clinical_entity" if "clinical_entity" in fields else "entity" if "entity" in fields else None
        if entity_key is None:
            raise ValueError("Input CSV needs a clinical_entity or entity column")
        context_key = next((x for x in ("chunk_text", "context", "text") if x in fields), None)
        for i, row in enumerate(reader, 1):
            entity = (row.get(entity_key) or "").strip()
            if not entity:
                raise ValueError(f"Empty clinical entity at input row {i + 1}")
            chunk_id = (row.get("chunk_id") or "").strip() or str(i)
            yield InputRow(chunk_id, entity, (row.get(context_key) or "") if context_key else "")


def load_input(path: str | Path) -> list[InputRow]:
    return list(iter_input(path))


def write_predictions(path: str | Path, rows: list[InputRow], predictions: list[dict[str, str]]) -> None:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count differs from input row count")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow({"chunk_id": row.chunk_id, "clinical_entity": row.entity,
                             "icd10cm_code": prediction["ICD10CM"], "snomed_code": prediction["SNOMEDCT_US"]})
