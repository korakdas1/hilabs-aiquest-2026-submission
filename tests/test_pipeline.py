import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from openpyxl import Workbook

from termmap.data import Example, load_input, load_training, normalize, write_predictions
from termmap.model import Candidate, TermMapper
from termmap.neural import NeuralMapper, TrainingAliasIndex, _unique_code_hits
from termmap.openrouter import OpenRouterAdjudicator
from termmap.terminology import Terminology, build_terminology


def test_prediction_schema_and_order(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("chunk_id,clinical_entity,chunk_text\nb,HTN,Known hypertension\na,CKD,Stage 3 kidney disease\n", encoding="utf-8")
    rows = load_input(source)
    assert [(row.chunk_id, row.entity) for row in rows] == [("b", "HTN"), ("a", "CKD")]
    target = tmp_path / "output.csv"
    write_predictions(target, rows, [
        {"ICD10CM": "I10", "SNOMEDCT_US": "38341003"},
        {"ICD10CM": "N18.30", "SNOMEDCT_US": "709044004"},
    ])
    with target.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert list(parsed[0]) == ["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"]
    assert [row["chunk_id"] for row in parsed] == ["b", "a"]


def test_clinical_noise_normalization() -> None:
    assert normalize("discomfort low<br>back") == "discomfort low back"
    assert normalize("h/o HTN w/ CKD") == "history of htn with ckd"
    assert normalize("C. diff infection") == "clostridioides difficile infection"


def test_training_uses_only_diagnoses() -> None:
    path = Path("hackathon_train.csv")
    if not path.exists():
        pytest.skip("Proprietary training file is unavailable")
    rows = load_training(path)
    assert rows
    assert all(row.system in {"ICD10CM", "SNOMEDCT_US"} for row in rows)
    assert all(row.entity and row.code for row in rows)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        expected = sum(row["subconcept"] == "DiagnosisCode" for row in csv.DictReader(handle))
    assert len(rows) == expected


def test_end_to_end_codes_are_in_supplied_dictionaries() -> None:
    model = Path("artifacts/model.joblib")
    terminology = Path("artifacts/terminology.sqlite")
    if not model.exists() or not terminology.exists():
        pytest.skip("Train the local artifacts before running the integration test")
    mapper = TermMapper.load(model, terminology)
    with Terminology(terminology) as db:
        for term, context in [
            ("HTN", "Patient has longstanding hypertension."),
            ("left leg pain", "Reports pain in the left lower limb."),
            ("chronickidney disease stag 3", "Kidney disease stage 3 is documented."),
        ]:
            prediction, _ = mapper.predict_with_db(term, context, db, top_k=1)
            assert db.contains("ICD10CM", prediction["ICD10CM"])
            assert db.contains("SNOMEDCT_US", prediction["SNOMEDCT_US"])


@pytest.mark.parametrize("model", ["qwen/qwen3-235b-a22b-2507",
                                   "deepseek/deepseek-v4.1-flash-1017"])
def test_openrouter_requires_zdr_and_validates_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     model: str) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", model)
    monkeypatch.delenv("OPENROUTER_ZERO_RETENTION_CONFIRMED", raising=False)
    with pytest.raises(ValueError, match="ZERO_RETENTION"):
        OpenRouterAdjudicator(tmp_path / "cache.jsonl", tmp_path / "empty.env")
    monkeypatch.setenv("OPENROUTER_ZERO_RETENTION_CONFIRMED", "true")
    calls = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data)
        calls.append(payload)
        response = {"choices": [{"message": {"content": json.dumps({
            "icd10cm_code": "INVALID", "snomed_code": "38341003"})}}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("termmap.openrouter.urllib.request.urlopen", fake_urlopen)

    class FakeTerminology:
        def contains(self, system, code):
            return (system, code) in {("ICD10CM", "I10"), ("SNOMEDCT_US", "38341003")}

    client = OpenRouterAdjudicator(tmp_path / "cache.jsonl", tmp_path / "empty.env")
    local = {"ICD10CM": "I10", "SNOMEDCT_US": "38341003"}
    candidates = {"ICD10CM": [{"code": "I10", "description": "Hypertension", "baseline": 0.5},
                               {"code": "I10", "description": "Hypertension", "baseline": 0.4}],
                  "SNOMEDCT_US": [{"code": "38341003", "description": "Hypertension", "baseline": 0.5},
                                  {"code": "38341003", "description": "Hypertension", "baseline": 0.4}]}
    result = client.adjudicate("HTN", "Known hypertension", local, candidates, FakeTerminology())
    assert result == local
    assert calls[0]["provider"]["zdr"] is True
    assert calls[0]["provider"]["data_collection"] == "deny"
    assert calls[0]["provider"]["only"] == ["deepinfra"]
    assert calls[0]["max_tokens"] == 384
    assert "reasoning" not in calls[0]
    client.adjudicate("HTN", "Known hypertension", local, candidates, FakeTerminology())
    assert len(calls) == 1
    assert client.stats["remote_calls"] == 1
    assert client.stats["cache_hits"] == 1


def test_licensed_terminology_imports_only_legal_codes(tmp_path: Path) -> None:
    for filename, rows in (
        ("icd10cm_codes.xlsx", [("code", "description"), ("I10", "Essential hypertension"),
                                 ("I10", "Essential hypertension"),
                                 ("I10", "High blood pressure")]),
        ("SNOMEDCT_US_codes_and_descriptions.xlsx", [("code", "description"),
                                                       ("38341003", "Hypertensive disorder"),
                                                       ("59621000", "Essential hypertension")]),
    ):
        book = Workbook()
        for row in rows:
            book.active.append(row)
        book.save(tmp_path / filename)
    umls = tmp_path / "MRCONSO.RRF"
    def atom(system: str, code: str, term: str) -> str:
        fields = [""] * 18
        fields[0] = "C123"
        fields[1], fields[11], fields[13], fields[14], fields[16] = "ENG", system, code, term, "N"
        return "|".join(fields) + "\n"
    umls.write_text(atom("ICD10CM", "I10", "High blood pressure") +
                    atom("SNOMEDCT_US", "38341003", "High blood pressure disorder") +
                    atom("ICD10CM", "FAKE", "Fake diagnosis"), encoding="utf-8")
    descriptions = tmp_path / "sct2_Description_Snapshot-en.txt"
    descriptions.write_text("id\tactive\tconceptId\tterm\n1\t1\t38341003\tHigh blood pressure disorder\n"
                            "2\t1\tFAKE\tFake diagnosis\n", encoding="utf-8")
    relationships = tmp_path / "sct2_Relationship_Snapshot.txt"
    relationships.write_text("id\tactive\tsourceId\tdestinationId\ttypeId\n"
                             "1\t1\t38341003\t59621000\t116680003\n"
                             "2\t1\t38341003\tFAKE\t116680003\n", encoding="utf-8")
    db_path = build_terminology(tmp_path, tmp_path / "terms.sqlite", umls_mrconso=umls,
                                snomed_rf2_descriptions=descriptions,
                                snomed_rf2_relationships=relationships)
    with Terminology(db_path) as db:
        assert list(db.iter_dense_descriptions("ICD10CM")) == [
            ("I10", "Essential hypertension"), ("I10", "High blood pressure")]
        assert list(db.iter_dense_descriptions("SNOMEDCT_US", limit_codes=1)) == [
            ("38341003", "Hypertensive disorder"),
            ("38341003", "High blood pressure disorder")]
        assert db.search("ICD10CM", "High blood pressure")["I10"][0] == 1.0
        assert db.search("SNOMEDCT_US", "High blood pressure disorder")["38341003"][0] == 1.0
        assert not db.contains("ICD10CM", "FAKE")
        assert db.snomed_neighbors(["38341003"])["38341003"] == ["59621000"]
        assert db.umls_links("ICD10CM", ["I10"])["I10"] == ["38341003"]


def test_dense_hits_collapse_aliases_and_training_index_uses_only_fit_examples() -> None:
    pytest.importorskip("faiss", reason="Optional FAISS dependency is unavailable")
    hits = _unique_code_hits(["A", "A", "B"], np.array([0.8, 0.9, 0.7]),
                             np.array([0, 1, 2]), 2)
    assert list(hits.items()) == [("A", 0.9), ("B", 0.7)]

    class FakeEncoder:
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=2))

        def encode(self, texts):
            return np.asarray([[1.0, 0.0] if "blood" in text else [0.0, 1.0]
                               for text in texts], dtype=np.float32)

    fit = {"ICD10CM": SimpleNamespace(examples=[
               Example("high blood pressure", "", "ICD10CM", "I10", "Hypertension")]),
           "SNOMEDCT_US": SimpleNamespace(examples=[])}
    index = TrainingAliasIndex(SimpleNamespace(models=fit), FakeEncoder())
    assert index.search_vector("ICD10CM", np.array([[1.0, 0.0]], dtype=np.float32)) == {"I10": 1.0}
    assert index.search_vector("SNOMEDCT_US", np.array([[1.0, 0.0]], dtype=np.float32)) == {}


def test_neural_override_requires_weak_local_margin_and_dense_agreement() -> None:
    class FakeModel:
        def __init__(self, margin: float):
            self.margin = margin

        def candidates(self, *_args, **_kwargs):
            return [Candidate("A", [0.0], 1.5, "alpha", 0.8),
                    Candidate("B", [0.0], 1.0, "beta", 0.95)]

        def rank_candidates(self, candidates, top_k=5):
            return [{"code": item.code, "score": self.margin if item.code == "A" else 0.0,
                     "description": item.matched_description, "baseline": item.baseline}
                    for item in candidates[:top_k]]

    class FakeDense:
        def search_vector(self, *_args, **_kwargs):
            return {"B": 0.95, "A": 0.8}

    class FakeCross:
        calls = 0

        def score(self, _query, descriptions):
            self.calls += 1
            return np.asarray([5.0 if description == "beta" else 0.0
                               for description in descriptions], dtype=np.float32)

    class FakeMapper:
        def __init__(self, margin):
            self.models = {"ICD10CM": FakeModel(margin)}

    neural = object.__new__(NeuralMapper)
    neural.dense = FakeDense()
    neural.cross = FakeCross()
    neural.local_mapper = FakeMapper(5.0)
    strong, _ = neural.rank_system("ICD10CM", "alpha", "", None,
                                   np.empty((1, 1), dtype=np.float32))
    assert strong[0]["code"] == "A"
    assert neural.cross.calls == 0
    neural.local_mapper = FakeMapper(1.0)
    weak, _ = neural.rank_system("ICD10CM", "alpha", "", None,
                                 np.empty((1, 1), dtype=np.float32))
    assert weak[0]["code"] == "B"
    assert neural.cross.calls == 1
    weak_shortlist, _ = neural.rank_system("ICD10CM", "alpha", "", None,
                                           np.empty((1, 1), dtype=np.float32), top_k=25)
    assert weak_shortlist[0]["code"] == "B"


def test_adjudicator_shortlist_includes_dense_candidates_without_changing_local_decision() -> None:
    class FakeModel:
        def candidates(self, *_args, dense_hits=None, **_kwargs):
            candidates = [Candidate(f"L{i:02d}", [0.0], 1.0, f"local {i}")
                          for i in range(30)]
            if dense_hits:
                candidates.append(Candidate("D", [0.0], 0.8, "dense match", 0.95))
            return candidates

        def rank_candidates(self, candidates, top_k=5):
            return [{"code": item.code,
                     "score": 5.0 if item.code == "L00" else 0.0,
                     "description": item.matched_description,
                     "baseline": item.baseline}
                    for item in candidates[:top_k]]

    class FakeDense:
        def search_vector(self, *_args, **_kwargs):
            return {"D": 0.95}

    class FakeCross:
        calls = 0

        def score(self, _query, descriptions):
            self.calls += 1
            return np.asarray([1.0 if desc == "dense match" else 0.0
                               for desc in descriptions], dtype=np.float32)

    neural = object.__new__(NeuralMapper)
    neural.local_mapper = type("FakeMapper", (), {"models": {"ICD10CM": FakeModel()}})()
    neural.dense = FakeDense()
    neural.cross = FakeCross()
    vector = np.empty((1, 1), dtype=np.float32)

    local, _ = neural.rank_system("ICD10CM", "alpha", "", None, vector, top_k=5)
    shortlist, _ = neural.rank_system("ICD10CM", "alpha", "", None, vector, top_k=25)
    assert local[0]["code"] == shortlist[0]["code"] == "L00"
    assert "D" not in {row["code"] for row in local}
    assert "D" in {row["code"] for row in shortlist}
    assert len(shortlist) == len({row["code"] for row in shortlist}) == 25
    assert neural.cross.calls == 1


def test_learned_neural_path_uses_adapter_and_scores_strong_local_case() -> None:
    class FakeModel:
        def candidates(self, *_args, **_kwargs):
            return [Candidate("A", [0.0], 1.5, "alpha", 0.8),
                    Candidate("B", [0.0], 1.0, "beta", 0.95)]

        def rank_candidates(self, candidates, top_k=5):
            return [{"code": item.code, "score": 5.0 if item.code == "A" else 0.0,
                     "description": item.matched_description, "baseline": item.baseline}
                    for item in candidates[:top_k]]

    class FakeDense:
        vectors = []

        def search_vector(self, _system, vector, **_kwargs):
            self.vectors.append(float(vector[0, 0]))
            return {"B": 0.95, "A": 0.8}

    class FakeCross:
        calls = 0

        def score(self, _query, descriptions):
            self.calls += 1
            return np.asarray([5.0 if description == "beta" else 0.0
                               for description in descriptions], dtype=np.float32)

    neural = object.__new__(NeuralMapper)
    neural.local_mapper = SimpleNamespace(models={"ICD10CM": FakeModel()})
    neural.dense = FakeDense()
    neural.cross = FakeCross()
    neural.adapter = SimpleNamespace(transform=lambda x: x + 1)
    neural.learned_dir = Path("unused")
    ranked, _ = neural.rank_system("ICD10CM", "alpha", "", None,
                                   np.array([[1.0]], dtype=np.float32))
    assert ranked[0]["code"] == "B"
    assert neural.cross.calls == 1
    assert neural.dense.vectors == [1.0, 2.0]
