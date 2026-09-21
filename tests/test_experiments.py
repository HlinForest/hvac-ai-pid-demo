import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.site import export, summarize_repeats, publishable
from hvac_pid.artifacts import read_json, write_json
from hvac_pid.core import Gains, Scenario, evaluate
from hvac_pid.fnn import FNN
from hvac_pid.rl import QPolicy
from hvac_pid.tuning import identify, simc, zn


REFERENCE = Path(__file__).resolve().parents[1] / "experiments/reference"


def test_reference_metrics_match_csv_and_current_controller():
    metrics = read_json(REFERENCE / "03-classical/metrics.json")
    for name in ["ZN", "SIMC"]:
        with (REFERENCE / f"03-classical/{name}.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        expected = sum(abs(float(r["temperature"]) - float(r["setpoint"])) * 0.1 for r in rows)
        assert metrics[name]["iae"] == pytest.approx(expected, abs=1e-8)
        gains = zn(identify(Scenario())[0]) if name == "ZN" else simc(identify(Scenario())[0])
        assert evaluate(Scenario(), gains).iae == pytest.approx(expected, abs=1e-6)


def test_reference_models_reproduce_recorded_nominal_evaluations():
    s = Scenario()
    identified = identify(s)[0]
    fnn = FNN.load(REFERENCE / "05-fnn/model.npz")
    q = QPolicy.load(REFERENCE / "06-qlearning/model.npz")
    assert evaluate(s, fnn.predict(identified)).iae == pytest.approx(read_json(REFERENCE / "05-fnn/metrics.json")["FNN"]["iae"], abs=1e-6)
    assert evaluate(s, simc(identified), q).iae == pytest.approx(read_json(REFERENCE / "06-qlearning/metrics.json")["Q-Learning"]["iae"], abs=1e-6)


def test_reference_labels_do_not_include_test_objects():
    labels = read_json(REFERENCE / "05-fnn/labels.json")
    splits = read_json(REFERENCE / "05-fnn/splits.json")["splits"]
    assert len(labels["train"]) == 48 and len(labels["validation"]) == 12
    train_objects = [r["scenario"] for r in labels["train"]]
    assert train_objects == splits["train"]
    assert not any(s in train_objects for s in splits["test"])


def test_site_export_preserves_handwritten_prose(tmp_path):
    chapter = tmp_path / "chapters/01.md"
    chapter.parent.mkdir()
    chapter.write_text("A human-written explanation.", encoding="utf-8")
    stale = tmp_path / "public/results/old-credentials.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("private stale artifact", encoding="utf-8")
    export(REFERENCE, tmp_path)
    assert not stale.exists()
    assert chapter.read_text(encoding="utf-8") == "A human-written explanation."
    assert (tmp_path / "generated/bo-trials.md").exists()
    status = read_json(REFERENCE / "08-llm/result.json")["status"]
    assert status in (tmp_path / "generated/llm-status.md").read_text(encoding="utf-8")
    manifest = read_json(tmp_path / "public/results/manifest.json")
    assert "08-llm/result.json" in manifest["assets"]
    assert "05-fnn/model.npz" in manifest["assets"]
    assert (tmp_path / "public/results/03-classical/ZN.csv").exists()


def test_public_artifacts_use_allowlist_and_reject_credentials(tmp_path):
    assert not publishable(Path(".env"))
    assert not publishable(Path("08-llm/credentials.json"))
    assert not publishable(Path("private/result.json"))
    assert publishable(Path("explain/sac.json"))
    source = tmp_path / "source"
    write_json(source / "08-llm/result.json", {"accidental_secret": "sk-" + "a" * 32})
    with pytest.raises(ValueError, match="Credential-like"):
        export(source, tmp_path / "site")


def test_repeats_summary_uses_independent_runs(tmp_path):
    for seed, value in [(0, 10.0), (1, 14.0)]:
        write_json(tmp_path / f"seed-{seed}/09-compare/frozen.json", [{"case": "nominal", "method": "BO", "iae": value}])
    summarize_repeats(tmp_path, [0, 1])
    row = read_json(tmp_path / "summary.json")["rows"][0]
    assert row["mean_iae"] == 12
    assert row["std_iae"] == pytest.approx(np.std([10, 14], ddof=1))


def test_live_reference_uses_same_initial_samples_and_reproduces_best_score():
    llm = read_json(REFERENCE / "08-llm/result.json")
    bo = read_json(REFERENCE / "04-bo/result.json")
    assert llm["status"] == "complete"
    assert llm["cost"]["calls"] == 15
    assert len(llm["exchanges"]) == 15
    assert llm["cost"]["simulations"] == 20
    for left, right in zip(llm["trials"][:5], bo["trials"][:5]):
        assert [left[k] for k in ("kp", "ki", "iae")] == pytest.approx([right[k] for k in ("kp", "ki", "iae")])
    best = min(t["iae"] for t in llm["trials"] if t["status"] == "evaluated")
    assert evaluate(Scenario(), Gains(**llm["best"])).iae == pytest.approx(best, abs=1e-7)
