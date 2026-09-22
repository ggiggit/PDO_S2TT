import importlib.util
import json
from pathlib import Path
import sys
import types

from pdo_s2tt.evaluation import quality_metrics


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def latency_stub(records: int) -> dict:
    pair = {"mean": 1.0, "p90": 1.0}
    return {
        "records": records,
        "FTL": pair,
        "FRD": pair,
        "LAAL_CU": pair,
        "LAAL_CA_star": pair,
        "RTF": pair,
        "max_compute_backlog": pair,
    }


def revision_stub() -> dict:
    return {
        key: {"records": 647, "mean": 0.0, "p90": 0.0}
        for key in (
            "normalized_erasure",
            "age_weighted_erasure",
            "first_stable_unit",
            "mean_finalization",
        )
    }


def test_direction_evaluation_can_disable_released_checkpoint_check(monkeypatch, tmp_path):
    module = load_script("evaluate_fleurs")
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    gold = {
        "id": "one",
        "source_text": "hello",
        "target_text": "hallo",
        "audio_duration_sec": 2.0,
    }
    prediction = {"id": "one", "target_lang": "de"}
    monkeypatch.setattr(
        module,
        "read_jsonl",
        lambda path: [gold] if path == manifest else [prediction],
    )
    monkeypatch.setattr(
        module,
        "quality_metrics",
        lambda *args: {
            "records": 1,
            "output_coverage": 1.0,
            "BLEU": 0.0,
            "COMET": None,
            "chrF++": 0.0,
        },
    )
    monkeypatch.setattr(module, "latency_metrics", lambda rows: latency_stub(len(rows)))
    monkeypatch.setattr(module, "revision_metrics", lambda *args: revision_stub())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_fleurs.py",
            "--target",
            "de",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--skip-comet",
            "--no-reference-check",
        ],
    )
    module.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["reproduction_check"]["enabled"] is False


def test_macro_evaluation_can_disable_released_checkpoint_check(monkeypatch, tmp_path):
    module = load_script("evaluate_macro")
    output = tmp_path / "macro.json"

    def metric_row(path: Path) -> dict:
        target = path.parent.name.removeprefix("en-")
        return {
            "direction": f"En→{target}",
            "quality": {
                "records": 647,
                "output_coverage": 1.0,
                "BLEU": 1.0,
                "COMET": 1.0,
                "chrF++": 1.0,
            },
            "revision": revision_stub(),
        }

    monkeypatch.setattr(module, "load", metric_row)
    monkeypatch.setattr(module, "read_jsonl", lambda path: [{} for _ in range(647)])
    monkeypatch.setattr(module, "latency_metrics", lambda rows: latency_stub(len(rows)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_macro.py",
            "--results-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--no-reference-check",
        ],
    )
    module.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["reproduction_check"]["enabled"] is False
    assert result["quality"]["records"] == 5 * 647


def test_quality_metrics_resolves_pinned_comet_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoints" / "model.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    class FakeModel:
        def predict(self, rows, batch_size, gpus):
            assert rows == [{"src": "hello", "mt": "hallo", "ref": "hallo"}]
            return types.SimpleNamespace(system_score=0.75)

    comet = types.ModuleType("comet")
    comet.load_from_checkpoint = lambda path: FakeModel()
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **kwargs: str(tmp_path)
    monkeypatch.setitem(sys.modules, "comet", comet)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    result = quality_metrics(
        [{"source_text": "hello", "translation": "hallo", "reference": "hallo"}],
        "de",
    )
    assert result["COMET"] == 75.0
