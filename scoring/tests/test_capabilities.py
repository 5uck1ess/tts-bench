"""Capability-matrix data coverage + invariants for the Capabilities page.

Guards the model-add checklist: a new model that lands in the harness registry
without capability data (SR / Expressive / License / Langs / ...) fails here, and
the curated cross-lingual flag can't drift onto a model that can't support it.
"""
import importlib
import re
from pathlib import Path

report = importlib.import_module("report")
harness = importlib.import_module("harness")
publish = importlib.import_module("publish")

TRACKED = [row[0] for row in harness.MODELS]
TRACKED_SET = set(TRACKED)
ROOT = Path(__file__).resolve().parents[2]

CAP_DICTS = ("MODEL_DISPLAY_NAMES", "MODEL_SR", "MODEL_EXPRESSIVE",
             "MODEL_LICENSE", "MODEL_LANGS", "MODEL_SIZE", "MODEL_KIND",
             "MODEL_RELEASE", "MODEL_URL")


def test_capability_data_matches_registry_exactly():
    drift = {}
    for name in CAP_DICTS:
        keys = set(getattr(report, name))
        missing = sorted(TRACKED_SET - keys)
        stale = sorted(keys - TRACKED_SET)
        if missing or stale:
            drift[name] = {"missing": missing, "stale": stale}
    assert not drift, f"capability data drift: {drift}"


def test_readme_model_counts_match_registry():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tracked_match = re.search(r"^## Models tracked \((\d+)\)$", readme, re.MULTILINE)
    cloning_match = re.search(
        r"\*\*(\d+) of the (\d+) tracked models can clone\*\*", readme)
    assert tracked_match, "README is missing the tracked-model count"
    assert cloning_match, "README is missing the cloning-model count"

    clone_capable = sum(row[6] is not False for row in harness.MODELS)
    assert int(tracked_match.group(1)) == len(TRACKED)
    assert int(cloning_match.group(1)) == clone_capable
    assert int(cloning_match.group(2)) == len(TRACKED)


def test_crosslingual_only_for_cloners():
    bad = [m for m in report.MODEL_CROSSLINGUAL if report.MODEL_KIND.get(m) != "cloning"]
    assert not bad, f"cross-lingual flag set on non-cloning models: {bad}"


def test_crosslingual_implies_multilingual():
    bad = [m for m in report.MODEL_CROSSLINGUAL if not report._is_multilingual(m)]
    assert not bad, f"cross-lingual but not multilingual: {bad}"


def test_commercial_heuristic_flags_known_licenses():
    assert report._is_commercial("kokoro") is True    # Apache 2.0
    assert report._is_commercial("coqui") is False     # CPML (non-commercial)
    assert report._is_commercial("f5tts") is False      # CC-BY-NC
    assert report._is_commercial("wavtts") is False      # weights CC-BY-NC 4.0


def test_sr_hz_parses_readme_cells():
    assert report._sr_hz("melotts") == 44100
    assert report._sr_hz("wavtts") == 16000
    assert report._sr_hz("voxcpm") == 48000


def test_cpu_filter_derived_from_harness_devices():
    # pocket advertises devices ["cpu"]; higgs_v3 is a cuda-only GPU_CLASS model.
    assert "pocket" in publish._CPU_OK
    assert "higgs_v3" not in publish._CPU_OK


def test_dual_capability_models_are_cloners():
    # outetts / voxtral ship preset voices AND cloning; MODEL_KIND tags them cloning.
    for m in publish._PRESET_AND_CLONE:
        assert report.MODEL_KIND.get(m) == "cloning"
