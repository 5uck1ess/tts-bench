"""Capability-matrix data coverage + invariants for the Capabilities page.

Guards the model-add checklist: a new model that lands in the harness registry
without capability data (SR / Expressive / License / Langs / ...) fails here, and
the curated cross-lingual flag can't drift onto a model that can't support it.
"""
import importlib

report = importlib.import_module("report")
harness = importlib.import_module("harness")
publish = importlib.import_module("publish")

TRACKED = [row[0] for row in harness.MODELS]

CAP_DICTS = ("MODEL_SR", "MODEL_EXPRESSIVE", "MODEL_LICENSE", "MODEL_LANGS",
             "MODEL_SIZE", "MODEL_KIND", "MODEL_RELEASE", "MODEL_URL")


def test_every_tracked_model_has_all_capability_data():
    missing = {}
    for name in CAP_DICTS:
        d = getattr(report, name)
        gaps = [m for m in TRACKED if m not in d]
        if gaps:
            missing[name] = gaps
    assert not missing, f"capability data gaps: {missing}"


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
