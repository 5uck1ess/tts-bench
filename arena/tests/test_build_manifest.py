from pathlib import Path
from arena.build_manifest import RIG_PRIO, DEV_PRIO, NO_PRESET_VOICE, scan_dirs


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"RIFF")


def test_scan_picks_best_rig_and_device(tmp_path):
    base = "https://x.test/tts-bench/"
    # same (kokoro, p1) on windows-cpu and linux-cuda -> windows wins (rig priority)
    _touch(tmp_path / "windows-default" / "kokoro_cpu_p1.wav")
    _touch(tmp_path / "linux-default" / "kokoro_cuda_p1.wav")
    clips, ref = scan_dirs(tmp_path, "default", base)
    urls = {(c["model"], c["prompt"]): c["url"] for c in clips}
    assert urls[("kokoro", 1)] == base + "windows-default/kokoro_cpu_p1.wav"
    assert ref is None


def test_scan_default_drops_no_preset_models(tmp_path):
    base = "https://x.test/tts-bench/"
    sample = next(iter(NO_PRESET_VOICE))
    _touch(tmp_path / "windows-default" / f"{sample}_cuda_p1.wav")
    _touch(tmp_path / "windows-default" / "kokoro_cuda_p1.wav")
    clips, _ = scan_dirs(tmp_path, "default", base)
    models = {c["model"] for c in clips}
    assert sample not in models
    assert "kokoro" in models


def test_scan_cloning_excludes_mac_and_finds_reference(tmp_path):
    base = "https://x.test/tts-bench/"
    _touch(tmp_path / "windows-cloning" / "echo_cuda_p1.wav")
    _touch(tmp_path / "mac-cloning" / "echo_mps_p1.wav")
    _touch(tmp_path / "windows-cloning" / "_reference.wav")
    clips, ref = scan_dirs(tmp_path, "cloning", base)
    urls = {(c["model"], c["prompt"]): c["url"] for c in clips}
    assert urls[("echo", 1)] == base + "windows-cloning/echo_cuda_p1.wav"  # not mac
    assert ref == base + "windows-cloning/_reference.wav"


def test_scan_cloning_fallback_uses_default_dir_for_no_preset(tmp_path):
    base = "https://x.test/tts-bench/"
    sample = next(iter(NO_PRESET_VOICE))
    # no cloning clip for `sample`, but it has a default-dir clip -> fallback includes it
    _touch(tmp_path / "windows-default" / f"{sample}_cuda_p1.wav")
    _touch(tmp_path / "windows-cloning" / "echo_cuda_p1.wav")
    clips, _ = scan_dirs(tmp_path, "cloning", base)
    models = {c["model"] for c in clips}
    assert sample in models  # pulled from default dir as a Chris clone


def test_build_manifest_embeds_model_meta_for_present_slugs(tmp_path):
    from arena.build_manifest import build_manifest
    base = "https://x.test/tts-bench/"
    _touch(tmp_path / "windows-default" / "kokoro_cpu_p1.wav")
    _touch(tmp_path / "windows-cloning" / "echo_cuda_p1.wav")
    meta = {"kokoro": {"name": "Kokoro", "url": "https://hf.test/hexgrad/Kokoro-82M"},
            "unbenched": {"name": "Ghost", "url": "https://hf.test/x/ghost"}}
    m = build_manifest(tmp_path, base, {"1": ["en", "hi"]}, meta)
    # only slugs with clips are emitted; meta passthrough for known, fallback for unknown
    assert set(m["models"]) == {"kokoro", "echo"}
    assert m["models"]["kokoro"]["url"] == "https://hf.test/hexgrad/Kokoro-82M"
    assert m["models"]["echo"] == {"name": "echo", "url": None}
