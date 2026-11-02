import io
import json
import wave

from scripts.hosted import core


class _Ad:
    slug = "fake_vendor"
    clone_kind = "voice_id"
    clone_cut = "30"
    clone_ref = None

    def create_clone(self, client, ref):
        raise AssertionError("a cached clone must be reused, not recreated")


def _wav(seconds, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\0\0" * int(seconds * sr))
    return buf.getvalue()


def _setup(tmp_path, monkeypatch, entry):
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    (ref_dir / "tym_hosted_30.wav").write_bytes(_wav(24.8))
    (ref_dir / "tym_hosted_30.txt").write_text("text", encoding="utf-8")
    clones = tmp_path / "clones.json"
    clones.write_text(json.dumps({"fake_vendor": entry}), encoding="utf-8")
    monkeypatch.setattr(core, "REF_DIR", ref_dir)
    monkeypatch.setattr(core, "CLONES", clones)


def test_cached_clone_reports_the_seconds_the_vendor_got(tmp_path, monkeypatch):
    """A console clone trimmed in the vendor's UI (AssemblyAI: 24.8 s cut -> 19 s) must be
    reported at the registry's length, not the local file's, or sidecars say 24.8."""
    _setup(tmp_path, monkeypatch, {"voice_id": "v1", "cut": "30", "seconds": 19.0})
    ad = _Ad()
    assert core.prepare_clone(None, ad) == "v1"
    assert ad.clone_ref["seconds"] == 19.0
    assert ad.voice("en") == "v1"


def test_cached_clone_without_seconds_keeps_the_file_length(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, {"voice_id": "v1", "cut": "30"})
    ad = _Ad()
    core.prepare_clone(None, ad)
    assert ad.clone_ref["seconds"] == 24.8
