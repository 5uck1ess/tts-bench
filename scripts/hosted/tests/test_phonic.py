import base64
import json

import pytest

from scripts.hosted import keys, speed_sample
from scripts.hosted.providers.phonic import PhonicMerritt


def _decode(body):
    return b"".join(PhonicMerritt("k").audio_chunks(iter([json.dumps(body).encode()])))


def test_decodes_one_base64_string():
    pcm = b"\x01\x02" * 100
    assert _decode({"audio": base64.b64encode(pcm).decode()}) == pcm


@pytest.mark.parametrize("audio", [
    base64.b64encode(b"\x01\x02\x03\x04\x05").decode() * 2,  # two padded segments
    "not base64!",
    base64.b64encode(b"\x01\x02\x03").decode(),               # odd PCM16 byte count
    "",
    None,
])
def test_malformed_audio_is_an_error_not_a_clip(audio):
    with pytest.raises(RuntimeError):
        _decode({"audio": audio})


def test_sampler_skips_vendor_missing_from_key_cache(tmp_path, monkeypatch):
    """A key cache made before a vendor was added must not fail the unattended pass."""
    monkeypatch.setattr(speed_sample, "OUT", tmp_path)
    monkeypatch.setattr(speed_sample, "LOG", tmp_path / "samples.jsonl")
    calls = []

    def fake_sample(slug, classes, pass_id, rows):
        calls.append(slug)
        if slug == "phonic_merritt":
            raise keys.MissingKey("no 'phonic' field")
        rows.append({"error": None})

    monkeypatch.setattr(speed_sample, "sample", fake_sample)
    monkeypatch.setattr("sys.argv", ["x", "--model", "phonic_merritt", "--model", "rime_coda"])
    assert speed_sample.main() == 0
    assert calls == ["phonic_merritt", "rime_coda"]
    assert "SKIP" in (tmp_path / "sampler.log").read_text()
