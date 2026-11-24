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


def test_duplicate_audio_fields_are_an_error():
    good = base64.b64encode(b"\x01\x02" * 50).decode()  # valid PCM16: the LAST field alone would pass
    raw = ('{"audio": "%s", "audio": "%s"}' % (base64.b64encode(b"abc").decode(), good)).encode()
    with pytest.raises(RuntimeError, match="not exactly"):
        b"".join(PhonicMerritt("k").audio_chunks(iter([raw])))


def _wav16k():
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 8821)  # odd data chunk + pad byte: even total, still not PCM
    return buf.getvalue()


@pytest.mark.parametrize("magic", [b"RIFF", b"RIFX", b"RF64", b"BW64", b"FORM", b"OggS", b"fLaC",
                                   b"ID3\x04"])
def test_audio_containers_are_an_error(magic):
    payload = magic + _wav16k()[4:]  # a real WAV body under each container's signature
    with pytest.raises(RuntimeError, match="container"):
        _decode({"audio": base64.b64encode(payload).decode()})


def test_extra_fields_rejected_fast():
    """Any field besides "audio" is rejected by one linear match, however many there are."""
    import time
    fields = ",".join(f'"f{i}": 0' for i in range(1_500_000))  # ~18 MB, the reviewer's case
    raw = ('{%s, "audio": "%s"}' % (fields, base64.b64encode(b"\x01\x02" * 10).decode())).encode()
    t = time.perf_counter()
    with pytest.raises(RuntimeError):
        b"".join(PhonicMerritt("k").audio_chunks(iter([raw])))
    assert time.perf_counter() - t < 1.0


def test_body_size_capped():
    from scripts.hosted.providers import phonic
    chunks = iter([b"x" * (1 << 20)] * (phonic.MAX_BODY // (1 << 20) + 1))
    with pytest.raises(RuntimeError, match="over"):
        b"".join(PhonicMerritt("k").audio_chunks(chunks))


def test_escaped_slash_in_base64_is_accepted():
    pcm = bytes([0xFF, 0xFE]) * 30  # base64 of this contains "/"
    b64 = base64.b64encode(pcm).decode()
    assert "/" in b64
    raw = ('{"audio": "%s"}' % b64.replace("/", "\\/")).encode()
    assert b"".join(PhonicMerritt("k").audio_chunks(iter([raw]))) == pcm


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
