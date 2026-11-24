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


_W64 = bytes.fromhex("726966662e91cf11a5d628db04c10000")
_BODY = b"\x00" * 4000


@pytest.mark.parametrize("header", [
    pytest.param(None, id="wav"),
    pytest.param(b"RIFX\x00\x00\x0f\xa0WAVE", id="rifx"),
    pytest.param(b"RF64\xff\xff\xff\xffWAVE", id="rf64"),
    pytest.param(b"BW64\xff\xff\xff\xffWAVE", id="bw64"),
    pytest.param(_W64 + b"\x00" * 8 + b"wave", id="wave64"),
    pytest.param(b"FORM\x00\x00\x0f\xa0AIFF", id="aiff"),
    pytest.param(b"FORM\x00\x00\x0f\xa0AIFC", id="aifc"),
    pytest.param(b"OggS\x00\x02" + b"\x00" * 8, id="ogg"),
    pytest.param(b"fLaC\x80\x00\x00\x22", id="flac"),
    pytest.param(b"ID3\x04\x00\x00\x00\x00\x02\x01", id="id3"),
])
def test_audio_containers_are_an_error(header):
    payload = _wav16k() if header is None else header + _BODY
    payload += b"\x00" * (len(payload) % 2)  # even length: the container check must be what fails
    with pytest.raises(RuntimeError, match="container"):
        _decode({"audio": base64.b64encode(payload).decode()})


@pytest.mark.parametrize("first_samples", [
    (17481, 51),          # b"ID3\x00": not an ID3v2 version byte
    (18770, 17990),       # b"RIFF" with no "WAVE" form type after it
    (26447, 21351),       # b"OggS" with a nonzero version byte after it
])
def test_pcm_that_starts_like_a_signature_is_kept(first_samples):
    import struct
    pcm = struct.pack("<2h", *[s - 65536 if s > 32767 else s for s in first_samples])
    pcm += bytes([1, 2]) + b"\x07\x00" * 200
    assert pcm[:4] in (b"ID3\x00", b"RIFF", b"OggS")  # the vector really is the signature
    assert _decode({"audio": base64.b64encode(pcm).decode()}) == pcm


def test_json_escapes_are_decoded_like_any_json():
    pcm = b"\x01\x02" * 60
    b64 = base64.b64encode(pcm).decode()
    raw = ('{"\\u0061udio": "\\u00%02x%s"}' % (ord(b64[0]), b64[1:])).encode()
    assert b"".join(PhonicMerritt("k").audio_chunks(iter([raw]))) == pcm


def test_non_json_whitespace_is_an_error():
    raw = b"\x0b" + json.dumps({"audio": base64.b64encode(b"\x01\x02" * 10).decode()}).encode()
    with pytest.raises(RuntimeError, match="not exactly"):
        b"".join(PhonicMerritt("k").audio_chunks(iter([raw])))


def test_one_byte_http_chunks_decode():
    pcm = b"\x03\x04" * 50_000
    raw = json.dumps({"audio": base64.b64encode(pcm).decode()}).encode()
    chunks = (raw[i:i + 1] for i in range(len(raw)))
    assert b"".join(PhonicMerritt("k").audio_chunks(chunks)) == pcm


def test_extra_fields_rejected_fast():
    """Extra fields are an error, and a body at the cap parses well under a second."""
    import time
    from scripts.hosted.providers import phonic
    b64 = base64.b64encode(b"\x01\x02" * 10).decode()
    fields, n = [], 0
    while n < phonic.MAX_BODY - 200:
        f = f'"f{len(fields)}": 0'
        fields.append(f)
        n += len(f) + 1
    raw = ('{%s, "audio": "%s"}' % (",".join(fields), b64)).encode()
    assert len(raw) <= phonic.MAX_BODY
    t = time.perf_counter()
    with pytest.raises(RuntimeError, match="not exactly"):
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


def _mp3_frames(n=3):
    # MPEG-1 Layer III, 128 kbps, 44.1 kHz, no padding: 144 * 128000 // 44100 = 417 bytes/frame
    return (b"\xff\xfb\x90\x64" + b"\x00" * 413) * n


def test_untagged_mp3_is_an_error():
    with pytest.raises(RuntimeError, match="container"):
        _decode({"audio": base64.b64encode(_mp3_frames()).decode()})


def test_one_frame_sync_alone_is_kept_as_pcm():
    pcm = _mp3_frames(1)[:416] +b"\x05\x00" * 300  # one header, then no second frame
    assert _decode({"audio": base64.b64encode(pcm).decode()}) == pcm


def test_riff_prefixed_pcm_survives_wav_conversion():
    """Headerless PCM starting with samples 18770, 17990 (b"RIFF") is saved as 44.1 kHz PCM."""
    from scripts.hosted import core
    pcm = b"RIFF" + b"\x07\x00" * 4410
    wav, sr, dur = core.to_wav(pcm, "pcm16", 44100, raw_pcm=PhonicMerritt.raw_pcm)
    assert sr == 44100 and wav.endswith(pcm) and abs(dur - len(pcm) / 2 / 44100) < 1e-9


class _Resp:
    status_code = 200
    headers = {"content-type": "application/json"}
    http_version = "HTTP/1.1"

    def __init__(self, body):
        self.body = body

    def iter_bytes(self):
        yield self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Client:
    def __init__(self, body):
        self.body = body

    def stream(self, *a, **k):
        return _Resp(self.body)


@pytest.mark.parametrize("body", [b'{"nope": 1}', b'{"call over": 1}'])
def test_decode_past_the_deadline_reports_the_deadline(monkeypatch, body):
    """Parsing that carries the call past MAX_CALL_S is a timeout, even if the parse then fails."""
    from scripts.hosted import core
    from scripts.hosted.providers import phonic
    now = {"t": 0.0}
    monkeypatch.setattr(core.time, "perf_counter", lambda: now["t"])
    real_loads = phonic.json.loads

    def slow_loads(*a, **k):
        now["t"] = core.MAX_CALL_S + 0.1  # the parse is what crosses the deadline
        return real_loads(*a, **k)

    monkeypatch.setattr(phonic.json, "loads", slow_loads)
    ad = PhonicMerritt("k")
    with pytest.raises(core.CallTimeout):  # the type, not text a response could contain
        core.synth(_Client(body), ad, "hi", "en")


def test_free_format_mp3_is_an_error():
    # MPEG-1 Layer III, 44.1 kHz, bitrate index 0 (free format): length = distance to next header
    frame = b"\xff\xfb\x00\x64" + b"\x11" * 496
    with pytest.raises(RuntimeError, match="container"):
        _decode({"audio": base64.b64encode(frame * 12).decode()})


def test_free_format_header_alone_is_kept_as_pcm():
    pcm = b"\xff\xfb\x00\x64" + b"\x11" * 996
    assert _decode({"audio": base64.b64encode(pcm).decode()}) == pcm


def test_other_vendors_still_reject_a_riff_body_that_is_not_wav():
    """raw_pcm is Phonic's opt-in; a pcm16 vendor's RIFF/AVI body is still an error."""
    from scripts.hosted import core
    avi = b"RIFF" + (1000).to_bytes(4, "little") + b"AVI LIST" + b"\x00" * 992
    with pytest.raises(RuntimeError, match="data chunk"):
        core.to_wav(avi, "pcm16", 48000)


def test_errors_never_quote_the_body_or_the_key():
    key = "sk-test-0123456789abcdef"
    ad = PhonicMerritt(key)
    body = ('{"error": "bad", "echo": "Authorization: Bearer %s"}' % key).encode()
    with pytest.raises(RuntimeError) as e:
        b"".join(ad.audio_chunks(iter([body])))
    assert key not in str(e.value) and "Bearer" not in str(e.value)


def test_synth_scrubs_the_key_from_any_adapter_error():
    from scripts.hosted import core
    key = "sk-test-0123456789abcdef"

    class Leaky(PhonicMerritt):
        def audio_chunks(self, raw_iter):
            raise RuntimeError(f"vendor said: Bearer {self.key}")
            yield b""

    with pytest.raises(RuntimeError) as e:
        core.synth(_Client(b"{}"), Leaky(key), "hi", "en")
    assert key not in str(e.value) and "[redacted]" in str(e.value)
