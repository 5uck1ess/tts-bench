"""WAV-wrapped PCM16 streams: metadata chunks before `data` must not be read as audio."""

import struct

import numpy as np

from scripts.hosted import core

SR = 48000


def _riff(chunks):
    body = b"WAVE" + b"".join(cid + struct.pack("<I", len(p)) + p + (b"\0" if len(p) % 2 else b"")
                              for cid, p in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _fmt():
    return b"fmt ", struct.pack("<HHIIHH", 1, 1, SR, SR * 2, 2, 16)


def _stream(meta):
    silence = np.zeros(SR // 10, dtype="<i2").tobytes()          # 100 ms
    speech = (np.sin(np.arange(SR // 10) / 5) * 20000).astype("<i2").tobytes()
    head = _riff([_fmt(), (b"LIST", meta)])
    data_hdr = b"data" + struct.pack("<I", 0xFFFFFFFF)            # streamed placeholder size
    data = head + data_hdr + silence + speech
    # arrivals: header+meta at 100 ms, silence at 200 ms, speech at 350 ms
    a1 = len(head) + 8
    return data, [(a1, 100.0), (a1 + len(silence), 200.0), (len(data), 350.0)]


def test_metadata_containing_data_bytes_is_not_speech():
    # Loud bytes inside the metadata, plus a literal b"data" before the real data chunk.
    meta = b"cue data " + bytes([0x00, 0x7F]) * 400
    data, arrivals = _stream(meta)
    assert core._first_speech_ms(data, "pcm16", arrivals) == 350.0


def test_offset_walks_chunks():
    data, _ = _stream(b"xxdataxx")
    off = core.wav_data_offset(data)
    assert data[off - 8:off - 4] == b"data"
    assert off > data.find(b"data")  # the substring inside LIST comes first


def test_to_wav_strips_header_and_keeps_rate():
    data, _ = _stream(b"meta")
    wav, sr, dur = core.to_wav(data, "pcm16", 24000)  # header rate wins over adapter default
    assert sr == SR and abs(dur - 0.2) < 1e-6


def test_incomplete_header_has_no_speech_time():
    data, arrivals = _stream(b"m" * 100)
    assert core._first_speech_ms(data[:30], "pcm16", arrivals) is None
