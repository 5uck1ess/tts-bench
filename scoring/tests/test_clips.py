import os
from scoring.clips import parse_wav_name, discover_clips, Clip


def test_parse_wav_name_ok():
    c = parse_wav_name("windows-cloning", "moss_tts_v15_cuda_p3.wav")
    assert c == Clip(dir="windows-cloning", wav="moss_tts_v15_cuda_p3.wav",
                     model="moss_tts_v15", dev="cuda", prompt_id="3", mode="cloning")


def test_parse_wav_name_default_mode_from_dir():
    c = parse_wav_name("linux-default", "kokoro_cpu_p1.wav")
    assert c.model == "kokoro" and c.mode == "default" and c.prompt_id == "1"


def test_parse_wav_name_rejects_non_clip():
    assert parse_wav_name("windows-default", "_reference.wav") is None
    assert parse_wav_name("windows-default", "results.csv") is None
    assert parse_wav_name("windows-default", "model_gpu_p1.wav") is None  # bad device


def test_discover_clips(tmp_path):
    d = tmp_path / "windows-default"
    d.mkdir()
    (d / "kokoro_cuda_p1.wav").write_bytes(b"x")
    (d / "kokoro_cuda_p2.wav").write_bytes(b"x")
    (d / "_reference.wav").write_bytes(b"x")
    (tmp_path / "not-a-mode").mkdir()
    (tmp_path / "not-a-mode" / "kokoro_cuda_p1.wav").write_bytes(b"x")
    clips = discover_clips(str(tmp_path))
    assert {c.wav for c in clips} == {"kokoro_cuda_p1.wav", "kokoro_cuda_p2.wav"}
    assert all(c.mode == "default" for c in clips)
