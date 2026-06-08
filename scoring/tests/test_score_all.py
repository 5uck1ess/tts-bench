from scoring.clips import Clip
from scoring.score_all import score_clips


class FakeUtmos:
    name = "utmos"
    def score(self, wav): return 4.0

class FakeWer:
    name = "wer"
    def score(self, wav, ref_text, lang): return 0.0

class FakeSim:
    name = "sim"
    def score(self, wav, ref): return 0.9


def test_score_clips_default_has_no_sim():
    clips = [Clip("windows-default", "kokoro_cuda_p1.wav", "kokoro", "cuda", "1", "default")]
    rows = score_clips(clips, FakeUtmos(), FakeWer(), FakeSim(),
                       ref_for_dir=lambda d: None)
    assert len(rows) == 1
    r = rows[0]
    assert r["utmos"] == "4.0000" and r["wer"] == "0.0000" and r["sim"] == ""


def test_score_clips_cloning_has_sim_when_ref_present():
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    rows = score_clips(clips, FakeUtmos(), FakeWer(), FakeSim(),
                       ref_for_dir=lambda d: "ref.wav")
    assert rows[0]["sim"] == "0.9000"


def test_score_clips_unknown_prompt_blank_wer():
    clips = [Clip("windows-default", "kokoro_cuda_p9.wav", "kokoro", "cuda", "9", "default")]
    rows = score_clips(clips, FakeUtmos(), FakeWer(), FakeSim(),
                       ref_for_dir=lambda d: None)
    assert rows[0]["wer"] == ""  # no prompt 9 in PROMPT_BY_ID


def test_score_clips_per_metric_failure_is_blank_not_crash():
    class Boom:
        name = "utmos"
        def score(self, wav): raise RuntimeError("boom")
    clips = [Clip("windows-default", "kokoro_cuda_p1.wav", "kokoro", "cuda", "1", "default")]
    rows = score_clips(clips, Boom(), FakeWer(), FakeSim(), ref_for_dir=lambda d: None)
    assert rows[0]["utmos"] == "" and rows[0]["wer"] == "0.0000"
