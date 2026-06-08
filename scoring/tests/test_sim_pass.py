from scoring.clips import Clip
from scoring.sim_pass import select_todo


def _row(d, w, sim=""):
    return {"dir": d, "wav": w, "model": "m", "mode": "cloning",
            "prompt_id": "1", "utmos": "4.0000", "wer": "0.0000", "sim": sim}


def test_select_todo_picks_cloning_with_ref_and_blank_sim():
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    existing = {("windows-cloning", "f5tts_cuda_p1.wav"): _row("windows-cloning", "f5tts_cuda_p1.wav")}
    todo = select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav")
    assert [c.wav for c in todo] == ["f5tts_cuda_p1.wav"]


def test_select_todo_skips_default_mode():
    clips = [Clip("windows-default", "kokoro_cuda_p1.wav", "kokoro", "cuda", "1", "default")]
    existing = {("windows-default", "kokoro_cuda_p1.wav"): _row("windows-default", "kokoro_cuda_p1.wav")}
    assert select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav") == []


def test_select_todo_skips_when_no_reference():
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    existing = {("windows-cloning", "f5tts_cuda_p1.wav"): _row("windows-cloning", "f5tts_cuda_p1.wav")}
    assert select_todo(clips, existing, ref_for_dir=lambda d: None) == []


def test_select_todo_skips_already_scored_unless_rescore():
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    existing = {("windows-cloning", "f5tts_cuda_p1.wav"):
                _row("windows-cloning", "f5tts_cuda_p1.wav", sim="0.8000")}
    assert select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav") == []
    assert len(select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav", rescore=True)) == 1


def test_select_todo_skips_clip_without_row():
    # score_all owns row creation; a clip not yet in scores.csv is not SIM's job.
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    assert select_todo(clips, {}, ref_for_dir=lambda d: "ref.wav") == []
