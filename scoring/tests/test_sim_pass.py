from scoring.clips import Clip
from scoring.sim_pass import select_todo, _sibling_cloning_dir


def _row(d, w, mode="cloning", sim=""):
    return {"dir": d, "wav": w, "model": "m", "mode": mode,
            "prompt_id": "1", "utmos": "4.0000", "wer": "0.0000", "sim": sim}


def _models(todo):
    return [c.wav for c, _ in todo]


def test_select_todo_picks_cloning_with_ref_and_blank_sim():
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    existing = {("windows-cloning", "f5tts_cuda_p1.wav"): _row("windows-cloning", "f5tts_cuda_p1.wav")}
    todo = select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav")
    assert _models(todo) == ["f5tts_cuda_p1.wav"]
    assert todo[0][1] == "ref.wav"


def test_select_todo_skips_default_mode_for_preset_model():
    # A real-preset model's default clip is NOT a Chris clone → no SIM.
    clips = [Clip("windows-default", "kokoro_cuda_p1.wav", "kokoro", "cuda", "1", "default")]
    existing = {("windows-default", "kokoro_cuda_p1.wav"): _row("windows-default", "kokoro_cuda_p1.wav", mode="default")}
    assert select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav") == []


def test_select_todo_scores_no_preset_default_clip_against_sibling_cloning_ref():
    # No-preset model benched only in default → its default clip is a bundled Chris
    # clone; score it against the sibling cloning dir's reference.
    clips = [Clip("linux-default", "vibevoice_15b_cuda_p1.wav", "vibevoice_15b", "cuda", "1", "default")]
    existing = {("linux-default", "vibevoice_15b_cuda_p1.wav"):
                _row("linux-default", "vibevoice_15b_cuda_p1.wav", mode="default")}
    ref = lambda d: f"{d}/_reference.wav"
    todo = select_todo(clips, existing, ref_for_dir=ref, no_preset={"vibevoice_15b"})
    assert _models(todo) == ["vibevoice_15b_cuda_p1.wav"]
    assert todo[0][1] == "linux-cloning/_reference.wav"  # sibling cloning dir


def test_select_todo_no_preset_with_cloning_clip_uses_cloning_not_default():
    # If the no-preset model HAS a cloning clip, its default clip is left alone.
    clips = [Clip("linux-default", "echo_cuda_p1.wav", "echo", "cuda", "1", "default"),
             Clip("linux-cloning", "echo_cuda_p1.wav", "echo", "cuda", "1", "cloning")]
    existing = {("linux-default", "echo_cuda_p1.wav"): _row("linux-default", "echo_cuda_p1.wav", mode="default"),
                ("linux-cloning", "echo_cuda_p1.wav"): _row("linux-cloning", "echo_cuda_p1.wav")}
    todo = select_todo(clips, existing, ref_for_dir=lambda d: "ref.wav", no_preset={"echo"})
    assert _models(todo) == ["echo_cuda_p1.wav"]
    assert todo[0][0].mode == "cloning"  # the cloning clip, not the default fallback


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
    clips = [Clip("windows-cloning", "f5tts_cuda_p1.wav", "f5tts", "cuda", "1", "cloning")]
    assert select_todo(clips, {}, ref_for_dir=lambda d: "ref.wav") == []


def test_sibling_cloning_dir():
    assert _sibling_cloning_dir("windows-default") == "windows-cloning"
    assert _sibling_cloning_dir("linux-default") == "linux-cloning"
    assert _sibling_cloning_dir("linux-cloning") is None
