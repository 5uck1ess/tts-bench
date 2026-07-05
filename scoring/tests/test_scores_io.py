from scoring.scores_io import FIELDNAMES, read_scores, write_scores, merge_rows


def test_roundtrip(tmp_path):
    p = tmp_path / "scores.csv"
    rows = [
        {"dir": "windows-default", "wav": "kokoro_cuda_p1.wav", "model": "kokoro",
         "mode": "default", "prompt_id": "1", "utmos": "4.1", "wer": "0.0", "sim": ""},
    ]
    write_scores(str(p), rows)
    back = read_scores(str(p))
    assert list(back.keys()) == [("windows-default", "kokoro_cuda_p1.wav")]
    assert back[("windows-default", "kokoro_cuda_p1.wav")]["utmos"] == "4.1"


def test_read_missing_returns_empty(tmp_path):
    assert read_scores(str(tmp_path / "nope.csv")) == {}


def test_merge_keeps_existing_unless_overwrite():
    existing = {("d", "w.wav"): {"dir": "d", "wav": "w.wav", "model": "m",
                                 "mode": "default", "prompt_id": "1",
                                 "utmos": "4.0", "wer": "0.0", "sim": ""}}
    fresh = {"dir": "d", "wav": "w.wav", "model": "m", "mode": "default",
             "prompt_id": "1", "utmos": "9.9", "wer": "0.0", "sim": ""}
    # default: existing wins (skip already-scored)
    merged = merge_rows(existing, [fresh], overwrite=False)
    assert merged[("d", "w.wav")]["utmos"] == "4.0"
    # overwrite (--rescore): fresh wins
    merged2 = merge_rows(existing, [fresh], overwrite=True)
    assert merged2[("d", "w.wav")]["utmos"] == "9.9"


def test_rescore_never_blanks_sim_from_the_wrong_venv():
    """Regression: `score_all --rescore` runs in the py3.11 venv where SIM always
    fails (no fairseq) → fresh rows carry sim="". The overwrite merge must keep
    the existing sim_pass-computed value instead of wiping the board's SIM column."""
    existing = {("d", "w.wav"): {"dir": "d", "wav": "w.wav", "model": "m",
                                 "mode": "cloning", "prompt_id": "1",
                                 "utmos": "4.0", "wer": "0.1", "sim": "0.8000",
                                 "health": "gap"}}
    fresh = {"dir": "d", "wav": "w.wav", "model": "m", "mode": "cloning",
             "prompt_id": "1", "utmos": "4.2", "wer": "0.1", "sim": "",
             "health": ""}
    merged = merge_rows(existing, [fresh], overwrite=True)
    row = merged[("d", "w.wav")]
    assert row["sim"] == "0.8000"    # blank fresh never clobbers a real value
    assert row["utmos"] == "4.2"     # non-blank fresh still wins
    assert row["health"] == ""       # health excluded: "" = clean is a real verdict


def test_write_is_sorted_and_has_header(tmp_path):
    p = tmp_path / "scores.csv"
    rows = [
        {"dir": "b", "wav": "z.wav", "model": "m", "mode": "default",
         "prompt_id": "1", "utmos": "1", "wer": "", "sim": ""},
        {"dir": "a", "wav": "y.wav", "model": "m", "mode": "default",
         "prompt_id": "1", "utmos": "2", "wer": "", "sim": ""},
    ]
    write_scores(str(p), rows)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(FIELDNAMES)
    assert lines[1].startswith("a,y.wav") and lines[2].startswith("b,z.wav")
