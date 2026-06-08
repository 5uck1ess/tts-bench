import importlib

import pytest

publish = importlib.import_module("publish")


def test_score_lookup_indexes_by_dir_wav(tmp_path, monkeypatch):
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text(
        "dir,wav,model,mode,prompt_id,utmos,wer,sim\n"
        "windows-cloning,f5tts_cuda_p1.wav,f5tts,cloning,1,4.20,0.00,0.97\n",
        encoding="utf-8")
    monkeypatch.setattr(publish, "SCORES_CSV", csv_path)
    look = publish._read_scores_csv()
    row = look[("windows-cloning", "f5tts_cuda_p1.wav")]
    assert row["utmos"] == 4.20 and row["sim"] == 0.97


def test_model_scores_means_over_picked_clips(monkeypatch):
    # Two prompts picked for one model; mean of utmos 4.0 & 5.0 = 4.5.
    look = {
        ("d", "m_cuda_p1.wav"): {"utmos": 4.0, "wer": 0.0, "sim": 0.8},
        ("d", "m_cuda_p2.wav"): {"utmos": 5.0, "wer": 0.0, "sim": 0.9},
    }
    monkeypatch.setattr(publish, "_pick_clip",
                        lambda dirs, model, pid: (f"d/m_cuda_p{pid}.wav", "win", "cuda"))
    agg = publish._model_scores("m", ["1", "2"], ("d",), look)
    assert agg["utmos"] == pytest.approx(4.5) and agg["sim"] == pytest.approx(0.85) and agg["n"] == 2


def test_model_scores_blank_metric_skipped(monkeypatch):
    look = {
        ("d", "m_cuda_p1.wav"): {"utmos": 4.0, "wer": 0.0, "sim": None},
        ("d", "m_cuda_p2.wav"): {"utmos": None, "wer": 0.0, "sim": None},
    }
    monkeypatch.setattr(publish, "_pick_clip",
                        lambda dirs, model, pid: (f"d/m_cuda_p{pid}.wav", "win", "cuda"))
    agg = publish._model_scores("m", ["1", "2"], ("d",), look)
    assert agg["utmos"] == 4.0   # only p1 had utmos
    assert agg["sim"] is None     # none had sim


def test_build_scores_renders_both_lenses(tmp_path, monkeypatch):
    # Minimal worktree: one default dir + one cloning dir, each with a results.csv
    # and one model's clip; a scores.csv covering them.
    # mars5 is a cloning-kind model NOT in NO_PRESET_VOICE, so it lands on the
    # Default board when it has default-dir clips — and its high WER triggers the
    # flagged row style.
    gh = tmp_path / "_gh-pages"
    (gh / "windows-default").mkdir(parents=True)
    (gh / "windows-cloning").mkdir(parents=True)
    for d, mode in (("windows-default", "default"), ("windows-cloning", "cloning")):
        (gh / d / "results.csv").write_text(
            "model,device,prompt_id,ok\n"
            "kokoro,cuda,1,True\n"
            "mars5,cuda,1,True\n",
            encoding="utf-8")
        (gh / d / "kokoro_cuda_p1.wav").write_bytes(b"x")
        (gh / d / "mars5_cuda_p1.wav").write_bytes(b"x")
    (gh / "windows-cloning" / "_reference.wav").write_bytes(b"x")
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "dir,wav,model,mode,prompt_id,utmos,wer,sim\n"
        "windows-default,kokoro_cuda_p1.wav,kokoro,default,1,4.10,0.00,\n"
        "windows-cloning,kokoro_cuda_p1.wav,kokoro,cloning,1,4.00,0.00,0.88\n"
        "windows-default,mars5_cuda_p1.wav,mars5,default,1,3.50,0.97,\n",
        encoding="utf-8")

    monkeypatch.setattr(publish, "WORKTREE", gh)
    monkeypatch.setattr(publish, "SCORES_CSV", scores)
    publish.build_scores()

    html = (gh / "scores.html").read_text(encoding="utf-8")
    assert "scores.html" in html and ">Scores<" in html  # nav tab present
    assert 'data-mode="default"' in html and 'data-mode="cloning"' in html
    assert "UTMOS" in html and "WER" in html and "SIM" in html
    assert 'data-sort="4.10' in html  # default utmos cell carries a numeric data-sort
    assert 'data-sort="0.88' in html  # cloning sim cell present
    assert 'class="flagged"' in html  # the high-WER mars5 row is flagged
