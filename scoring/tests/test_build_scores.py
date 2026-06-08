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
