import csv
import hashlib
import importlib
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import pytest

from scoring.longform_chunker import MAX_CHARS, chunk, split_in_two
from scoring.longform_es import PASSAGES, es_int_words, es_number_words, es_ordinal_words, spell_digits

spanish_longform = importlib.import_module("scripts.spanish_longform")
report = importlib.import_module("report")

# A stand-in runner speaking the --stdin protocol. MODE selects the behaviour under test.
STUB = textwrap.dedent('''
    import json, sys, time, wave, struct
    mode = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    def wav(path, s):
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
            w.writeframes(struct.pack("<h", 1000) * int(8000 * s))
    if mode == "reject":
        sys.exit(3)
    if mode == "silent":
        time.sleep(60)
    if mode == "partial":
        sys.stdout.write('{"rea'); sys.stdout.flush(); time.sleep(60)
    if mode == "chatty":
        sys.stdout.write('loading weights...\\n{"ready": true}\\n'); sys.stdout.flush()
    else:
        print(json.dumps({"ready": True}), flush=True)
    for i, line in enumerate(sys.stdin):
        job = json.loads(line)
        if mode == "hang" and i >= 1:
            time.sleep(60)
        wav(job["out"], secs)
        print(json.dumps({"ok": True, "run_index": i, "audio_s": secs, "gen_s": secs / 10}), flush=True)
''')


@pytest.fixture
def stub(tmp_path):
    path = tmp_path / "stub_runner.py"
    path.write_text(STUB, encoding="utf-8")

    def argv(mode, secs=1.0):
        return [sys.executable, str(path), mode, str(secs)]
    return argv


ONE = [{"id": "p", "text": "Primera frase del pasaje. Segunda frase del pasaje."}]


# ---------------------------------------------------------------- passages and numbers

def test_passages_are_frozen_public_domain_text():
    assert [p["id"] for p in PASSAGES] == ["pd_galdos", "pd_becquer", "pd_cervantes", "pd_cadiz"]
    for p in PASSAGES:
        assert 100 <= len(p["text"].split()) <= 180, p["id"]
        assert hashlib.sha256(p["text"].encode("utf-8")).hexdigest() == p["sha256"], p["id"]
        assert p["license"].startswith("Public domain"), p["id"]
        assert "wikisource" in p["source"] and "revid" in p["source"], p["id"]
        assert not any(c.isdigit() for c in p["text"]), p["id"]


def test_passages_carry_no_wiki_markup_or_roman_numerals():
    for p in PASSAGES:
        assert not re.search(r"[\[\]{}|<>]", p["text"]), p["id"]
        assert "editar" not in p["text"], p["id"]
        assert not re.search(r"\b[IVXLC]{2,}\b", p["text"]), p["id"]


def test_chunker_invariants():
    text = " ".join(p["text"] for p in PASSAGES)
    chunks = chunk(text)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert " ".join(chunks) == " ".join(text.split())
    long = " ".join(["palabra"] * 300)
    assert len(chunk(long)) > 1 and " ".join(chunk(long)) == long
    assert chunk("   ") == []
    assert split_in_two("uno dos tres, cuatro cinco seis") == ["uno dos tres,", "cuatro cinco seis"]


def test_spanish_number_words():
    cases = {"72": "setenta y dos", "21": "veintiuno", "100": "cien", "101": "ciento uno",
             "1200": "mil doscientos", "1812": "mil ochocientos doce", "21000": "veintiún mil",
             "0.30": "cero coma treinta", "2000000": "dos millones", "1000000000": "mil millones"}
    for digits, words in cases.items():
        assert es_number_words(digits) == words


def test_ordinals_are_spelled_as_ordinals():
    assert " ".join(spell_digits("Artículo 4º").split()) == "Artículo cuarto"
    assert " ".join(spell_digits("la 1ª vez").split()) == "la primera vez"
    assert [es_ordinal_words(n) for n in (11, 13, 17, 18, 21)] == [
        "undécimo", "decimotercero", "decimoséptimo", "decimoctavo", "vigésimo primero"]


def test_large_numbers_never_raise():
    assert " ".join(spell_digits("1.000.000.000").split()) == "mil millones"
    assert es_int_words(10 ** 15).split()[:2] == ["uno", "cero"]
    spell_digits("9" * 40)


def test_digit_hypothesis_scores_like_spelled_reference():
    import jiwer
    from scoring.wer import normalize_intl
    assert jiwer.wer(normalize_intl("setenta y dos vergas"), normalize_intl(spell_digits("72 vergas"))) == 0.0
    assert jiwer.wer(normalize_intl("Artículo cuarto"), normalize_intl(spell_digits("Artículo 4º"))) == 0.0


# ---------------------------------------------------------------- runner protocol

def test_reader_honours_deadline_on_partial_line(stub):
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        spanish_longform.Runner(stub("partial"), ready_timeout=0.3)
    assert time.monotonic() - t0 < 2.0


def test_ready_line_buffered_with_other_output_is_seen(stub):
    r = spanish_longform.Runner(stub("chatty"), ready_timeout=5)
    r.close()


def test_silent_startup_kills_the_child(stub, monkeypatch):
    seen = []
    real = spanish_longform.subprocess.Popen

    def spy(*a, **k):
        p = real(*a, **k)
        seen.append(p)
        return p
    monkeypatch.setattr(spanish_longform.subprocess, "Popen", spy)
    with pytest.raises(TimeoutError):
        spanish_longform.Runner(stub("silent"), ready_timeout=0.3)
    assert seen and seen[0].poll() is not None


# ---------------------------------------------------------------- collection

def test_startup_rejection_and_reply_timeout_become_failure_rows(stub, tmp_path):
    rej = spanish_longform.synthesize("m1", "builtin", stub("reject"), ONE, tmp_path, ready_timeout=5)
    assert rej and all(r["ok"] is False for r in rej) and "exited" in rej[0]["error"]
    hang = spanish_longform.synthesize("m2", "builtin", stub("hang"), ONE, tmp_path, ready_timeout=5,
                                       chunk_timeout=0.5)
    assert hang and hang[-1]["ok"] is False and "TimeoutError" in hang[-1]["error"]


def test_failed_model_keeps_completed_models_in_manifest(stub, tmp_path, monkeypatch):
    plans = {"good": stub("ok"), "bad": stub("reject")}
    monkeypatch.setattr(spanish_longform, "runner_argv", lambda m, v: plans[m])
    monkeypatch.setattr(spanish_longform, "MODELS", {m: (None, None, [], None, "builtin", None) for m in plans})
    spanish_longform.main(["--models", "good,bad", "--passages", "pd_galdos", "--out", str(tmp_path / "o")])
    rows = list(csv.DictReader(open(tmp_path / "o" / spanish_longform.MANIFEST, encoding="utf-8")))
    assert {r["model"] for r in rows} == {"good", "bad"}
    assert all(r["ok"] == "True" for r in rows if r["model"] == "good")
    assert all(r["ok"] == "False" for r in rows if r["model"] == "bad")


def test_relative_out_is_resolved_for_the_runner(stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spanish_longform, "runner_argv", lambda m, v: stub("ok"))
    monkeypatch.setattr(spanish_longform, "MODELS", {"good": (None, None, [], None, "builtin", None)})
    spanish_longform.main(["--models", "good", "--passages", "pd_galdos", "--out", "rel/out"])
    rows = list(csv.DictReader(open(tmp_path / "rel" / "out" / spanish_longform.MANIFEST, encoding="utf-8")))
    assert rows and all(r["wav"].startswith(str(tmp_path.resolve())) for r in rows)
    assert all(r["ok"] == "True" for r in rows)


def test_overlong_unsplittable_audio_is_never_scored(stub, tmp_path):
    rows = spanish_longform.synthesize("m", "builtin", stub("ok", 31), [{"id": "p", "text": "Artículo primero. La"}],
                                       tmp_path, ready_timeout=5)
    assert rows and all(r["ok"] is False for r in rows)
    assert all("cannot be split" in r["error"] for r in rows)


def test_unsupported_voice_override_is_rejected():
    with pytest.raises(SystemExit):
        spanish_longform.runner_argv("reverb", "typo")
    with pytest.raises(SystemExit):
        spanish_longform.runner_argv("nope", "builtin")
    assert spanish_longform.runner_argv("reverb", "builtin")[-1] == "cpu"


# ---------------------------------------------------------------- scoring and reports

class _Wer:
    def __init__(self, hyps):
        self.hyps = iter(hyps)

    def transcribe(self, wav, lang):
        h = next(self.hyps)
        if isinstance(h, Exception):
            raise h
        return h


class _Health:
    def score(self, wav):
        return ""


def test_one_bad_hypothesis_does_not_stop_scoring(stub, tmp_path):
    rows = spanish_longform.synthesize("m", "builtin", stub("ok"), [{"id": "p", "text": "Uno. Dos."}], tmp_path,
                                       ready_timeout=5)
    rows = rows + [dict(rows[0], chunk_seq=8), dict(rows[0], chunk_seq=9)]
    path = tmp_path / spanish_longform.MANIFEST
    spanish_longform.write_rows(path, rows)
    spanish_longform.score(path, wer=_Wer(["Uno. Dos.", ValueError("boom"), "1" * 40]), health=_Health())
    got = list(csv.DictReader(open(path, encoding="utf-8")))
    assert got[0]["wer"] == "0.0000"
    assert got[1]["wer"] == "" and "boom" in got[1]["error"]
    assert got[2]["wer"] != ""


def test_longform_output_does_not_break_report_index(tmp_path, monkeypatch):
    ordinary = tmp_path / "2026-11-01_0000"
    ordinary.mkdir()
    (ordinary / "results.csv").write_text(
        "prompt_id,model,device,variant,can_clone,run_index,is_cold,ttfa_ms,gen_s,audio_s,rtf,peak_mem_mb,"
        "peak_vram_mb,wall_s,ok,error\n1,kokoro,cpu,,False,0,True,10,0.1,1.0,10,1,,0.2,True,\n", encoding="utf-8")
    es = tmp_path / "es-longform"
    es.mkdir()
    spanish_longform.write_rows(es / spanish_longform.MANIFEST, [
        {"model": "a", "voice": "v", "passage_id": "p", "chunk_seq": 0, "ok": True, "audio_s": "1", "gen_s": "1"}])
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    idx = report.build_index()
    assert "2026-11-01_0000" in idx.read_text(encoding="utf-8")


def test_render_table_from_csv(tmp_path):
    path = tmp_path / "es_longform.csv"
    rows = [
        {"model": "a", "voice": "v1", "passage_id": "p", "chunk_seq": 0, "ok": True, "audio_s": "10.0",
         "gen_s": "2.0", "edits": 1, "ref_words": 20, "wer": "0.05", "health_flags": ""},
        {"model": "a", "voice": "v1", "passage_id": "p", "chunk_seq": 1, "ok": True, "audio_s": "10.0",
         "gen_s": "2.0", "edits": 0, "ref_words": 20, "wer": "0.0", "health_flags": "gap"},
        {"model": "b", "voice": "builtin", "passage_id": "p", "chunk_seq": 0, "ok": False, "error": "x"},
    ]
    spanish_longform.write_rows(path, rows)
    table = spanish_longform.render_table(path).splitlines()
    assert table[2] == "| a | v1 | 2 | 0 | 0.0250 | 1 | 20.0 | 5.00 |"
    assert table[3] == "| b | builtin | 1 | 1 | n/a | 0 | 0.0 | n/a |"


def test_manifest_is_not_named_results_csv():
    assert spanish_longform.MANIFEST != "results.csv"
    assert json.dumps(spanish_longform.MANIFEST)


# ---------------------------------------------------------------- round-1 review fixes

def test_legacy_spanish_results_csv_is_skipped_by_report(tmp_path, monkeypatch):
    ordinary = tmp_path / "2026-11-01_0000"
    ordinary.mkdir()
    (ordinary / "results.csv").write_text(
        "prompt_id,model,device,variant,can_clone,run_index,is_cold,ttfa_ms,gen_s,audio_s,rtf,peak_mem_mb,"
        "peak_vram_mb,wall_s,ok,error\n1,kokoro,cpu,,False,0,True,10,0.1,1.0,10,1,,0.2,True,\n", encoding="utf-8")
    legacy = tmp_path / "2026-11-22_es-longform"
    legacy.mkdir()
    with open(legacy / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=spanish_longform.FIELDS)
        w.writeheader()
        w.writerow({"model": "reverb", "voice": "builtin", "passage_id": "p", "chunk_seq": 0, "ok": True})
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    assert "2026-11-01_0000" in report.build_index().read_text(encoding="utf-8")
    assert report._is_bench_csv(ordinary / "results.csv") and not report._is_bench_csv(legacy / "results.csv")


def test_recorded_voice_must_match_final_argv():
    good = spanish_longform.runner_argv("supertonic", "M5")
    spanish_longform.check_argv_voice("supertonic", "M5", good)
    stripped = [good[0], *[a for a in good[1:] if a not in ("--reference", "M5")]]
    with pytest.raises(SystemExit):
        spanish_longform.check_argv_voice("supertonic", "M5", stripped)
    with pytest.raises(SystemExit):
        spanish_longform.runner_argv("supertonic", "builtin")
    with pytest.raises(SystemExit):
        spanish_longform.check_argv_voice("reverb", "x", spanish_longform.runner_argv("reverb", "builtin"))


def test_partial_rerun_keeps_other_passages(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(spanish_longform, "runner_argv", lambda m, v: stub("ok"))
    monkeypatch.setattr(spanish_longform, "check_argv_voice", lambda m, v, a: None)
    monkeypatch.setattr(spanish_longform, "MODELS", {"good": (None, None, [], None, "builtin", None)})
    out = tmp_path / "o"
    spanish_longform.main(["--models", "good", "--passages", "pd_galdos,pd_cadiz", "--out", str(out)])
    spanish_longform.main(["--models", "good", "--passages", "pd_galdos", "--out", str(out)])
    rows = list(csv.DictReader(open(out / spanish_longform.MANIFEST, encoding="utf-8")))
    assert {r["passage_id"] for r in rows} == {"pd_galdos", "pd_cadiz"}
    assert all(Path(r["wav"]).exists() for r in rows)
    keys = [(r["passage_id"], r["chunk_seq"]) for r in rows]
    assert len(keys) == len(set(keys))


def test_score_refuses_a_foreign_csv_without_writing(tmp_path):
    path = tmp_path / "results.csv"
    original = "prompt_id,model,device,ok\n1,kokoro,cpu,True\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(SystemExit):
        spanish_longform.score(path, wer=_Wer([]), health=_Health())
    assert path.read_text(encoding="utf-8") == original


def test_rescoring_clears_a_stale_score_error(stub, tmp_path):
    rows = spanish_longform.synthesize("m", "builtin", stub("ok"), [{"id": "p", "text": "Uno. Dos."}], tmp_path,
                                       ready_timeout=5)
    path = tmp_path / spanish_longform.MANIFEST
    spanish_longform.write_rows(path, [dict(rows[0], error="score: ValueError: old")])
    spanish_longform.score(path, wer=_Wer(["Uno. Dos."]), health=_Health())
    got = list(csv.DictReader(open(path, encoding="utf-8")))
    assert got[0]["error"] == "" and got[0]["wer"] == "0.0000"


def test_decimals_after_thousands_groups_and_degree_sign():
    assert " ".join(spell_digits("1.200,50").split()) == "mil doscientos coma cincuenta"
    assert " ".join(spell_digits("1,200.50").split()) == "mil doscientos coma cincuenta"
    assert "trigésimo" not in spell_digits("30°")


# ---------------------------------------------------------------- round-2 review fixes

def test_builtin_label_rejects_a_reference_in_exact_argv():
    argv = spanish_longform.runner_argv("reverb", "builtin") + ["--reference", "reference/jo.wav"]
    with pytest.raises(SystemExit):
        spanish_longform.check_argv_voice("reverb", "builtin", argv)


def test_scoring_never_transcribes_an_overlong_legacy_take(stub, tmp_path):
    rows = spanish_longform.synthesize("m", "builtin", stub("ok", 31), [{"id": "p", "text": "Artículo primero. La"}],
                                       tmp_path, ready_timeout=5)
    wav = Path(rows[0]["wav"])
    wav.with_suffix(".txt").write_text("Artículo primero. La", encoding="utf-8")
    path = tmp_path / spanish_longform.MANIFEST
    spanish_longform.write_rows(path, [dict(rows[0], ok=True, error="", audio_s="31.000", gen_s="3.1")])
    spanish_longform.score(path, wer=_Wer([]), health=_Health())  # an empty _Wer raises if transcribe is called
    got = list(csv.DictReader(open(path, encoding="utf-8")))
    assert got[0]["wer"] == "" and "left unscored" in got[0]["error"]
    assert "| 1 | 1 | n/a |" in spanish_longform.render_table(path)


def test_publish_discovery_skips_legacy_spanish_results_csv(tmp_path):
    publish = importlib.import_module("publish")
    legacy = tmp_path / "es"
    legacy.mkdir()
    spanish_longform.write_rows(legacy / "results.csv", [{"model": "m", "voice": "v", "passage_id": "p",
                                                          "chunk_seq": 0, "ok": True}])
    assert not publish._is_bench_csv(legacy / "results.csv")
