import csv
import hashlib
import importlib

from scoring.longform_chunker import MAX_CHARS, chunk, split_in_two
from scoring.longform_es import PASSAGES, es_number_words, spell_digits

spanish_longform = importlib.import_module("scripts.spanish_longform")


def test_passages_are_frozen_public_domain_text():
    assert [p["id"] for p in PASSAGES] == ["pd_galdos", "pd_becquer", "pd_cervantes", "pd_cadiz"]
    for p in PASSAGES:
        assert 100 <= len(p["text"].split()) <= 180, p["id"]
        assert hashlib.sha256(p["text"].encode("utf-8")).hexdigest() == p["sha256"], p["id"]
        assert p["license"].startswith("Public domain"), p["id"]
        assert "wikisource" in p["source"] and "revid" in p["source"], p["id"]
        assert not any(c.isdigit() for c in p["text"]), p["id"]


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
             "0.30": "cero coma treinta", "2000000": "dos millones"}
    for digits, words in cases.items():
        assert es_number_words(digits) == words


def test_digit_hypothesis_scores_like_spelled_reference():
    import jiwer
    from scoring.wer import normalize_intl
    ref = normalize_intl("setenta y dos vergas")
    hyp = normalize_intl(spell_digits("72 vergas"))
    assert jiwer.wer(ref, hyp) == 0.0


def test_render_table_from_csv(tmp_path):
    path = tmp_path / "results.csv"
    rows = [
        {"model": "a", "voice": "v1", "passage_id": "p", "chunk_seq": 0, "ok": True, "audio_s": "10.0",
         "gen_s": "2.0", "edits": 1, "ref_words": 20, "wer": "0.05", "health_flags": ""},
        {"model": "a", "voice": "v1", "passage_id": "p", "chunk_seq": 1, "ok": True, "audio_s": "10.0",
         "gen_s": "2.0", "edits": 0, "ref_words": 20, "wer": "0.0", "health_flags": "gap"},
        {"model": "b", "voice": "builtin", "passage_id": "p", "chunk_seq": 0, "ok": False, "error": "x"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=spanish_longform.FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    table = spanish_longform.render_table(path).splitlines()
    assert table[2] == "| a | v1 | 2 | 0 | 0.0250 | 1 | 20.0 | 5.00 |"
    assert table[3] == "| b | builtin | 1 | 1 | n/a | 0 | 0.0 | n/a |"
