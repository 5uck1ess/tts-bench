from scoring.wer import normalize_fr, wer_value


def test_normalize_fr_lowercases_strips_punct_keeps_accents():
    assert normalize_fr("Bonjour, je m'appelle Cicéro!") == "bonjour je m appelle cicéro"


def test_wer_value_identical_is_zero():
    assert wer_value("open the browser", "open the browser") == 0.0


def test_wer_value_one_substitution():
    # 1 error / 3 ref words
    assert abs(wer_value("open the browser", "open a browser") - (1 / 3)) < 1e-9


def test_wer_value_empty_ref_is_none():
    assert wer_value("", "anything") is None
