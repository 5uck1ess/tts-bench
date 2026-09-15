from scoring.prompts import PROMPT_BY_ID
from scoring.wer import normalize_fr, normalize_intl, wer_value


def test_normalize_fr_lowercases_strips_punct_keeps_accents():
    assert normalize_fr("Bonjour, je m'appelle Cicéro!") == "bonjour je m appelle cicéro"


def test_normalize_fr_is_the_old_name_for_normalize_intl():
    assert normalize_fr is normalize_intl


def test_normalize_intl_keeps_spanish_letters_and_drops_inverted_punctuation():
    """ñ/ü/accents must survive or WER counts correct words as errors; ¿ and ¡
    must not, or they attach to a word and every Spanish clip scores a miss."""
    got = normalize_intl("¿Está bilingüe el señor? ¡Sí!")
    assert got == "está bilingüe el señor sí"


def test_spanish_prompt_normalizes_to_plain_words():
    """The real prompt 6 round-trips to bare words — no stray punctuation tokens."""
    _lang, text = PROMPT_BY_ID["6"]
    words = normalize_intl(text).split()
    assert all(w.isalpha() for w in words), words
    assert "bilingüe" in words and "añade" in words and "español" in words


def test_wer_value_identical_is_zero():
    assert wer_value("open the browser", "open the browser") == 0.0


def test_wer_value_one_substitution():
    # 1 error / 3 ref words
    assert abs(wer_value("open the browser", "open a browser") - (1 / 3)) < 1e-9


def test_wer_value_empty_ref_is_none():
    assert wer_value("", "anything") is None
