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


from scoring.wer import head_dropped


def test_head_dropped_flags_missing_opening_words():
    # Ming p1 on the truncated reference transcript: only the last word survived.
    assert head_dropped("open the browser and read my email", "email")
    # Cloner that ate the first ~3 s of prompt 2.
    assert head_dropped("ill start a new git branch push the changes and open a pull request",
                        "push the changes and open a pull request")


def test_head_dropped_ignores_misheard_first_word_and_garbage():
    assert not head_dropped("run pytest test slash test", "ron pytest test slash test")
    # Opening spoken but run together by Whisper (prompt 4): misheard, not dropped.
    assert not head_dropped("run pytest test slash test underscore voice dot py with verbose flag",
                            "were on paytesttest slash test underscore voice dot py with verbose flag")
    assert not head_dropped("open the browser and read my email", "open the browser and read my email")
    # Matches nothing at all: garbage, WER already says so.
    assert not head_dropped("open the browser and read my email", "female")
    # Leaked reference tail BEFORE an intact prompt is not a dropped head.
    assert not head_dropped("open the browser", "youre amazing open the browser")
    assert not head_dropped("", "x") and not head_dropped("a b c", "")
