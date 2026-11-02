from scripts.hosted.catalog import CATALOG, MODEL_ID, MODEL_NOTE, label
from scripts.hosted.providers import ADAPTERS


def test_model_id_matches_what_the_adapters_send():
    """The Scores page can't import the adapters (Linux has no hosted venv), so catalog.py
    carries a copy of each model id. It must stay equal to what we actually call."""
    assert MODEL_ID == {s: ADAPTERS[s].model for s in CATALOG}


def test_label_names_the_model_not_just_the_company():
    assert label("rime_coda") == "Rime Coda · coda"


def test_unpinned_aliases_say_the_vendor_picks():
    """Gradium ("default") and Palabra ("auto") take no pinned model id, so the board must not
    present the alias as if it were a model."""
    assert label("gradium_tts") == "Gradium · default · vendor picks"
    assert label("palabra_tts") == "Palabra TTS · auto · vendor picks"
    assert set(MODEL_NOTE) <= set(CATALOG)
