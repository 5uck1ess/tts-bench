from scripts.hosted.catalog import CATALOG, MODEL_ID, label
from scripts.hosted.providers import ADAPTERS


def test_model_id_matches_what_the_adapters_send():
    """The Scores page can't import the adapters (Linux has no hosted venv), so catalog.py
    carries a copy of each model id. It must stay equal to what we actually call."""
    assert MODEL_ID == {s: ADAPTERS[s].model for s in CATALOG}


def test_label_names_the_model_not_just_the_company():
    assert label("gradium_tts") == "Gradium · default"
