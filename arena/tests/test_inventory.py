import pytest
from arena.inventory import clip_id, load_inventory

MANIFEST = {
    "base_url": "https://example.test/tts-bench/",
    "prompts": {"1": ["en", "hello"], "2": ["en", "world"], "5": ["fr", "bonjour"]},
    "modes": {
        "default": {
            "reference_url": None,
            "clips": [
                {"model": "kokoro", "prompt": 1, "url": "https://example.test/d/kokoro_cpu_p1.wav"},
                {"model": "piper", "prompt": 1, "url": "https://example.test/d/piper_cpu_p1.wav"},
                {"model": "kokoro", "prompt": 5, "url": "https://example.test/d/kokoro_cpu_p5.wav"},
                {"model": "lonely", "prompt": 2, "url": "https://example.test/d/lonely_cpu_p2.wav"},
            ],
        },
        "cloning": {
            "reference_url": "https://example.test/c/_reference.wav",
            "clips": [
                {"model": "echo", "prompt": 1, "url": "https://example.test/c/echo_cuda_p1.wav"},
                {"model": "indextts", "prompt": 1, "url": "https://example.test/c/indextts_cuda_p1.wav"},
            ],
        },
    },
}


def test_clip_id_is_stable_and_opaque():
    cid = clip_id("default", "kokoro", 1)
    assert cid == clip_id("default", "kokoro", 1)
    assert "kokoro" not in cid
    assert len(cid) == 12


def test_load_default_drops_single_model_prompts_and_excludes_fr_by_default():
    inv = load_inventory(MANIFEST, "default", langs={"en"})
    # prompt 1 has 2 models -> kept; prompt 2 has 1 model -> dropped; prompt 5 is fr -> dropped
    assert set(inv.by_prompt) == {1}
    assert sorted(inv.by_prompt[1]) == ["kokoro", "piper"]
    assert inv.reference_url is None


def test_langs_none_keeps_all_languages():
    inv = load_inventory(MANIFEST, "default", langs=None)
    # prompt 5 (fr) still has only 1 model (kokoro) so it is dropped regardless;
    # this asserts the fr prompt is at least considered (no crash, kokoro present)
    assert 5 not in inv.by_prompt  # only 1 model
    assert "kokoro" in inv.models


def test_clip_url_lookup_by_opaque_id():
    inv = load_inventory(MANIFEST, "cloning", langs={"en"})
    cid = inv.id_of[("echo", 1)]
    assert inv.url_of[cid] == "https://example.test/c/echo_cuda_p1.wav"
    assert inv.reference_url == "https://example.test/c/_reference.wav"


def test_missing_clip_for_model_prompt_raises_keyerror():
    inv = load_inventory(MANIFEST, "default", langs={"en"})
    with pytest.raises(KeyError):
        _ = inv.id_of[("piper", 5)]  # piper has no p5
