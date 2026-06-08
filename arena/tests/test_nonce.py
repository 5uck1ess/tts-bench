import time
from arena.nonce import pair_fields, make_nonce, verify_nonce

SECRET = b"test-secret-key"


def test_pair_fields_is_canonical_and_order_sensitive():
    assert pair_fields("default", "aa", "bb", 3) == "default|aa|bb|3"
    assert pair_fields("default", "bb", "aa", 3) != pair_fields("default", "aa", "bb", 3)


def test_make_then_verify_roundtrip():
    fields = pair_fields("cloning", "x1", "y2", 5)
    nonce = make_nonce(SECRET, fields, ts=1000)
    assert verify_nonce(SECRET, nonce, fields, now=1000, max_age_s=600) is True


def test_verify_rejects_tampered_fields():
    fields = pair_fields("cloning", "x1", "y2", 5)
    nonce = make_nonce(SECRET, fields, ts=1000)
    other = pair_fields("cloning", "x1", "y2", 4)  # different prompt
    assert verify_nonce(SECRET, nonce, other, now=1000, max_age_s=600) is False


def test_verify_rejects_expired_and_future():
    fields = pair_fields("default", "a", "b", 1)
    nonce = make_nonce(SECRET, fields, ts=1000)
    assert verify_nonce(SECRET, nonce, fields, now=2000, max_age_s=600) is False  # too old
    assert verify_nonce(SECRET, nonce, fields, now=900, max_age_s=600) is False   # future skew


def test_verify_rejects_malformed():
    fields = pair_fields("default", "a", "b", 1)
    assert verify_nonce(SECRET, "garbage", fields, now=1000, max_age_s=600) is False
    assert verify_nonce(SECRET, "", fields, now=1000, max_age_s=600) is False


def test_make_nonce_unique_per_issuance_even_same_inputs():
    # Same fields + same one-second ts must still yield distinct nonces (random
    # salt), so two honest voters served a hot pair in the same second don't
    # collide into a false replay. Both must verify.
    fields = pair_fields("default", "a", "b", 1)
    n1 = make_nonce(SECRET, fields, ts=1000)
    n2 = make_nonce(SECRET, fields, ts=1000)
    assert n1 != n2
    assert verify_nonce(SECRET, n1, fields, now=1000, max_age_s=600) is True
    assert verify_nonce(SECRET, n2, fields, now=1000, max_age_s=600) is True
