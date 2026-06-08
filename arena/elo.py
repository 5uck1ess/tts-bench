"""Elo for blind pairwise votes. Ported math from naq_lab/vote.py (no NAQ code)."""

from typing import Iterable

K_FACTOR = 24.0
BASE_ELO = 1000.0


def expected(ra: float, rb: float) -> float:
    """Expected score of A vs B (logistic, 400-point divisor)."""
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def apply_vote(elo: dict, games: dict, left: str, right: str, choice: str) -> None:
    """Mutate ``elo``/``games`` for one resolved vote.

    ``choice`` in {"left","right","tie"}. Callers must NOT pass "bad" (it is
    excluded from Elo). Unknown models are skipped silently.
    """
    if left not in elo or right not in elo:
        return
    ea = expected(elo[left], elo[right])
    eb = 1.0 - ea
    sa = 1.0 if choice == "left" else 0.0 if choice == "right" else 0.5
    elo[left] += K_FACTOR * (sa - ea)
    elo[right] += K_FACTOR * ((1.0 - sa) - eb)
    games[left] += 1
    games[right] += 1


def derive(models: Iterable[str],
           votes: Iterable[tuple]) -> tuple[dict, dict]:
    """Rebuild (elo, games) from scratch by replaying ``votes`` in order.

    ``votes`` items are ``(left_model, right_model, choice)``. "bad" choices
    and votes referencing a model not in ``models`` are skipped.
    """
    elo = {m: BASE_ELO for m in models}
    games = {m: 0 for m in models}
    for left, right, choice in votes:
        if choice == "bad":
            continue
        apply_vote(elo, games, left, right, choice)
    return elo, games
