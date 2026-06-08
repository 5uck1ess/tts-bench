"""Informative pair selection. Ported weighting from naq_lab/vote.py (no NAQ code).

Pick a prompt, then model A biased toward under-sampled, then B biased toward
Elo-close and rarely-paired-with-A. Sides are randomized. RNG is injected for
deterministic tests.
"""

import random

CLOSENESS_SCALE = 200.0  # Elo gap at which the closeness weight halves


def wchoice(items, weights: dict, rng: random.Random):
    """Weighted choice. ``weights`` maps item -> weight; floored at 1e-9."""
    ws = [max(weights.get(x, 0.0), 1e-9) for x in items]
    r = rng.random() * sum(ws)
    upto = 0.0
    for x, w in zip(items, ws):
        upto += w
        if upto >= r:
            return x
    return items[-1]


def choose_pair(by_prompt: dict, elo: dict, games: dict,
                pair_count: dict, rng: random.Random) -> tuple:
    """Return ``(prompt_id, left_model, right_model)`` for an informative pair.

    ``by_prompt`` maps prompt_id -> list of >=2 models with a clip for it.
    ``pair_count`` maps ``frozenset({m1, m2})`` -> times that pair was shown.
    """
    prompt = rng.choice(list(by_prompt))
    ms = by_prompt[prompt]
    a = wchoice(ms, {m: 1.0 / (games[m] + 1) for m in ms}, rng)
    others = [m for m in ms if m != a]

    def bweight(m):
        close = 1.0 / (1.0 + abs(elo[a] - elo[m]) / CLOSENESS_SCALE)
        rare = 1.0 / (pair_count.get(frozenset((a, m)), 0) + 1)
        return close * rare

    b = wchoice(others, {m: bweight(m) for m in others}, rng)
    left, right = rng.sample([a, b], 2)
    return prompt, left, right
