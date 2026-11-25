#!/usr/bin/env python3
"""Sentence-boundary text chunker for long-form TTS (shared by the bake-off, the audiobook synthesis and
the tts-bench Spanish long-form script).

chunk(text, max_chars=240) splits at sentence ends (. ! ? … ; :), then at commas, then at the last space
before the limit. Joining the chunks with single spaces reproduces the whitespace-normalized input.
"""

import re
import sys

MAX_CHARS = 240
_SENT = re.compile(r"(?<=[.!?…;:])\s+")
_COMMA = re.compile(r"(?<=,)\s+")


def _norm(text):
    return re.sub(r"\s+", " ", text).strip()


def _split_hard(piece, max_chars):
    out = []
    while len(piece) > max_chars:
        cut = piece.rfind(" ", 0, max_chars + 1)
        if cut <= 0:
            cut = max_chars
        out.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        out.append(piece)
    return out


def _pack(parts, max_chars):
    out, cur = [], ""
    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = f"{cur} {p}"
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def chunk(text, max_chars=MAX_CHARS):
    text = _norm(text)
    if not text:
        return []
    pieces = []
    for sent in _SENT.split(text):
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        for part in _pack(_COMMA.split(sent), max_chars):
            pieces.extend(_split_hard(part, max_chars) if len(part) > max_chars else [part])
    return _pack(pieces, max_chars)


def split_in_two(text):
    """Split one chunk at its middle comma, else its middle space (used when audio runs over 28 s)."""
    mid = len(text) // 2
    commas = [m.end() for m in re.finditer(r",\s", text)]
    spaces = [m.start() for m in re.finditer(r"\s", text)]
    cands = commas or spaces
    if not cands:
        return [text]
    cut = min(cands, key=lambda i: abs(i - mid))
    a, b = text[:cut].strip(), text[cut:].strip()
    return [x for x in (a, b) if x]


def _self_test():
    fails = 0
    sample = ("Primero: invocar el auxilio divino. Segundo: consultar con inteligentes y fieles; "
              "tercero, observar con prontitud, resolución y disposición. ") * 6
    cs = chunk(sample)
    if any(len(c) > MAX_CHARS for c in cs):
        print("FAIL chunk over max_chars"); fails += 1
    if " ".join(cs) != _norm(sample):
        print("FAIL join != normalized input"); fails += 1
    long = " ".join(["palabra"] * 300)
    cl = chunk(long)
    if len(cl) < 2 or any(len(c) > MAX_CHARS for c in cl) or " ".join(cl) != long:
        print("FAIL long unpunctuated sentence"); fails += 1
    if chunk("   ") != []:
        print("FAIL empty"); fails += 1
    a = split_in_two("uno dos tres, cuatro cinco seis siete")
    if len(a) != 2 or " ".join(a) != "uno dos tres, cuatro cinco seis siete":
        print(f"FAIL split_in_two {a}"); fails += 1
    print("CHUNKER SELF-TEST", "FAIL" if fails else "OK")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    for c in chunk(sys.stdin.read()):
        print(c)
