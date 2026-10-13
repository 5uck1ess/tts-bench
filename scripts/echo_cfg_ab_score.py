"""Score echo_cfg_ab clips with the board's WER + UTMOS scorers; SIM runs on Linux.

venvs/scoring/Scripts/python scripts/echo_cfg_ab_score.py DIR
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scoring.prompts import PROMPT_BY_ID  # noqa: E402
from scoring.utmos import UtmosScorer  # noqa: E402
from scoring.wer import WerScorer  # noqa: E402

d = Path(sys.argv[1])
utmos, wer = UtmosScorer(), WerScorer()
with open(d / "scores.jsonl", "w", encoding="utf-8") as f:
    for line in open(d / "runs.jsonl", encoding="utf-8"):
        r = json.loads(line)
        wav = str(d / r["wav"])
        lang, text = PROMPT_BY_ID[str(r["prompt"])]
        r["utmos"] = utmos.score(wav)
        r["wer"] = wer.score(wav, text, lang)
        f.write(json.dumps(r) + "\n")
        print(r["wav"], round(r["utmos"], 3), round(r["wer"], 3), flush=True)
