"""SIM pass for echo_cfg_ab clips (Linux scoring_sim venv). Adds "sim" to scores.jsonl.

venvs/scoring_sim/bin/python scripts/echo_cfg_ab_sim.py DIR
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scoring.sim import SimScorer  # noqa: E402

d = Path(sys.argv[1])
ref = str(REPO / "reference" / "chris_hemsworth_15s.wav")
sim = SimScorer()
rows = [json.loads(l) for l in open(d / "scores.jsonl", encoding="utf-8")]
for r in rows:
    r["sim"] = sim.score(str(d / r["wav"]), ref)
    print(r["wav"], round(r["sim"], 3), flush=True)
with open(d / "scores_sim.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
