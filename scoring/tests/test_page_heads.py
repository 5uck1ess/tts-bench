"""Every generated page must carry a viewport meta: without it a phone renders the page at
desktop width and shrinks it until nothing is readable (speed.html and archive.html shipped
that way until 2026-11-02)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VP = '<meta name="viewport" content="width=device-width, initial-scale=1">'


def test_every_page_head_has_a_viewport_meta():
    missing = []
    for name in ("publish.py", "report.py"):
        for i, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"<head>\s*<meta charset", line) and VP not in line:
                missing.append(f"{name}:{i}")
    assert not missing, f"page head without a viewport meta: {missing}"
