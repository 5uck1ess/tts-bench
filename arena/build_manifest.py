"""LOCAL build tool: scan _gh-pages canonicals -> arena/clips_manifest.json.

Replicates naq_lab/vote.py build_inventory selection rules (rig/device priority,
NO_PRESET_VOICE drop, Mac-cloning exclusion, cloning->default fallback) but emits
gh-pages URLs instead of local paths. Run locally after publishing; the committed
manifest is what the Space loads. Imports scoring.prompts (public) — never naq_lab.
"""

import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GH = os.path.join(REPO, "_gh-pages")

RIG_PRIO = {"windows": 0, "linux": 1, "mac": 2}
DEV_PRIO = {"cuda": 0, "mps": 1, "cpu": 2}

# Mirrors naq_lab/vote.py NO_PRESET_VOICE and publish.py: zero-shot models whose
# no-reference "default" run is actually a Chris clone -> excluded from default lens.
NO_PRESET_VOICE = {
    "moss_tts", "moss_tts_v15", "moss_tts_nano", "fish_15", "fish_s2", "metavoice",
    "openvoice", "zipvoice", "zonos", "vibevoice_15b", "vibevoice_7b", "echo", "dots_tts",
}

_WAV_RE = re.compile(r"(.+)_(cuda|mps|cpu)_p(\d+)\.wav$")


def _base_url() -> str:
    """gh-pages base from the git remote (https://<user>.github.io/<repo>/)."""
    import subprocess
    url = subprocess.check_output(
        ["git", "-C", REPO, "remote", "get-url", "origin"], text=True).strip()
    s = url[len("https://"):] if url.startswith("https://") else url
    s = s[len("git@"):].replace(":", "/") if s.startswith("git@") else s
    parts = s.replace(".git", "").split("/")
    return f"https://{parts[1]}.github.io/{parts[2]}/"


def scan_dirs(gh_root, mode: str, base_url: str):
    """Return ``(clips, reference_url)`` for ``mode``.

    ``clips`` is a list of ``{"model","prompt","url"}`` (best rig/device per
    (model, prompt)). ``reference_url`` is the cloning target wav (or None).
    Pure over the filesystem under ``gh_root`` — no network.
    """
    gh_root = str(gh_root)
    found = {}   # (model, prompt) -> (rank, url)
    drop = NO_PRESET_VOICE if mode == "default" else set()

    def scan(glob_pat, only=None):
        for d in glob.glob(glob_pat):
            rig = os.path.basename(d).split("-")[0]
            if rig not in RIG_PRIO:
                continue
            if mode == "cloning" and rig == "mac":
                continue
            for w in glob.glob(os.path.join(d, "*.wav")):
                fn = os.path.basename(w)
                m = _WAV_RE.match(fn)
                if not m:
                    continue
                model, dev, p = m.group(1), m.group(2), int(m.group(3))
                if model in drop or (only is not None and model not in only):
                    continue
                rank = (RIG_PRIO[rig], DEV_PRIO.get(dev, 9))
                key = (model, p)
                url = base_url + os.path.basename(d) + "/" + fn
                if key not in found or rank < found[key][0]:
                    found[key] = (rank, url)

    scan(os.path.join(gh_root, f"*-{mode}"))
    if mode == "cloning":
        missing = NO_PRESET_VOICE - {model for (model, _) in found}
        if missing:
            scan(os.path.join(gh_root, "*-default"), only=missing)

    reference_url = None
    if mode == "cloning":
        for d in sorted(glob.glob(os.path.join(gh_root, "*-cloning")),
                        key=lambda p: RIG_PRIO.get(os.path.basename(p).split("-")[0], 9)):
            cand = os.path.join(d, "_reference.wav")
            if os.path.exists(cand):
                reference_url = base_url + os.path.basename(d) + "/_reference.wav"
                break

    clips = [{"model": m, "prompt": p, "url": url}
             for (m, p), (_, url) in sorted(found.items())]
    return clips, reference_url


def build_manifest(gh_root, base_url, prompts: dict) -> dict:
    modes = {}
    for mode in ("default", "cloning"):
        clips, ref = scan_dirs(gh_root, mode, base_url)
        modes[mode] = {"reference_url": ref, "clips": clips}
    return {"base_url": base_url, "prompts": prompts, "modes": modes}


def main():
    ap = argparse.ArgumentParser(description="Build arena/clips_manifest.json from _gh-pages.")
    ap.add_argument("--gh-root", default=GH)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "clips_manifest.json"))
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from scoring.prompts import PROMPT_BY_ID  # {str(pid): (lang, text)} — public
    prompts = {pid: [lang, text] for pid, (lang, text) in PROMPT_BY_ID.items()}

    manifest = build_manifest(args.gh_root, _base_url(), prompts)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    n = {m: len(b["clips"]) for m, b in manifest["modes"].items()}
    print(f"wrote {args.out}  clips={n}  base={manifest['base_url']}")


if __name__ == "__main__":
    main()
