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
    "openvoice", "zipvoice", "zonos", "zonos2", "vibevoice_15b", "vibevoice_7b", "echo",
    "breeze_tts2",
    # dots_tts is NOT here: it has a real (if per-prompt varying) default voice and
    # shows on the Default board with a badge. Mirrors publish.py -- keep in sync.
    "cosyvoice",  # pure zero-shot cloning, no model-native preset (mirrors publish.py)
    "miotts_01b", "miotts_06b",  # MioTTS: pure zero-shot cloner, cloning board only
    "wavtts",  # WavTTS: pure zero-shot cloner (ref wav + .txt), cloning board only
    # Qwen3-TTS *Base* (both runtimes): reference-only, no model-native preset voice.
    # CustomVoice (qwentts_06b_custom) is the Qwen row with real preset timbres.
    "qwentts", "qwentts_06b", "qwentts_fast",
    # Sopro V2 Turbo has no model-native preset voice; both vocoder engines
    # require a reference wav and belong on the cloning board only.
    "sopro", "sopro_streaming",
    # No model-native preset voice: the no-reference run clones the house Chris
    # reference, so these are reference-only (cloning board). Their fallback was
    # jo.wav until 2026-07-29 -- see docs/known-issues.md.
    "f5tts", "indextts", "step_editx",
    # Same defect, caught in the follow-up pass: these stored the fallback as a bare
    # stem ("jo") rather than "jo.wav", so the first sweep's grep missed them.
    "neutts_air", "neutts_nano", "luxtts",
}

# Held OUT of the manifest entirely (BOTH lenses) — known-bad output that would
# pollute the live Elo. Add a slug here to hold it; remove once it's fixed/re-OK'd.
#   (empty) — cosyvoice un-held 2026-06-16: long-form output is unstable, but Tym
#   OK'd adding it (cloning lens only — it stays in NO_PRESET_VOICE) so public votes
#   reflect its real, sometimes-bad quality.
HOLD_FROM_POOL = set()

# Mirrors publish.py SPEED_ONLY: models with a speed row but no place in the vote
# pool because their audio duplicates another tracked model (kokoro_mlx == kokoro).
SPEED_ONLY = {
    "kokoro_mlx",
    # qwentts: no publishable clip set -- the stock-runtime decode runs away on the
    # longer prompts, so only the pre-fix jo clone exists (wrong voice). Held out of
    # the vote pool; qwentts_fast is the same checkpoint and stays votable.
    # Mirrors publish.py -- keep in sync. See docs/known-issues.md.
    "qwentts",
    # breeze_tts2: clones a consent-clean reference, not the house one, so a cloning
    # pairing would ask voters to match it against a voice it never targeted.
    # Mirrors publish.py -- keep in sync. See docs/known-issues.md.
    "breeze_tts2",
    # qwentts_06b (0.6B Base): same runaway, plus truncated audio on the runs that
    # did finish. Never benched into a canonical. Mirrors publish.py.
    "qwentts_06b",
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
    # HOLD_FROM_POOL (known-bad, not votable yet) and SPEED_ONLY (audio duplicates
    # another tracked model) are both dropped in every mode.
    drop = (NO_PRESET_VOICE if mode == "default" else set()) | HOLD_FROM_POOL | SPEED_ONLY

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


def build_manifest(gh_root, base_url, prompts: dict, model_meta: dict | None = None) -> dict:
    """model_meta: {slug: {"name": display, "url": checkpoint_url}} — embedded so
    the Space can reveal model names+links AFTER a vote without importing report.py.
    Only slugs that actually appear in the clips are emitted."""
    modes = {}
    slugs = set()
    for mode in ("default", "cloning"):
        clips, ref = scan_dirs(gh_root, mode, base_url)
        modes[mode] = {"reference_url": ref, "clips": clips}
        slugs.update(c["model"] for c in clips)
    models = {s: (model_meta or {}).get(s, {"name": s, "url": None}) for s in sorted(slugs)}
    return {"base_url": base_url, "prompts": prompts, "modes": modes, "models": models}


def main():
    ap = argparse.ArgumentParser(description="Build arena/clips_manifest.json from _gh-pages.")
    ap.add_argument("--gh-root", default=GH)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "clips_manifest.json"))
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from scoring.prompts import PROMPT_BY_ID  # {str(pid): (lang, text)} — public
    from report import MODEL_DISPLAY_NAMES, MODEL_URL  # presentation registry — public
    prompts = {pid: [lang, text] for pid, (lang, text) in PROMPT_BY_ID.items()}
    meta = {s: {"name": MODEL_DISPLAY_NAMES.get(s, s), "url": MODEL_URL.get(s)}
            for s in set(MODEL_DISPLAY_NAMES) | set(MODEL_URL)}

    manifest = build_manifest(args.gh_root, _base_url(), prompts, meta)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    n = {m: len(b["clips"]) for m, b in manifest["modes"].items()}
    print(f"wrote {args.out}  clips={n}  base={manifest['base_url']}")


if __name__ == "__main__":
    main()
