---
name: add-tts-model
description: End-to-end workflow for adding a new TTS model to the tts-bench harness and getting it live on the leaderboard and voting arena. Use this whenever someone proposes, links, or asks about adding a TTS model to the bench — including bare model links ("is this for us?", "thoughts on this?", a HuggingFace/GitHub URL with no other text), "should we bench X", "add X to the board", "can we add this system", or a request to publish/republish bench results. Also use it when triaging a model you might SKIP, since the scope decision and its written record are part of this workflow. The order of the gates matters more than any individual gate — most failures here are sequencing mistakes, not knowledge gaps.
---

# Adding a TTS model to tts-bench

The bench is a public leaderboard. A half-added model doesn't error — it silently
renders as a raw slug with empty cells, or publishes speed numbers that aren't
comparable to the rows around it. Nothing in the pipeline will tell you. That's
why this is a sequence, not a checklist you can do in any order.

Every failure recorded so far was an **ordering** mistake by someone who already
knew the individual facts. Read the whole flow before starting.

## Phase 0 — Scope decision (do this before installing anything)

Check the four inclusion bars in `docs/considered.md` and the `bench-scope-decisions`
memory. A model is in scope if it: installs cross-platform in a self-contained venv;
has a clean single-process Python API (text in, wav out); has a license permitting
redistribution of benchmark outputs; and brings value the board doesn't already have.
CPU support is preferred, not required.

**When a model's card leads with "we win N of M columns", check whether those columns
are independent metrics.** NISQA emits MOS + noisiness + coloration + discontinuity
together — counting them as four wins inflates the case fourfold. Gepard 1.0 was
skipped on exactly this, plus a SIM score outside the field.

**A skip is a deliverable.** Write it into `docs/considered.md` with the reasoning and
a concrete revisit condition. Half this file's value is stopping the same model from
being re-triaged in six months.

**Watch for the model behind the model.** A link is often an optimization, a fine-tune,
or a wrapper — the base checkpoint may be the better row, or the pairing of both may be
better than either. When two variants share weights and differ only in engine, benching
both makes the speed delta attributable, which a single row can't do.

## Phase 1 — Install

One venv per model family at `venvs/<name>`, shared by variants. On Windows, install
cu128 torch **last** — telemetry and lightning packages silently downgrade it. Pin
`transformers` exactly when the checkpoint ships `trust_remote_code` modeling code;
a free resolve picks a major version that loads the weights and then breaks generation
far from the error site.

Add the stanza to **both** `install.ps1` and `install.sh` as you go, not afterward. The
Linux script differs: PyPI torch is already CUDA-enabled there, and `triton` is a real
wheel (Windows needs `triton-windows` for anything using `torch.compile`).

## Phase 2 — Smoke with the REAL canonical prompts

Before writing the runner, probe the model directly and answer these empirically rather
than from the card:

- What is `codec_sample_rate` / the true output SR?
- Does the no-reference path produce a genuine preset voice, or does it fall back to
  cloning the bundled reference? This decides `NO_PRESET_VOICE` vs `_PRESET_AND_CLONE`
  and therefore which leaderboard lens the model appears in.
- Does cloning work, and does it need a transcript? (convention: sibling `.txt`)
- Does the long prompt complete?

**Use `bench.py`'s actual `PROMPTS`, especially prompt 3.** Short test text has hidden a
hard failure that would otherwise have shipped. Prompt 3 is where token caps, duration
estimators, and chunking break.

**The generation-cap trap.** Many models default to a `max_new_tokens` that looks
generous and isn't. Compute the ceiling in seconds — `cap ÷ frames_per_second` — and
compare it against measured prompt-3 frames. Audio8's compiled path defaulted to 400
frames at ~21.5 fps (18.6 s) while cloned prompt 3 measured 390. Ten frames of margin,
and with static shapes the cap was baked into the compiled graph. Raise the cap; the
loop still exits at EOS, so the cost is memory, not speed.

## Phase 3 — Runner

Copy the shape of an existing runner (`runners/inflect_runner.py` for a simple one,
`runners/qwentts_fast_runner.py` for an engine-variant pair). Contract:

- args `--text --out --device --reference --variant --runs --language --stdin`
- one JSON line per run on stdout: `{"ok", "run_index", "ttfa_ms", "gen_s", "audio_s", **_meminfo.sample(device)}`
- `{"ready": true}` first in `--stdin` mode
- failures emit `{"ok": false, "error": ...}` and exit cleanly — never crash the cell
- non-streaming models report `ttfa_ms == gen_s`; say so in the docstring so the number
  isn't read as a streaming latency
- pin seeds and any voice selector so reruns are reproducible

Document *why* in the module docstring — the gotchas you hit are the most valuable
thing in the file, and the next person is usually you.

## Phase 4 — Register everywhere (the silent-failure zone)

`report.py` has a registry-drift guard that fails at import, which covers most of this.
It does not cover the README.

1. `harness.py` `MODELS` — `(name, venv, runner, multilingual, devices, variant, can_clone)`.
   `multilingual=True` makes the French prompt run; only claim devices you have actually run.
2. **All nine registries in `report.py`**: `MODEL_DISPLAY_NAMES`, `MODEL_SIZE`, `MODEL_URL`,
   `MODEL_KIND`, `MODEL_RELEASE`, `MODEL_SR`, `MODEL_EXPRESSIVE`, `MODEL_LICENSE`, `MODEL_LANGS`.
   `MODEL_CROSSLINGUAL` is curated — add only on positive evidence of cloning in one
   language and speaking another.
3. `publish.py` — `NO_PRESET_VOICE` (cloning-only, no real default voice) or
   `_PRESET_AND_CLONE` (has both; fills both lenses). Keep in sync with
   `arena/build_manifest.py`, which mirrors these sets.
4. **`README.md` — there are TWO counts**, and the tests check both: the
   `## Models tracked (N)` heading and the `**N of the M tracked models can clone**`
   sentence. Add the table row to the right section, alphabetically by display name.

**Params come from the model card or from computing over the safetensors — never
from a sibling model.** A NeuTTS variant was mislabeled for exactly that reason.

## Phase 5 — Tests

```
python -m pytest scoring/tests/
```

Run this *before* benching, not before publishing. It catches README count drift and
registry gaps in under a second. `scyllasband` shipped live with zero registries because
this was skipped.

## Phase 6 — Bench at CANONICAL parameters

Read `results/<rig>-default/meta.json` and match its run parameters. Getting this wrong
produces rows that look fine and aren't comparable to their neighbours:

```
python bench.py --models <new> --runs 3
python bench.py --models <new> --runs 3 --reference reference/chris_hemsworth_15s.wav
```

The canonical dirs use **3 runs across cpu + cuda**, not 1 run on cuda. Three runs is
also what produces the warm column — and warm-vs-cold is how you detect state bugs in
engines that reuse KV caches or CUDA graphs between calls. If run 2 or 3 degrades, the
runner is holding state it shouldn't.

The per-prompt cell timeout is 600 s and covers load + all runs of **one** prompt. A slow
CPU path or a cold `torch.compile` can sit inside it; if a first cell times out on a
compiled model, warm the inductor cache with a throwaway single-prompt run and retry
before treating it as a real failure.

## Phase 7 — Health check, then ears

`scoring/health.py`'s `HealthScorer().detail(wav)` flags clipping, silence and gaps. It
catches hard defects only — clean health is necessary, not sufficient.

Then get a human verdict. Stage the diagnostic clips (prompt 2 and the long prompt 3,
default and cloning) somewhere playable and ask. For engine-variant pairs the question
is sharper than "is this good": it's **"does the fast row sound the same as the base
row?"** — if compilation changed the numerics audibly, the speed win is costing quality
and the row is misleading. Build the comparison so that question is answerable side by side.

Do not publish before this gate.

## Phase 8 — Publish

```
git -C _gh-pages reset --hard origin/gh-pages     # FIRST. Always.
python merge.py --into results/<rig>-default  --from results/<scratch> --models <new>
python merge.py --into results/<rig>-cloning  --from results/<scratch> --models <new>
python publish.py results/<rig>-default
python publish.py results/<rig>-cloning
```

**`publish.py`'s `ensure_worktree()` does not pull.** Another rig publishes to the same
branch. Without the reset you build on stale state and either get rejected (fine) or
force-push over another machine's rows (not fine).

**If the push is rejected, do not rebase and never force.** gh-pages content is fully
generated: `reset --hard origin/gh-pages`, then re-run `publish.py` for each canonical
dir. `build_top_level()` rebuilds the landing pages from *every* canonical dir present
in the worktree, so this correctly merges the other rig's data with yours. Verify by
grepping both your new model **and** the other rig's rows out of the rebuilt files
before pushing.

Two mechanical notes: the shell's working directory can drift into `_gh-pages/` after a
`cd`, so run `publish.py` from an explicit repo root and use `git -C <abs path>` for
worktree commands. And `results/` is gitignored — the canonical dirs live only on the
rig's disk and in the gh-pages branch, so never delete them casually.

## Phase 9 — Verify live, not just pushed

Pushed ≠ served. Check the Pages build reports `built` on your commit
(`gh api repos/<owner>/<repo>/pages/builds`), then fetch the live pages **with a
cache-buster** — a plain fetch will hand you a stale CDN copy and make a correct deploy
look broken. Confirm the display name renders (a raw slug means a missing registry) and
that a published wav returns 200.

## Phase 10 — Arena

`arena/build_manifest.py` globs the gh-pages canonicals and drops
`NO_PRESET_VOICE | HOLD_FROM_POOL | SPEED_ONLY`, so a correctly-registered model is
picked up automatically — but the Space bakes the manifest in at startup and must be
redeployed:

```
venvs/arena/Scripts/python.exe -m arena.build_manifest
python -m pytest arena/tests/      # use the arena venv; system python lacks fastapi
```

Then upload via the Python API, **not** `hf upload`:

```python
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="arena/clips_manifest.json",
                    path_in_repo="clips_manifest.json",
                    repo_id="<owner>/<space>", repo_type="space",
                    commit_message="arena: ...")
```

`hf upload` calls `/api/repos/create` (exist_ok) before uploading, and **creating a Docker
Space now requires HF PRO** — so the CLI dies with `402 Payment Required` even when the
Space already exists and is RUNNING. It reads like a billing wall; it isn't. `upload_file`
never touches that endpoint. (Hit on 2026-08-08 publishing vaniq.) Run it from any venv
that has `huggingface_hub` — the arena venv does not.

`git push` to a HuggingFace remote will hang on a credential dialog in a non-interactive
shell; both the `hf` CLI and the API use the cached token instead.

**Diff the manifest against the deployed one before uploading, and back up the live
copy first.** The manifest reflects every bench change since the last deploy, not just
yours — expect models to leave the pool as well as join. Confirm each delta is intended;
a model leaving the default lens usually means it correctly landed in `NO_PRESET_VOICE`.
Also diff the whole tree: if the repo's `app.py`/`config.py` have drifted ahead of the
deployed Space, uploading everything can deploy unrelated changes. Prefer a
manifest-only upload after confirming the schema is unchanged.

## Phase 11 — Record it

- `docs/known-issues.md` — the traps, with the measurements that prove them
- memory: scope decision, shipped state, new SHAs, model count
- `homebase/coordination/tasks.md` — a task for the rigs that still lack the model.
  Objective scores (UTMOS / WER / **SIM**) can only be produced on Linux; SIM's fairseq
  stack does not run on Windows. If a human liked the cloning by ear, the number they
  actually want is gated on that pass.

Write predictions into the task where you have them (expected RTFx on other hardware,
and why). A prediction that can be checked later is worth more than a summary.
