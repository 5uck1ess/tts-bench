# Bench architecture

How the bench is structured, what each piece does, and how to add a new model.

---

## What the bench measures

Five prompts, English + one French, mixing conversational and technical content:

1. `"Open the browser and read my email."` — short / conversational
2. `"I'll start a new git branch, push the changes, and open a pull request when the tests pass."` — medium / dev-flavored
3. ~30-word Parakeet-TDT paragraph with embedded numbers — long / technical
4. Shell command read aloud — punctuation / symbol density stress
5. `"Bonjour, je m'appelle Cicero..."` — French / multilingual (skipped on EN-only models)

Per `(model, device, prompt)` cell:
- One subprocess loads the model once
- Generates the prompt **N times** (default 3 = 1 cold + 2 warm)
- Run 1 = cold (no warm cache, no JIT prime)
- Runs 2..N = warm (model resident, codecs primed)

Both numbers matter for different reasons. Cold matters for "user opens the app and says something". Warm matters for "user is in a conversation, every turn after the first".

---

## Layers

Four layers, all stdlib in the orchestrator:

| Layer | What |
|---|---|
| `harness.py` | Shared model registry + subprocess plumbing. Defines `MODELS`, builds the list of runnable `(model, device)` cells for the current machine, owns the JSON-line protocol. Imported by everything else. |
| `bench.py` | Formal benchmark. Loops prompt × cell × runs, writes `results.csv`, prints per-prompt summary. |
| `compare.py` | One-shot A/B listening tool. Takes one piece of text, runs it through every installed model on every available device, dumps a wav per cell, plays them out loud as they finish, prints a comparison table. |
| `speak.py` | Interactive REPL. Keeps a single runner subprocess alive across turns so warm-run latency is measurable. Uses `winsound` / `afplay` / `aplay` for playback. |
| `runners/*.py` | One runner per model. Loads the model in its own venv, generates audio, writes WAV, emits one JSON line per run to stdout. Also supports `--stdin` mode for `speak.py`. |

Runners communicate via JSON lines so each model can live in its own conflicting dependency tree. The orchestrator never imports any TTS library directly.

---

## Three tools, three jobs

- **`bench.py`** — numbers. 5 prompts × every model × cold + warm. CSV output. Reach for this when you want hard data.
- **`compare.py`** — ears. One phrase → every model → audio out loud, side by side. Reach for this when you want to *hear* which model sounds best on your line.
- **`speak.py`** — feel. REPL that holds one model in memory. Type a prompt, hear it, type the next. Reach for this when you want to feel the warm-run latency of one model interactively.

---

## Adding a model

1. Make a venv under `venvs/<name>/` (add to `install.ps1` and `install.sh`).
2. Write `runners/<name>_runner.py` matching the existing runner protocol:
   - Single-shot mode: `--text TEXT --out PATH --runs N` → one JSON line per run on stdout: `{"ok": true, "run_index": N, "ttfa_ms": ..., "gen_s": ..., "audio_s": ..., "peak_mem_mb": ..., "peak_vram_mb": ..., "naq": ..., "naq_harm": ..., "naq_buzz": ...}`
   - REPL mode: `--stdin` → on startup emit `{"ready": true}` after model load, then per stdin line: `{"text": "...", "out": "path.wav"}` → respond with one JSON line per generation
   - Use the shared helpers: `import _meminfo` and `import _naq` near the top of the runner; call `_meminfo.reset_peak(args.device)` before generation and spread `**_meminfo.sample(args.device)` + `**_naq.score(out_path)` into the success JSON
3. Add a row to `MODELS` in `bench.py` and `speak.py`.
4. Test: `python bench.py --models <name> --prompts 1 --runs 1`

The two existing runners (`pocket_runner.py`, `neutts_runner.py`) are ~150 lines each — short enough to copy and adapt.

---

## Repo layout

```
tts-bench/
├── harness.py            # shared model registry + subprocess plumbing (imported by all 3 tools)
├── bench.py              # formal benchmark — CSV + per-prompt summary + auto-generates report.html
├── compare.py            # one-shot A/B listening tool — every model × every device, plays out loud
├── speak.py              # interactive REPL — feel warm-run latency for one model
├── report.py             # build HTML report (table + inline audio players, light/dark, sort, filter) from a results/ dir
├── publish.py            # ship a chosen run to the gh-pages branch for GitHub Pages hosting (managed via a git worktree at _gh-pages/)
├── docs/                 # documentation pages (you are here)
├── install.ps1           # Windows installer
├── install.sh            # macOS / Linux installer
├── runners/
│   ├── _meminfo.py       # shared CPU/CUDA memory sampling
│   ├── _naq.py           # shared NAQ scoring (see docs/naq.md)
│   └── <model>_runner.py # one per model
├── reference/            # voice cloning reference audio (.wav + .txt pairs)
├── venvs/                # one isolated venv per model (gitignored)
└── results/              # bench output WAVs + CSV (gitignored)
```
