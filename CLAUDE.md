# tts-bench: agent rules

The code, `README.md` and `docs/` are the source of truth. For adding, re-benching or publishing a model,
follow `.claude/skills/add-tts-model/SKILL.md`. Its phases are ordered, and the order is the point.

## Rules

- **Record bench findings in `docs/known-issues.md` and scope calls in `docs/considered.md`.** Never open GitHub issues for them. GitHub issues are for reports from other people, and two homes drift apart.
- **The NAQ recipe never enters this repo.** That covers features, training, the vote pipeline and the private scoring pass, in code, docs, PRs and the skill. It lives in the gitignored `naq_lab/`. The per-clip scores in `scoring/naq_scores.csv` and the short description in `docs/methodology.md` are public.
- **Before `git worktree remove`, unlink every junction or symlink in the worktree and check that none is left** (Windows: `[System.IO.Directory]::Delete(path, $false)` for directory links, `Remove-Item` for file links, then `Get-ChildItem -Recurse -Force -Attributes ReparsePoint` must print nothing; Linux: `rm <link>`, then `find . -type l`). Run the remove only if the unlink succeeded; never chain it after `;`. git follows a junction and deletes the real `venvs/` or `results/` behind it. This has happened twice.
- **Don't prune `venvs/` or model weights to save disk.** Rows get re-benched. A venv that is gone can be rebuilt with `install.ps1 <slugs>` / `install.sh <slugs>`.
- **A worktree has no `venvs/`, `models/`, `results/` or reference wavs** (all gitignored). Junction the first three and copy the reference wavs into `reference/`: default-lens runs of reference-only models fall back to `reference/chris_hemsworth_15s.wav` there. For scoring, also link `scoring/thirdparty` and the WavLM checkpoint, or every SIM cell comes back blank (the error only goes to stderr).
- **Run tests from inside the tree you changed.** From a worktree, running the main checkout's tests imports the main checkout's code.
- **The Linux rig has no audio output.** Do every by-ear check on Windows. When the call belongs to the owner, send the wavs to them directly, with a line on what each variant is, rather than just naming repo paths.
- **Fix review findings that a real vendor or model response can trigger, plus real regressions, in one round.** If the next review is NEEDS_HUMAN with only crafted-input findings, say so in one line and recommend merging; don't open another fix round. Adapters are bench code, and the bar is that they decode real responses correctly.
- **Ask the owner any remaining gate questions together, in one message.** Once you've agreed a fix is needed, make it without waiting for a second go-ahead.
- **When a session queues a candidate model, tell the owner directly.** A backlog line in `docs/tasks.md` alone has gone unseen for two months.
- **Never ask for an API key in chat or in a `!` command**, because both land in the transcript. Hosted keys come from `scripts/hosted/keys.py`.
- **Never quote arena vote totals from the docs.** They are snapshots; the board counts per lens, so pull `python -m arena.export_votes --clean --mode cloning` (or `--mode default`) with `--out` outside the repo: it prints the count, and the CSV carries voter tokens. It needs Turso creds in `arena/.env`, or it reads the local SQLite.
- **A Codex run that needs a model venv can't test on Windows through a junctioned venv.** Run the venv tests yourself before accepting a "tests pass" report.
