---
title: TTS Voting Arena
emoji: 🎤
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# TTS Voting Arena

Public blind 2AFC voting for the [tts-bench](https://github.com/5uck1ess/tts-bench)
leaderboard. Two lenses (Default voice quality, Cloning fidelity); audio is served
from gh-pages; votes persist to Turso; a live human-preference Elo is shown.

Hosted APIs have their own lenses and Elo. Separately, `ARENA_MIX_RATE` (default 0.15) of
Default-voice rounds pit a hosted API against a local model; those votes are logged as mode
`mixed-default` and never move either Elo board.

## What it is / isn't
- Collects human preference at scale (research ground truth). PUBLIC.
- Contains **no NAQ code** — pairing/Elo only. NAQ stays private R&D.

## Environment (HF Space secrets)
| Var | Purpose | Required |
|---|---|---|
| `TURSO_URL` | libSQL database URL (`libsql://…`) | prod |
| `TURSO_TOKEN` | Turso auth token | prod |
| `TURNSTILE_SECRET` | Cloudflare Turnstile secret | prod (empty disables) |
| `TURNSTILE_SITEKEY` | Turnstile sitekey (rendered in page) | prod |
| `HMAC_SECRET` | pair-nonce + ip-hash signing secret | prod |
| `ADMIN_TOKEN` | header token for `/admin/flag_token` | prod |
| `ARENA_LANGS` | votable prompt langs (default `en`; `all` for every) | no |
| `ARENA_MIX_RATE` | share of Default rounds that are hosted vs local (default `0.15`; `0` turns it off) | no |

Local dev runs with all of these empty (Turnstile disabled, sqlite at `arena/arena.db`).

## Anti-abuse posture
Every vote is logged; only votes passing all gates (Turnstile, both-clips-played,
dwell ≥ 1.5 s, valid pair-nonce, token not flagged, rate limits) feed the live Elo.
Pollution is reversible — flag a token via `POST /admin/flag_token` and the board
re-derives without it. The persistent anonymous `voter` token is the primary
Sybil control; the per-IP cap (`ratelimit.IP_MAX_TOKENS_PER_HOUR`, default 8) is a
best-effort secondary signal — IP is spoofable and shared behind NAT/CGNAT, so the
cap is intentionally generous to avoid dropping honest shared-IP voters. Tune it in
`arena/ratelimit.py` if a deployment's audience is unusually concentrated.

## Regenerate the clip pool
After publishing new bench clips to gh-pages:
```
venvs/arena/Scripts/python.exe -m arena.build_manifest
git add arena/clips_manifest.json && git commit -m "chore(arena): refresh clip manifest"
```

## Deploy
The Space files live at the Space repo root, and a GitHub push does not deploy them.
Upload with the Python API, using a **write** token:
`HfApi().upload_file(..., repo_type="space")` for the manifest alone, or
`upload_folder` on a staged copy of the tree (`git archive HEAD:arena`, which leaves out the
gitignored `.env`, `arena.db` and `backups/`).
- Don't use `hf upload`. It calls the repo-create endpoint first, which has returned 402 for
  Docker Spaces on a free account even when the Space already exists.
- Don't `git push` to the Space remote. It hangs on a credential prompt in a non-interactive
  shell.

Set the secrets above. The Space builds the Dockerfile and serves on port 7860, and it reads
the manifest only at startup.

Before uploading a manifest:
- Check that every new clip URL already returns 200 from gh-pages.
- Diff it against the live deployed manifest, not just the repo copy.

To debug the running Space, read its logs from `GET https://huggingface.co/api/spaces/<id>/logs/run`
(server-sent events, with a bearer token).

If the live board shows 0 votes, check whether Turso's free tier suspended the idle database
(seen 2026-07-05). It came back on its own and no data was lost.
