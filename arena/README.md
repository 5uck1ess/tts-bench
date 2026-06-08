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

Local dev runs with all of these empty (Turnstile disabled, sqlite at `arena/arena.db`).

## Regenerate the clip pool
After publishing new bench clips to gh-pages:
```
venvs/arena/Scripts/python.exe -m arena.build_manifest
git add arena/clips_manifest.json && git commit -m "chore(arena): refresh clip manifest"
```

## Deploy
Push the `arena/` tree to the HF Space git remote (Docker SDK). Set the secrets
above. The Space builds the Dockerfile and serves on port 7860.
