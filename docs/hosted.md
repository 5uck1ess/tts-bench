# Hosted (paid API) models — design

Status: DRAFT 2026-10-28. Origin: [issue #14](https://github.com/5uck1ess/tts-bench/issues/14) (Giggy asked
whether hosted-only models can enter the gallery and arena).

## Rule: separate everything

Hosted models never touch the local board. Separate clip tree, results CSV, speed table, Scores table,
page tab and arena pool with its own Elo. No hosted row is ranked against a local row anywhere, and no
merged "overall" ranking exists. The local board stays a measure of *models on known hardware*; the
hosted board is a measure of *products as a buyer gets them*.

## Roster (v1)

**Locked 2026-10-29 (latest + best per vendor, 24 models):**

1. ElevenLabs Eleven v4
2. ElevenLabs Eleven v4 Turbo
3. HeyGen Voice
4. Qwen-Audio-3.1-TTS-Plus
5. Cartesia Sonic 3.6
6. Gemini 3.8 Flash TTS
7. Inworld Realtime TTS-2 (Tym: released 2026-10-29; confirm the API model id at adapter time)
8. Speechify SIMBA 3.5 (replaces 3.2: newer, adds fr/es)
9. Fish Audio Drama 3
10. Mistral Voxtral 3 TTS
11. xAI TTS
12. Rime Coda (replaces Arcana, no longer on Rime's pricing page; 8 langs, $50/1M)
13. AssemblyAI TTS-1
14. Soniox TTS v2
15. Giggy TTS
16. StepFun stepaudio-3-tts (replaces StepAudio 2.5; $36/1M nominal)
17. Smallest.ai Lightning v3.1 Pro
18. Deepgram Flux TTS (~$45/1M third-party, English only)
19. Microsoft MAI-Voice-2.1 ($22/1M, 23 langs)
20. MiniMax speech-2.8-hd ($100/1M) — **via OpenRouter** (`minimax/speech-2.8-hd`) to avoid MiniMax's $25
    minimum top-up. Default voice only (OpenRouter's speech endpoint has no cloning); the row and its speed
    numbers are labelled "via OpenRouter" (extra relay hop)
21. VUI Labs Luna ($15/1M; AA #7 on 2026-09-25)
22. OpenAI gpt-4o-mini-tts (retires 2027-01-06; check for a successor at adapter time)
23. Murf Falcon 2 ($10/1M)

Dropped: Gemini Flash-Lite (cheap tier of the same vendor); Hume Octave 2 (Hume is shutting down its
TTS voices, per Tym 2026-10-29).

Account funding (Tym, 2026-10-29): HeyGen $5 minimum, Mistral $10 minimum, StepFun $10 (loaded),
xAI $20 left, ElevenLabs on existing credits; the rest have free credit. Sources: homebase
`wiki/research/deepfake/voice/tts-refresh-sep2026.md`, `wiki/research/pipecat-service-catalog-oct2026.md`.
The table below predates the lock and still has older per-row notes.

Prices are list $/1M chars, taken from homebase research (each row there was read off the vendor's
pricing page; read dates 2026-09-25 → 2026-10-28): `wiki/research/deepfake/voice/tts-refresh-sep2026.md`
(TR), `wiki/research/pipecat-service-catalog-oct2026.md` (PC), `wiki/research/phone-voice-agent-cost-sep2026.md`
(PV), `wiki/setup/artifact-refresh.md` (AR). Per-minute/token prices are converted at ~900 chars per
minute of speech and marked ≈. Re-check each against the vendor page on publish day.

| Model | $/1M chars | As published | Source | Notes |
|---|---|---|---|---|
| Eleven v4 | $80 | $0.08/1k | TR | launch promo ended Oct 12 |
| Eleven v4 Turbo | $40 | $0.04/1k | TR | |
| Rime Coda | $50 | $0.05/1k | TR | |
| Giggy (streaming) | $10 | $10/1M | TR | Pro pack or larger; Fast $5, Batch free |
| Cartesia Sonic 3.6 | $38–65 | credits = chars, by plan | TR | $65 Pro · $45 Startup · $38 Scale |
| Gemini 3.8 Flash TTS | ≈$15 | $0.50/1M text in + $9/1M audio out | TR | ≈$0.0135/min; doubles 2027-01-01 |
| Inworld TTS-2 | $25 | $25/1M on-demand | TR | → $5 enterprise |
| Speechify SIMBA 3.5 | $12–20 (launch $6–10) | $20/$16/$12 per 1M by plan | TR (10-30) | half-price launch, no end date; 2× SIMBA 3.2 |
| Fish Drama 3 | $10 now, $50 list | per 1M UTF-8 bytes | AR | $10 launch price ends 2026-11-12 |
| Mistral Voxtral TTS 3 | $16 | $16/1M | PC | AR: Mistral lists Voxtral TTS 3 without a price |
| xAI TTS | $15 | $15/1M chars | TR (10-30) | |
| AssemblyAI TTS-1 | $15 | $15/1M | TR | flat, all voices |
| Soniox tts-rt-v2 | ≈$14.1 | ≈$0.70/hr of speech | PC | |
| StepFun stepaudio-3-tts | $36 (≈$18 EN) | $0.36/10k chars, 2 EN letters = 1 char | TR | |
| Smallest Lightning v3.1 Pro | $19.5 | $0.195/10k | TR | |
| Deepgram Flux TTS | $45 | $0.045/1k | TR | |
| MAI-Voice-2.1 | $22 | $22/1M on OpenRouter; ≈$1.19/generated hr on Azure | TR (10-30), PV | |
| VUI Luna | $15 | $15/1M chars | vuilabs.ai/pricing (10-30) | cloning $3/clone; Luna-TTS Character same price |
| OpenAI gpt-4o-mini-tts | ≈$17 | $0.60/1M text in + $12/1M audio out | TR | ≈$0.015/min; retires 2027-01-06 |
| OpenAI gpt-realtime-2.1 | token-priced | $4/1M text in, $64/1M audio out | PV | |
| Murf Falcon 2.1 | $10 | $0.01/1k | TR | /api page says "1¢/min", a mismatch |
| HeyGen Voice 1 | $30 | $30/1M direct | TR | $15/1M via OpenRouter |
| MiniMax speech-2.8-hd | $100 | $100/1M on OpenRouter (same as direct) | TR (10-30) | |
| Bland Speech v3 | $40 | $40/1M | TR | |
| Gradium | $35.9–57.8 | by plan | PC | |
| Async Flash v1.5 | ≈$9.3 | $0.50 per hour of generated audio | docs.async.com Billing & Pricing (10-30) | prepaid wallet; Pro v1.0 $1.00/hr |

Already collected: Fish Drama 3 web-app clips in `hosted/drama-3-webapp/` (fidelity check for the `drama3`
port). These are web-app clips, not API clips, and stay out of the hosted board.

Full roster sweep ≈ $1.25. Speed sampling (below) multiplies that ~5×; still under $10 total.

## How we test (why this differs from the big boards)

The big arenas let each vendor pick their best voice, use short, friendly prompts and publish no audio
or request bodies. Here:

1. **Same prompts as the local board**: the 6 canonical prompts, including p3 (long, numbers-heavy) and
   p4 (code read aloud), plus fr and es. No vendor-specific text normalization. A model that reads
   "pytest" badly gets scored on it.
2. **Default lens = the vendor's documented default voice**, pinned by voice id, default settings. The
   chosen voice id and every non-default parameter are written in the row. No hunting for each vendor's
   best voice.
3. **Cloning lens = one shared reference** for every provider (see open question 1), sent through each
   API's own instant-clone path. The clone id is recorded.
4. **We call the APIs ourselves.** No vendor-submitted samples, Giggy included.
5. **3 takes per cell, all published.** No cherry-picking; failures (truncation, wrong language,
   silence) ship as defects, the same as the local board.
6. **Full provenance per clip**: the JSON sidecar `collect_hosted.py` already writes (endpoint,
   model id sent, request body, response headers, UTC time, key fingerprint, sha256, SR).
7. **Same scorers**: UTMOS / WER / SIM / Health on the Linux box, the same code as the local Scores lens.
   Clips are scored at their native SR, which goes in a column.

## Speed (its own table)

All timings are **measured from Eastern US** (Tym's home connection). Every hosted speed row and the
table header say so; no other location claims are made.

### Captured per call (in the clip's JSON sidecar)

Network phases, timed by the client so network can be separated from generation:

- `dns_ms`, `tcp_connect_ms`, `tls_ms`: connection setup. Measured on a **fresh connection every
  call** (no keep-alive reuse), so each call pays the same setup and the phases are comparable.
- `resolved_ip`, `http_version`, and any region/POP header the vendor returns (e.g. `cf-ray`,
  `x-amz-cf-pop`, `server`), so the endpoint we actually hit is on record.
- `ttfb_ms`: request fully sent → first response byte (headers).
- `ttfa_ms`: request sent → **first decoded audio sample** (not the first byte: a WAV header or JSON
  envelope isn't audio). Only on streaming endpoints; non-streaming APIs record `null`, never their total.
- `total_ms`: request sent → last byte.
- `audio_s` (decoded duration) and the implied `rtf = audio_s / (total_ms/1000)`.
- **Vendor-reported server timing** when the API exposes it (`server-timing`, `x-processing-ms`,
  `request-id`, usage/char counts in the body). Stored raw, never mixed with our numbers.
- `rtt_ms`: a TCP-connect probe to the same host just before each call (median of 3), as a
  per-call network baseline. `ttfa_ms - rtt_ms` approximates the vendor-side time to first audio.
- `utc`, `endpoint`, `streaming` (bool), request body, key fingerprint (already in the sidecar).

### Published (hosted speed table)

- **Time to first speech** (`ttfa_speech_ms`: first sample above −40 dBFS, so leading silence
  doesn't count) leads the table and is the speed ranking.
- **Total** and RTFx are shown as "delivery time, includes vendor pacing": some vendors (Deepgram,
  Soniox) send audio at roughly playback speed on purpose, so a 6 s clip takes 4–5 s to arrive
  even though first audio is ~200 ms. Never rank by Total/RTFx.
- **Transport**: each vendor runs on whichever of its paths gave the faster warm first speech in an
  A/B on 2026-10-30 (Gemini `streamGenerateContent`; Cartesia and AssemblyAI WebSocket; the rest
  HTTP). The transport is a column.
- Each is the **median over ≥10 calls spread across ≥3 days and different times of day**, with p90
  and the call count shown. Failed or errored calls are counted and shown (error rate), not dropped.
- Per-prompt for p1 (short) and p3 (long), since TTFA vs length is the interesting curve.
- Labelled with the date range and endpoint. The header says plainly that these are not comparable
  to the local speed tables.

## Cost column

List $/1M chars and the measured cost of one sweep, with the date the price was read. Prices change
(Gemini doubles in January), so the column carries its read date.

## Arena

Two new pools, `hosted-default` and `hosted-cloning`, each with its own Elo and vote counter. Same
gates, nonce and Turnstile as the local pools. Pairs are drawn only within a pool. Clips go through
`build_manifest.py` from `hosted/`, with `HOLD_CLIPS` available as on the local pools.

## Code

- `scripts/collect_hosted.py` → a provider-adapter layout: one small `synth()` per provider returning
  `(audio, request_logged, headers, ttfa, total)`, sharing `save()` / the sidecar / the CLI. Add `--stream`
  for TTFA and `--runs`.
- `scripts/run_hosted.ps1` → decrypts only the key for the provider being run, from
  `$HOME\.<provider>_api_key.dpapi` (masked entry by Tym, never in chat). Nothing is stored except an
  8-char fingerprint.
- `hosted/<model>/<lens>/p<N>_t<take>.wav` (+ `.json`). Results go to a separate `results_hosted/` CSV
  that `merge.py` never reads.
- `report.py` → a "Hosted" tab rendered from `results_hosted/` only.
- Every adapter gets a `--probe` like Fish's: confirm the model id we send is the model we get (Fish
  silently falls back to its default on an unknown id), and that the response is the expected SR/format.

## Before anything is published

- **ToS per provider.** Check that publishing generated clips (and benchmarking) is allowed. Recorded
  per provider in this doc before its clips go live.
- **Cloning consent.** See open question 1; several vendors require you to attest you have rights to
  the cloned voice.

## Open questions (Tym)

1. **Cloning reference — DECIDED 2026-10-28: not the house reference** (Chris Hemsworth; vendors require
   a rights/consent attestation we can't give for a celebrity). A new consented sample is needed.
   `reference/tym.wav` is only 13.6 s, under Gemini's 30 s minimum and ElevenLabs' recommended 1–2 min.
   Proposed: record ~90 s of Tym reading neutral prose (not the bench prompts), 48 kHz mono, quiet room,
   no music, with a verbatim transcript. Cut at sentence boundaries into a 15 s and a 30 s version with
   matching transcripts; each provider gets the longest it accepts, recorded in the sidecar.
   Store it as `reference/hosted_clone{,_15s,_30s}.{wav,txt}`.
   **Published unnamed (2026-10-29):** the page labels it "Reference voice" only; no speaker name in the
   page, sidecars, filenames or commit messages.
2. **Grok TTS**: confirm which product/model id you mean (AA lists "SpaceXAI TTS").
3. **Eleven v4 (non-Turbo)**: your list has Turbo only. Include v4 at $80, or Turbo only?
4. **Reply on issue #14**: tell Giggy they're in, we call their API ourselves with our prompts, and point
   them to this doc once it's public?

## Order of work

1. Answer the open questions; Tym enters keys (DPAPI, own terminal).
2. Refactor the collector into adapters; `--probe` each provider; check ToS.
3. One sweep per provider (both lenses, 3 takes) → score on Linux → Hosted tab (Listen + Scores).
4. Speed sampling over ≥3 days → hosted speed table.
5. Arena pools.
