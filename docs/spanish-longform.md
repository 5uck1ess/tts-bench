# Spanish long-form (Apple M4 Max)

The bench's Spanish coverage is one short prompt: p6, about 6 s of speech. The Windows 5090 rig scores it
(`docs/known-issues.md`: default-lens median WER 0.0000, mean UTMOS 3.73). This page adds **long-form**
Spanish. Seven preset-voice models were run on four public-domain passages of 100-180 words each (about
40-75 s of speech per passage, about 3-5 minutes per model when all chunks complete; the measured totals are
in the table). That
exercises chunking, number and ordinal reading, dialogue, and older literary register, which a single
sentence cannot. All runs are on one Mac.

## Rig

| | |
|---|---|
| Machine | Apple M4 Max, 128 GB unified memory |
| OS | macOS 15.7.7 (arm64) |
| Devices | CPU for every model except `kokoro_mlx` (MLX/Metal, `mps` cell) |
| Python | 3.11.14 (supertonic, paradee, reverb, scoring), 3.12.12 (kokoro_mlx, lfm3_audio, qwentts, zonos3) |
| Install | `./install.sh <slug>` from master `f6dada2`, one slug per call |

## Passages

There are four public-domain texts, frozen verbatim in `scoring/longform_es.py` with their es.wikisource revision
ids and sha256 values. Each is 100-180 words.

| id | text | what it stresses |
|---|---|---|
| `pd_galdos` | Galdós, *Marianela* (1878), cap. I | narrative prose, a parenthetical aside |
| `pd_becquer` | Bécquer, *Los ojos verdes* (1861; text of the 1885 *Obras*) | dialogue, em-dashes, ellipses, exclamations |
| `pd_cervantes` | Cervantes, *Don Quijote* I, cap. I (ed. 1905) | early-modern vocabulary and syntax |
| `pd_cadiz` | Constitución de Cádiz (1812), arts. 1-5 | ordinals ("Artículo primero"), enumerations |

## Method

- **Chunking.** `scoring/longform_chunker.py` splits at sentence ends, then at commas, into chunks of at most
  240 characters. The model's runner reads them in `--stdin` mode, and one warm-up chunk is discarded.
- **Whisper window.** `scoring/wer.py` feeds Whisper only the first 30 s of a file. A chunk whose audio runs over
  28 s is therefore split in two and re-read, so every scored file fits the window.
- **WER.** Whisper-large-v3 (the pinned scoring venv) transcribes every chunk. Digit groups in the
  *hypothesis* are spelled out in Spanish (`spell_digits`) before `normalize_intl`, because Whisper writes "72"
  for a spoken "setenta y dos" and `wer.py` normalizes numbers for English only. The reported value is total
  edits divided by total reference words over every chunk of the four passages.
- **Health.** `scoring/health.py` runs per chunk, before any joining silence: gaps, clipping and dead audio. The
  column counts flagged chunks.
- **RTFx.** Total audio seconds divided by total generation seconds over the warm chunks.
- **Voice.** Each model's own Spanish male preset. Where a model has several, the one with the lowest WER on a
  two-passage screen was kept (supertonic `M4` of M1-M5, kokoro `em_alex` of em_alex/em_santa, Qwen3-TTS
  CustomVoice `aiden` of its five male speakers). `reverb`, `paradee` and `zonos3_mini` use their runner's
  Spanish preset (`es_male_standard_1`, `em_alex`, `Rodrigo_es_M_1`). `lfm3_audio` uses `male`.

Reproduce:

```
python scripts/spanish_longform.py --out results/es-longform
venvs/scoring/bin/python scripts/spanish_longform.py --score results/es-longform/es_longform.csv
python scripts/spanish_longform.py --render-table results/es-longform/es_longform.csv
```

## Results

Measured 2026-11-22 with `scripts/spanish_longform.py` on all seven models, then scored with `--score`.

| model | voice | chunks | failed | WER | Health-flagged chunks | audio (s) | RTFx |
|---|---|---:|---:|---:|---:|---:|---:|
| qwentts_06b_custom | aiden | 19 | 0 | 0.0275 | 0 | 262.1 | 0.50 |
| zonos3_mini | builtin | 14 | 2 | 0.0307 | 0 | 130.6 | 0.20 |
| kokoro_mlx | em_alex | 19 | 0 | 0.0344 | 0 | 191.0 | 43.51 |
| paradee | builtin | 19 | 0 | 0.0413 | 0 | 196.2 | 62.84 |
| lfm3_audio | male | 19 | 0 | 0.0448 | 0 | 191.7 | 0.06 |
| supertonic | M4 | 19 | 0 | 0.0740 | 0 | 227.2 | 6.16 |
| reverb | builtin | 19 | 0 | 0.0757 | 0 | 189.1 | 4.85 |

Read the WER column with these notes:

- **One chunk accounts for half of the `reverb` and `supertonic` totals.** This is `pd_cadiz` chunk 1.
  - For both models Whisper returned only "Artículo 4º", the last two words of a 24-word chunk. That costs 22 edits.
  - Both clips have continuous speech energy, and the same chunk fails in the same way on two different engines.
    So the failure is on the ASR side, not a dropped reading.
  - Without that chunk, `reverb` is at 0.0395 and `supertonic` at 0.0377.
- **The `pd_cervantes` errors are shared by every model that read it.** Whisper hears "della" as "de ella" and
  "sayo" as "ensayo". These are 17th-century forms that Whisper normalizes.
- **RTFx was measured while other workloads ran on the Mac** (load average around 10 to 16). Read it as an
  order of magnitude:
  - `kokoro_mlx` and `paradee`: 40-60x realtime.
  - `supertonic` and `reverb`: about 5-6x.
  - `qwentts_06b_custom`, `zonos3_mini` and `lfm3_audio`: below realtime on CPU.

## Caveats

- **Single rig, CPU paths.** The RTFx figures are this Mac's CPU (or MLX for kokoro). They are not comparable
  to the CUDA rows on the speed board.
- **WER is a failure detector, not a fine ranking.** On long-form text it mostly separates "reads every word"
  from "drops or garbles some". Naturalness is not measured here. UTMOS was computed during this work but is
  left out on purpose, because it is English-trained and marks non-English speech down (see `docs/tasks.md`).
- **Preset voices only.** No cloning lens.

## What failed or degraded on macOS ARM

- **`zonos3_mini` (ggml Q4, CPU): partial long-form run.**
  - It completed `pd_galdos` and `pd_becquer` and the first `pd_cervantes` chunk: 12 chunks, all scored.
  - The next chunk passed the 300 s runner timeout.
  - The runner was then restarted for `pd_cadiz` and timed out again. Its server log showed ZAC decode RTF climbing
    from 26 to 39 in an earlier attempt.
  - Both failures are recorded as failed rows; the other models' results are kept.
- **`lfm3_audio` (CPU): works, but at RTFx 0.06-0.10** depending on host load.
- **Spanish speaker note for `qwentts_06b_custom`.** The CustomVoice checkpoint reads Spanish cleanly with
  `--language es` and the male speaker `aiden`; its four other male speakers had a similar WER on a two-passage
  screen. Its `harness.MODELS` row lists `{"en", "fr"}` only, so the bench's own p6 Spanish lens never runs it.
- **Every install worked.** The selectors `supertonic`, `kokoro_mlx`, `paradee`, `reverb`, `lfm3_audio`,
  `qwentts`, `zonos3_mini` and `scoring` each installed on macOS ARM with `./install.sh <slug>`.
