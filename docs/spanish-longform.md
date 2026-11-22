# Spanish long-form (Apple M4 Max)

The bench's Spanish coverage is one short prompt: p6, about 6 s of speech. The Windows 5090 rig scores it
(`docs/known-issues.md`: default-lens median WER 0.0000, mean UTMOS 3.73). This page adds **long-form**
Spanish. Seven preset-voice models were run on four public-domain passages of a minute or more each; six
completed. That
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
venvs/scoring/bin/python scripts/spanish_longform.py --score results/es-longform/results.csv
python scripts/spanish_longform.py --render-table results/es-longform/results.csv
```

## Results

| model | voice | chunks | failed | WER | Health-flagged chunks | audio (s) | RTFx |
|---|---|---:|---:|---:|---:|---:|---:|
| qwentts_06b_custom | aiden | 19 | 0 | 0.0275 | 0 | 273.7 | 0.73 |
| kokoro_mlx | em_alex | 19 | 0 | 0.0361 | 0 | 191.1 | 45.47 |
| supertonic | M4 | 19 | 0 | 0.0464 | 0 | 227.7 | 6.86 |
| paradee | builtin | 19 | 0 | 0.0481 | 0 | 195.8 | 66.28 |
| lfm3_audio | male | 19 | 0 | 0.0515 | 0 | 191.5 | 0.10 |
| reverb | builtin | 19 | 0 | 0.0825 | 0 | 189.1 | 4.41 |
| zonos3_mini | builtin | 4 | 4 | n/a | 0 | 0.0 | n/a |

Read the WER column with these notes:

- **One chunk dominates `reverb`.** In `pd_cadiz` chunk 1, a 9.6 s clip with continuous speech energy, Whisper
  returned only "Artículo 4º", the last two words. That one chunk costs 23 of `reverb`'s 48 edits; without it,
  `reverb` is at 0.0448. This looks like an ASR failure rather than a dropped reading, but it is not confirmed by ear.
- **Every model shares the `pd_cervantes` errors.** "della" is heard as "de ella" and "sayo" as "ensayo". These
  are 17th-century forms that Whisper normalizes, so they show up for all six models that completed.
- **Speed.** `kokoro_mlx` and `paradee` run 45-66x faster than realtime on this Mac. `qwentts_06b_custom` and
  `lfm3_audio` run below realtime on CPU (0.73x and 0.10x).


## Caveats

- **Single rig, CPU paths.** The RTFx figures are this Mac's CPU (or MLX for kokoro). They are not comparable
  to the CUDA rows on the speed board.
- **WER is a failure detector, not a fine ranking.** On long-form text it mostly separates "reads every word"
  from "drops or garbles some". Naturalness is not measured here. UTMOS was computed during this work but is
  left out on purpose, because it is English-trained and marks non-English speech down (see `docs/tasks.md`).
- **Preset voices only.** No cloning lens.

## What failed or degraded on macOS ARM

- **`zonos3_mini` (ggml Q4, CPU): no usable long-form run.**
  - In a book-length run, one chunk passed the 300 s runner timeout, and the server's own log showed ZAC decode
    RTF climbing from 26 to 39.
  - The next start failed with "zonos3-server not ready after 300s". Its ready probe never finished.
  - Two shorter chunks carried Health flags. The default-lens short prompt (p6) did run.
- **`lfm3_audio` (CPU): works, but at RTFx 0.10.** A 125-minute book would take about 21 h on this Mac.
- **Spanish speaker note for `qwentts_06b_custom`.** The CustomVoice checkpoint read Spanish cleanly with
  `--language es` and a male speaker (`aiden`; the four other male speakers had a similar WER on a two-passage
  screen). The model's `harness.MODELS` row lists `{"en", "fr"}` only, so the bench's own p6 Spanish lens never
  runs it.
- **Every install worked.** The selectors `supertonic`, `kokoro_mlx`, `paradee`, `reverb`, `lfm3_audio`,
  `qwentts`, `zonos3_mini` and `scoring` each installed on macOS ARM with `./install.sh <slug>`.

