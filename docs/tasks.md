# Tasks & pending work

> Open issues + planned features that aren't yet GitHub issues. Move to GitHub when they sharpen into actionable items.

## Active

- [x] **Zonos2 — install + speed bench + scoring on the Linux-3090. DONE 2026-07-21** (branch `zonos2-add`). Installed (`./install.sh zonos2` — `uv sync` builds the sgl-kernel/flashinfer CUDA stack), benched (cloning the Chris ref, cuda), scored, merged into `results/linux-cloning`. **Speed:** warm RTFx ~2.0–2.1× (rises with clip length as fixed engine overhead amortizes), cold TTFA 2.5–8.3 s. **Scores:** UTMOS 4.282 / WER 0.082 (p3 numeric prompt 0.27 — genuine number-reading errors, not truncation) / **SIM 0.355 (low — weak Chris clone; UTMOS quality strong but speaker similarity is well below the top cloners)**. **First-run VERIFY findings (all resolved in `zonos2_runner.py`):** (1) `TTSLLM("Zyphra/ZONOS2")` auto-selects CUDA, no device kwarg; (2) `result["audio"]` is raw PCM float32 **bytes**, NOT a tensor — the draft `_audio_len` collapsed it to 1 sample → RTFx read ~0; **fixed** to `len(bytes)//4`; (3) `TTSLLM` spins up an internal sglang-style engine (paged KV cache sized to fill VRAM → ~21 GB, whole-GPU `peak_vram_mb` like vLLM/Orpheus); (4) **two long-form defaults caught by ear** (p3 audibly cut off mid-paragraph): `generate_one`'s default `max_tokens=1024` frames capped output at ~11.9 s (server auto-raises to the model limit, in-process path does not) — **fixed** via `resolve_max_tokens()`; and with the cap lifted the default `memory_ratio=0.9` left only ~2.1 GB free so the DAC vocoder OOM'd on the full clip — **fixed** with `memory_ratio=0.8` (~4.5 GB headroom). p3 now renders complete at ~14.4 s. Remaining: merge `zonos2-add` → master + publish gh-pages (Linux owns the full publish — Windows-incapable); move out of considered.md's queued section.
- [ ] Subjective listening pass on predefined-voice tier (CUDA) — cloning got the full ranking; predefined didn't
- [ ] MARS5 CUDA investigation — 0.1× RTFx + cloning that doesn't match reference. Both unusable. Needs deeper look or "skipped after investigation" entry in docs/considered.md
- [ ] Qwen3-TTS Base cloning timeout on long prompts — at the 15s Chris Hemsworth ref, prompts 2-5 hit the 10-min per-cell wall
- [ ] ChatterBox / F5-TTS / Coqui XTTS on Mac MPS — skipped earlier (all GPU-class). Worth re-running once MPS torch perf improves
- [ ] LuxTTS on Mac arm64 / Windows — depends on piper-phonemize, no wheels for those platforms

## Model queue (candidates, not yet evaluated)

- [ ] **HumeAI TADA** — `HumeAI/tada-1b` + `HumeAI/tada-3b-ml` (multilingual 3B); speech-language model claiming 700s+ coherent audio via text-acoustic dual alignment, ~10 languages. Official MLX variants exist (`HumeAI/mlx-tada-1b/-3b`) → Mac path. Surfaced via voicebox's engine list (2026-06-11). Check license + clone support at evaluation time.
- [ ] **ChatterBox Multilingual** — 23 languages via `ChatterboxMultilingualTTS` in the same `chatterbox` pip pkg we already install (multilingual weights in the main `ResembleAI/chatterbox` repo; per-language fine-tunes like `Chatterbox-Multilingual-hi` also on HF). Caveat against the inclusion bar: bench prompts are English-only, so its whole value-add (the other 22 languages) wouldn't show in scores — English output likely ≈ base ChatterBox, which we bench.
- [ ] **Supertonic 1 / 2** — community ask (2026-06-09); same pip pkg as Supertonic 3 via the model arg, so install cost is near zero.

### Linux-rig pass — re-queued from "skipped (Windows-blocked)"

These were skipped on the **Windows-primary** judgement (vLLM has no Blackwell wheel / Linux-only deps), not because they fail on Linux. The 2026-05-24 Linux-3090 rig makes them re-evaluable. Ranked by value-for-effort; attempt each in its own venv. See [considered.md](considered.md) for the original skip detail.

- [x] **Orpheus TTS** (`canopylabs/orpheus-3b-0.1-ft`, 3B Llama + SNAC) — **DONE 2026-06-16** (3be5027 / 08f5020, published linux-default). `orpheus-speech` co-resolved a *modern* stack (vllm 0.23 / torch 2.11), not the feared 0.7.3 — the vLLM v1 engine works fine. Preset-voice only (`can_clone=False`), English, RTFx ~1.0× (warm TTFA ~360 ms). Gotchas: gated HF repo (license-accept once); vLLM runs the model in a spawned EngineCore subprocess → drive it on one persistent event loop (asyncio.run-per-call hangs the warm run + leaks the GPU) and report whole-GPU `gpu_used_mb` (process-local VRAM is blind).
- [x] **CosyVoice 3** (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) — **DONE 2026-06-16** (chose v3 over v2 per Tym — v3 is newer/better; the cu121/torch-2.3.1 blocker was a 5090/Blackwell issue, native on the Ampere 3090). Source clone + Matcha-TTS submodule, py3.10. Pure zero-shot cloning (no preset → house-ref default lens, `NO_PRESET_VOICE`), multilingual, RTFx 0.2-2.4× (LLM sampling, variable). Install gotchas: `--index-strategy unsafe-best-match` (protobuf), `setuptools<81` (pkg_resources), staged `--no-build-isolation`. **v2 not pursued.** Known issue: p1 cold clip truncates (WER 1.0) — candidate QA warn-badge / re-bench.
- [ ] **GLM-TTS** (`zai-org/GLM-TTS`) — *low priority.* Linux-viable but Chinese-centric (ZH/EN) and drags in deepspeed + funasr + pynini. Only worth it if multilingual/Chinese coverage becomes a bench goal.
- [ ] **Step-Audio-TTS-3B** (`stepfun-ai/Step-Audio-TTS-3B`) — *low priority / likely dead end.* Linux-only compiled `.so` extensions make it natively Linux, but upstream github is 404 (abandoned), successor isn't TTS, ~70 downloads/month. Attempt only if the bundled `.so` still loads via `trust_remote_code` on the current torch. NOT the same as `step_editx` (already in).
- [ ] **VibeVoice-Large 9B** (`aoi-ot/VibeVoice-Large`) — quality-ceiling; ~20-24 GB fp16 is **borderline on the 3090's 24 GB** with our 100+ token prompts. Fit-check (does KV cache fit?), not a clean add — or ship a 4-bit GGUF variant. See considered.md "Future enhancement".

## Tuning notes (documented, not actioned)

- [ ] **Fish S2 temperature** — community tip (Reddit, 2026-06-23): Fish S2 "stays stable at high temp but sounds much more alive than the defaults." A/B'd on the cloning ref at temp 0.80 (bench default) / 0.95 / 1.00, scored through our lenses: **higher temp does not score better.** Per-temp avg — UTMOS 4.232 / 4.221 / 4.202 (flat, slight drop), WER 0.040 / 0.071 / 0.040 (0.95 worst), SIM 0.707 / 0.700 / **0.683** (1.0 measurably erodes clone fidelity; the long-prompt clip audibly drops in volume at 1.0). Bench kept at 0.8. The "more alive" quality is real but our objective lenses can't measure expressivity (same gap as the NAQ-redesign note below — UTMOS rewards stationary tonal output), so it's a real-world usage lever, not a bench win. `fish_s2_runner.py` now exposes `--temperature` (default 0.8) for ad-hoc tuning.

## Future

- [ ] **NAQ redesign (v3)** — current ARTIFACT macro (HNR-driven) anti-correlates with expressive speech: punishes natural noisiness (breath, plosives, sibilance) and rewards stationary tonal output. NATURALNESS macro rewards raw F0 variance regardless of whether it's linguistically appropriate. Needs features that proxy what they're named for, not what's easy to measure. NAQ has been pulled from the bench entirely — redesign happens offline over the saved wavs, not during a run.
- [ ] **Community voting system** — head-to-head sample voting on the gh-pages reports → labeled ground truth for NAQ refinement
- [ ] Prefer 48 kHz output models for new additions (preference, not hard gate)
- [ ] **Streaming bench lens** — separate harness track measuring sub-sentence latency (time-to-first-chunk, mean inter-chunk gap, drift). Unlocks fair benching of streaming/real-time models: MOSS-TTS-Realtime, VibeVoice Realtime 0.5B (currently benched in non-streaming mode), and future streaming entries.
- [ ] Buy Me a Coffee / Sponsor link (defer until external traction)
