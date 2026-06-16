# Tasks & pending work

> Open issues + planned features that aren't yet GitHub issues. Move to GitHub when they sharpen into actionable items.

## Active

- [ ] Subjective listening pass on predefined-voice tier (CUDA) — cloning got the full ranking; predefined didn't
- [ ] MARS5 CUDA investigation — 0.1× RTF + cloning that doesn't match reference. Both unusable. Needs deeper look or "skipped after investigation" entry in docs/considered.md
- [ ] Qwen3-TTS Base cloning timeout on long prompts — at the 15s Chris Hemsworth ref, prompts 2-5 hit the 10-min per-cell wall
- [ ] ChatterBox / F5-TTS / Coqui XTTS on Mac MPS — skipped earlier (all GPU-class). Worth re-running once MPS torch perf improves
- [ ] LuxTTS on Mac arm64 / Windows — depends on piper-phonemize, no wheels for those platforms

## Model queue (candidates, not yet evaluated)

- [ ] **HumeAI TADA** — `HumeAI/tada-1b` + `HumeAI/tada-3b-ml` (multilingual 3B); speech-language model claiming 700s+ coherent audio via text-acoustic dual alignment, ~10 languages. Official MLX variants exist (`HumeAI/mlx-tada-1b/-3b`) → Mac path. Surfaced via voicebox's engine list (2026-06-11). Check license + clone support at evaluation time.
- [ ] **ChatterBox Multilingual** — 23 languages via `ChatterboxMultilingualTTS` in the same `chatterbox` pip pkg we already install (multilingual weights in the main `ResembleAI/chatterbox` repo; per-language fine-tunes like `Chatterbox-Multilingual-hi` also on HF). Caveat against the inclusion bar: bench prompts are English-only, so its whole value-add (the other 22 languages) wouldn't show in scores — English output likely ≈ base ChatterBox, which we bench.
- [ ] **Supertonic 1 / 2** — community ask (2026-06-09); same pip pkg as Supertonic 3 via the model arg, so install cost is near zero.

### Linux-rig pass — re-queued from "skipped (Windows-blocked)"

These were skipped on the **Windows-primary** judgement (vLLM has no Blackwell wheel / Linux-only deps), not because they fail on Linux. The 2026-05-24 Linux-3090 rig makes them re-evaluable. Ranked by value-for-effort; attempt each in its own venv. See [considered.md](considered.md) for the original skip detail.

- [ ] **Orpheus TTS** (`canopyai/Orpheus-TTS`, 3B Llama) — **top priority.** Only blocker was vLLM-no-Windows-Blackwell-wheel; vLLM runs natively on the 3090 (proven by `voxtral`). High-quality emotional speech, ~200 ms streaming. Install path is `pip install orpheus-speech` (pins vllm 0.7.3 — confirm it co-resolves with the rig's CUDA stack, same vLLM lane voxtral uses). considered.md says "revisit on Mac" — disregard; Linux/CUDA is the right target, not Mac.
- [ ] **CosyVoice 2-0.5B** (`FunAudioLLM/CosyVoice2-0.5B`) — target **v2** over v3 (more battle-tested, the listicles' pick). Source clone + `third_party/Matcha-TTS` submodule. Every blocker from the v3 skip (`torch==2.3.1`/cu121, Linux-only onnxruntime + tensorrt feeds, pynini frontend) resolves natively on the Ampere 3090. 40+ exact pins are fragile → isolated venv. Bench v3 too if v2 installs clean.
- [ ] **GLM-TTS** (`zai-org/GLM-TTS`) — *low priority.* Linux-viable but Chinese-centric (ZH/EN) and drags in deepspeed + funasr + pynini. Only worth it if multilingual/Chinese coverage becomes a bench goal.
- [ ] **Step-Audio-TTS-3B** (`stepfun-ai/Step-Audio-TTS-3B`) — *low priority / likely dead end.* Linux-only compiled `.so` extensions make it natively Linux, but upstream github is 404 (abandoned), successor isn't TTS, ~70 downloads/month. Attempt only if the bundled `.so` still loads via `trust_remote_code` on the current torch. NOT the same as `step_editx` (already in).
- [ ] **VibeVoice-Large 9B** (`aoi-ot/VibeVoice-Large`) — quality-ceiling; ~20-24 GB fp16 is **borderline on the 3090's 24 GB** with our 100+ token prompts. Fit-check (does KV cache fit?), not a clean add — or ship a 4-bit GGUF variant. See considered.md "Future enhancement".

## Future

- [ ] **NAQ redesign (v3)** — current ARTIFACT macro (HNR-driven) anti-correlates with expressive speech: punishes natural noisiness (breath, plosives, sibilance) and rewards stationary tonal output. NATURALNESS macro rewards raw F0 variance regardless of whether it's linguistically appropriate. Needs features that proxy what they're named for, not what's easy to measure. NAQ has been pulled from the bench entirely — redesign happens offline over the saved wavs, not during a run.
- [ ] **Community voting system** — head-to-head sample voting on the gh-pages reports → labeled ground truth for NAQ refinement
- [ ] Prefer 48 kHz output models for new additions (preference, not hard gate)
- [ ] **Streaming bench lens** — separate harness track measuring sub-sentence latency (time-to-first-chunk, mean inter-chunk gap, drift). Unlocks fair benching of streaming/real-time models: MOSS-TTS-Realtime, VibeVoice Realtime 0.5B (currently benched in non-streaming mode), and future streaming entries.
- [ ] Buy Me a Coffee / Sponsor link (defer until external traction)
