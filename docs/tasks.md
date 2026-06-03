# Tasks & pending work

> Open issues + planned features that aren't yet GitHub issues. Move to GitHub when they sharpen into actionable items.

## Active

- [ ] Subjective listening pass on predefined-voice tier (CUDA) — cloning got the full ranking; predefined didn't
- [ ] MARS5 CUDA investigation — 0.1× RTF + cloning that doesn't match reference. Both unusable. Needs deeper look or "skipped after investigation" entry in docs/considered.md
- [ ] Qwen3-TTS Base cloning timeout on long prompts — at the 15s Chris Hemsworth ref, prompts 2-5 hit the 10-min per-cell wall
- [ ] ChatterBox / F5-TTS / Coqui XTTS on Mac MPS — skipped earlier (all GPU-class). Worth re-running once MPS torch perf improves
- [ ] LuxTTS on Mac arm64 / Windows — depends on piper-phonemize, no wheels for those platforms

## Future

- [ ] **NAQ redesign (v3)** — current ARTIFACT macro (HNR-driven) anti-correlates with expressive speech: punishes natural noisiness (breath, plosives, sibilance) and rewards stationary tonal output. NATURALNESS macro rewards raw F0 variance regardless of whether it's linguistically appropriate. Needs features that proxy what they're named for, not what's easy to measure. NAQ has been pulled from the bench entirely — redesign happens offline over the saved wavs, not during a run.
- [ ] **Community voting system** — head-to-head sample voting on the gh-pages reports → labeled ground truth for NAQ refinement
- [ ] Prefer 48 kHz output models for new additions (preference, not hard gate)
- [ ] **Streaming bench lens** — separate harness track measuring sub-sentence latency (time-to-first-chunk, mean inter-chunk gap, drift). Unlocks fair benching of streaming/real-time models: MOSS-TTS-Realtime, VibeVoice Realtime 0.5B (currently benched in non-streaming mode), and future streaming entries.
- [ ] Buy Me a Coffee / Sponsor link (defer until external traction)
