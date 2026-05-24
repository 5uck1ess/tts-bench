# Bench results (May 2026)

Full per-rig, per-prompt result tables. For the headline numbers and key takeaways, see the [main README](../README.md). For the live audio of every cell, see [the Demos site](https://5uck1ess.github.io/tts-bench/).

Same five prompts run on both rigs above. Numbers shown are from short prompts; long prompts scale RTF linearly. Warm averages over runs 2-3 across all prompts the model can speak.

### Windows desktop — Ryzen 9 9950X3D CPU

#### Predefined-voice models (pick from a baked-in voice list)

| Model | Size / License | TTFA cold | TTFA warm | RTF warm | Languages | Notes |
|---|---|---|---|---|---|---|
| **[Piper](https://github.com/OHF-voice/piper1-gpl)** (OHF-voice, formerly rhasspy) | per-voice ~25MB / MIT | **72ms** | **39ms** | **47×** | 40+ via separate voice models | leader on this hardware; streaming-native, bundles espeak-ng (no Windows wheel pain) |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** (hexgrad) | 82M / Apache 2.0 | 335ms | 245ms | 13× | 9 (a/b/e/f/h/i/j/p/z codes) | 54 voices; misaki tokenizer needs spaCy preinstall (see Known issues) |
| **[KittenTTS](https://github.com/KittenML/KittenTTS)** (KittenML) | <100M / Apache 2.0 | 516ms | 487ms | 6.6× | EN only | 8 voices; non-streaming so TTFA == gen_s |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** (Microsoft, community fork) | 0.5B / MIT | ~3.9s | ~3.7s | **~0.5×** | EN only (7 preset voices) | streaming-class but heavy diffusion; DDPM steps tunable (5 default). Predefined `.pt` voice embeddings auto-downloaded |
| [Magpie-TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) (NVIDIA, NeMo) | 357M / NVIDIA Open Model License | pending | pending | pending (CUDA: 1.7× warm — see below) | 9 (en/es/de/it/vi/zh/fr/hi/ja) | fixed speaker embeddings (this checkpoint variant); HF accept-terms gated; install skips `[tts]` extra to avoid `pynini` on Windows — runner forces `apply_TN=False` to compensate |
| **[Supertonic](https://github.com/supertone-inc/supertonic)** (Supertone Inc., ONNX) | ~99M / MIT code + OpenRAIL-M weights | 560ms | **509ms** | **6.1×** | 31 (ar/bg/hr/cs/da/nl/en/et/fi/fr/de/el/hi/hu/id/it/ja/ko/lv/lt/pl/pt/ro/ru/sk/sl/es/sv/tr/uk/vi) | pure-ONNX runtime, no torch dep — tiny venv. Open-weight release is fixed-voice only (cloning via hosted Voice Builder / Supertone Play API). 44.1 kHz output |

#### Zero-shot voice cloning models (accept a reference wav at inference time)

| Model | Size / License | TTFA cold | TTFA warm | RTF warm | Cloning ref | Notes |
|---|---|---|---|---|---|---|
| **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** (Kyutai, predefined mode) | 100M / MIT | 95-150ms | 97-150ms | **2.9-3.0×** | wav or voice name | 26 voices unauth; BYO-voice path is HF accept-terms gated on `kyutai/pocket-tts` |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M / Apache 2.0 | 1.2s | 0.43-0.51s | 1.3-1.4× | wav + transcript | multilingual fallback; separate `.gguf` per language |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M / Apache 2.0 | 1.7-1.9s | 0.67-0.70s | 0.88-0.90× | wav + transcript | below realtime on CPU — needs GPU |
| [ChatterBox-TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI) | ~1.2B / MIT | ~8s | ~8s | **~0.30×** | wav (no transcript) | 1000 diffusion steps — GPU-targeted, community quality leader |
| [F5-TTS](https://github.com/SWivid/F5-TTS) (v1 Base) | ~330M / MIT | ~48s | ~48s | **~0.05×** | wav + transcript | flow matching, very slow on CPU; needs GPU |
| [Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS) (idiap fork) | ~750M / CPML 1.0 (non-commercial) | pending | pending | pending | wav (no transcript) | de facto multilingual cloning baseline; ~2GB download on first use; auto-accepts CPML via `COQUI_TOS_AGREED=1` |
| [OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa) | TBD / see upstream | pending | pending | pending | wav + transcript | 600+ languages; diffusion-LM, vendor-claimed 0.025× RTF (GPU); voice design tags (gender/age/whisper) |
| [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (OpenBMB) | 2B / see upstream | pending | pending | pending | wav (no transcript) | tokenizer-free, 48kHz, 30 langs; in-process via `voxcpm` pip pkg (not the optional Nano-vLLM server path). Earlier 0.5B variant doesn't support cloning — skipped. |
| [Qwen3-TTS-Base 1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) (Alibaba Qwen) | 1.7B / Apache 2.0 | pending | pending | pending | wav + transcript | 10 langs (zh/en/ja/ko/de/fr/ru/pt/es/it); claimed 97ms streaming TTFA; FlashAttention 2 skipped on Windows |
| [IndexTTS-2](https://github.com/index-tts/index-tts) (Bilibili Index) | ~1.5B / Apache 2.0 | pending | pending | pending | wav (no transcript) | zero-shot cloning + optional emotion-reference conditioning; source-clone install (no pip wheel); ~5 GB weights auto-download from HF on first use |
| [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) (Sesame AI) | 1B / Apache 2.0 | pending | pending | pending | wav + transcript (as prior-turn context) | conversational speech model; in-context cloning via apply_chat_template; native transformers >= 4.52.1; **HF manual-approval gated** — request access on the model page before first run |
| [MARS5-TTS](https://github.com/Camb-ai/MARS5-TTS) (CAMB.AI) | ~1.2B (750M AR + 450M NAR) / **AGPL-3.0** | pending | pending | pending | wav (shallow clone) or wav + transcript (deep clone, higher quality) | English-only; loaded via `torch.hub.load`; reference audio must be 1-12 seconds; **AGPL-3.0 means non-commercial unless you license from CAMB.AI** |
| [LuxTTS](https://github.com/ysharma3501/LuxTTS) (k2-fsa-based) | — | — | — | — | wav | install blocked on Windows (see [known-issues.md](known-issues.md)) |

**Reading the tables:** TTFA = milliseconds until the first audio sample. RTF = `audio_seconds / generation_seconds` (1.0× = realtime, higher = faster than realtime). Non-streaming models (KittenTTS, ChatterBox, F5-TTS) emit full audio in one call so TTFA = gen_s by definition.

**Top-line takeaway:** if you don't need voice cloning, **Piper wins by a huge margin on this CPU** (39ms warm TTFA, 47× RTF). Pocket-TTS is the fastest cloning-capable option (with the HF accept-terms caveat). NeuTTS Air/Nano give clean BYO-voice without auth gates but at lower RTF. ChatterBox + F5-TTS are GPU-class — file them under "bench-cold but not deployable" until 5090 runs land.

### Mac — Apple M4 (10C, 16 GB) CPU + MPS

Same harness, 5 prompts × 3 runs each, warm averages across all runnable prompts. ChatterBox / F5-TTS / Coqui XTTS not run on this rig — already labeled GPU-class for CPU; M4 CPU would be even worse.

#### Predefined-voice models

| Model | Device | TTFA cold | TTFA warm | RTF warm | Notes |
|---|---|---|---|---|---|
| **[Piper](https://github.com/OHF-voice/piper1-gpl)** | cpu | **268ms** | **202ms** | **33.0×** | still the leader. 5× slower TTFA than Ryzen 9 but well above the headroom needed for an always-on agent |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** | mps | 2995ms | 486ms | **15.4×** | MPS gives ~50% RTF lift over CPU after warmup; cold-load tax (~3s) hits the first turn |
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** | cpu | 994ms | 741ms | 10.2× | |
| **[KittenTTS](https://github.com/KittenML/KittenTTS)** | cpu | 929ms | 1031ms | 8.0× | non-streaming so TTFA == gen_s; CPU-only (no MPS path in upstream) |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** | mps | 10760ms | 8287ms | **1.1×** | finally at realtime on MPS — CPU can't get there. Still 10s+ first-turn cold load |
| **[VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice)** | cpu | 30305ms | 25519ms | 0.4× | below realtime on M4 CPU (Ryzen 9 hit ~0.5×; M4's fewer cores hurt diffusion) |
| **[Magpie-TTS Multi 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)** (NVIDIA NeMo) | cpu | 26716ms | 27459ms | 0.4× | 9 langs (en/es/de/it/vi/zh/fr/hi/ja). HF-gated; works once `hf auth login` is done. NeMo CPU is heavy — close to RTX 5090's 0.97× drops to 0.4× on M4 |

#### Zero-shot voice cloning models

| Model | Device | TTFA cold | TTFA warm | RTF warm | Cloning ref | Notes |
|---|---|---|---|---|---|---|
| **[Pocket-TTS](https://github.com/kyutai-labs/pocket-tts)** (predefined mode) | cpu | **77ms** | **42ms** | **7.8×** | wav or voice name | fastest cloning-capable option here too. M4 single-thread perf actually beats Ryzen 9 on TTFA |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | cpu | 815ms | 270ms | 3.0× | wav + transcript | multilingual via separate `.gguf` per language |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | mps | 1491ms | 444ms | 2.8× | wav + transcript | MPS gives no win — GGUF inference runs CPU-side via llama-cpp |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | cpu | 1436ms | 364ms | 2.1× | wav + transcript | ~2.4× faster than Windows numbers thanks to M4 single-thread |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | mps | 2399ms | 568ms | 2.1× | wav + transcript | same — MPS doesn't help GGUF path |
| **[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)** (k2-fsa, 600+ langs) | mps | 5802ms | 5064ms | **0.9×** | wav + transcript | nearly realtime on MPS for short/medium prompts. **Long prompts (30+ words) OOM the MPS allocator** on 16 GB at ~3.4 GiB — 1 of 5 prompts failed |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (k2-fsa, 600+ langs) | cpu | 13653ms | 11444ms | 0.6× | wav + transcript | below realtime on CPU, expected for diffusion-LM |
| [VoxCPM-0.5B](https://huggingface.co/openbmb/VoxCPM-0.5B) (OpenBMB) | cpu | 10883ms | 9582ms | 0.7× | wav only | no MPS path in harness; reasonably close to RTX 5090's 1.0× — VoxCPM is less GPU-dependent than the others |
| [LuxTTS](https://github.com/ysharma3501/LuxTTS) (zipvoice-based) | — | — | — | — | wav | install blocked on arm64 Mac too (see [known-issues.md](known-issues.md)) |

**Top-line takeaway on Mac:** Piper wins again (33× RTF, 202ms warm TTFA — drop-in for an always-on agent). Among cloning models, **Pocket-TTS is the clear winner on M4** — its 42ms warm TTFA actually beats the Windows number, because Pocket-TTS is single-thread dominated and M4 has strong single-thread perf. VibeVoice/MPS is the only diffusion-class model that reaches realtime on this machine; CPU diffusion isn't viable. NeuTTS gets no MPS benefit because its hot path is GGUF (llama-cpp, CPU-side). The new GPU-class additions (OmniVoice, VoxCPM, Magpie) all land sub-realtime on M4 — useful as "works at all on a Mac" data points, not as deploy candidates.

### Windows desktop — RTX 5090 CUDA

Same 5 prompts × 3 runs. Warm averages over the warm runs across all prompts the model can speak. `VRAM` is `torch.cuda.max_memory_allocated()` for the cold run. Pocket-TTS, KittenTTS, and Supertonic are CPU-only (skipped). LuxTTS install is still blocked on Windows.

#### Predefined-voice models

| Model | Size | TTFA cold | TTFA warm | RTF warm | VRAM | Notes |
|---|---|---|---|---|---|---|
| **[Kokoro-82M](https://github.com/hexgrad/kokoro)** | 82M | **926ms** | **69ms** | **101×** | 0.7 GB | clear winner on this rig; warm TTFA matches Piper/cpu while RTF is 2× higher |
| [VibeVoice-Realtime-0.5B](https://github.com/vibevoice-community/VibeVoice) | 0.5B | 4702ms | 4568ms | 2.1× | 2.6 GB | finally hits comfortable realtime on GPU (CPU was 0.5×). Diffusion cold-load tax still ~4.7s |
| [Magpie-TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | 357M | 6628ms | 5016ms | 1.7× | 3.6 GB | 9 langs; slow first turn (NeMo init), warm RTF reasonable. HF accept-terms gated |

#### Zero-shot voice cloning models

Two runs reported per model: **default voice** (the cloning model with its bundled fallback wav) and **cloning** (`--reference reference/chris_hemsworth_15s.wav` + matching transcript). Cloning numbers are usually slightly slower because the 15-second reference is longer than each model's bundled default.

| Model | Size | Mode | TTFA cold | TTFA warm | RTF warm | VRAM | Notes |
|---|---|---|---|---|---|---|---|
| **[OmniVoice](https://github.com/k2-fsa/OmniVoice)** (k2-fsa, 600+ langs) | ~1B | default | **1177ms** | **811ms** | **8.5×** | 2.0 GB | top RTF on long prompts (20× warm on the Parakeet paragraph). Same wins on cloning (9.2×). |
| OmniVoice | ~1B | cloning | 1267ms | 869ms | **9.2×** | 2.4 GB | same fast as default; consistently the fastest cloning model on this hardware |
| **[F5-TTS](https://github.com/SWivid/F5-TTS)** (v1 Base) | 330M | default | 1319ms | 872ms | **5.2×** | 0.8 GB | second-fastest cloner; smallest VRAM footprint of the cloning models |
| F5-TTS | 330M | cloning | 1597ms | 1096ms | 5.3× | 0.8 GB | |
| **[Coqui XTTS-v2](https://github.com/idiap/coqui-ai-TTS)** | 750M | default | 2433ms | 2113ms | 4.2× | 2.0 GB | multilingual cloning baseline (CPML 1.0 non-commercial) |
| Coqui XTTS-v2 | 750M | cloning | 2496ms | 1789ms | 4.1× | 1.9 GB | |
| [NeuTTS Nano](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M | default | 658ms | 273ms | 2.7× | 3.3 GB | best TTFA among cloning models; GGUF runs CPU-side so VRAM stays small. EN/FR/DE/ES |
| NeuTTS Nano | 748M | cloning | 741ms | 306ms | 2.5× | 3.3 GB | |
| [ChatterBox-TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI) | 1.2B | default | 3645ms | 2842ms | 2.0× | 3.1 GB | community quality leader for cloning; 1000-step diffusion is the wall-time tax |
| ChatterBox-TTS | 1.2B | cloning | 4468ms | 3333ms | 2.0× | 3.5 GB | |
| [NeuTTS Air](https://github.com/neuphonic/neutts) (GGUF Q4) | 748M | default | 1156ms | 434ms | 1.6× | 3.3 GB | EN only; same GGUF CPU-side path as Nano |
| NeuTTS Air | 748M | cloning | 1202ms | 448ms | 1.4× | 3.3 GB | |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (OpenBMB, 30 langs) | 2B | default | 5639ms | 5658ms | 1.2× | 5.4 GB | tokenizer-free 48 kHz output; 5+ GB VRAM is the heaviest in the cloning set |
| VoxCPM2 | 2B | cloning | 7445ms | 5081ms | 1.2× | 5.7 GB | |
| [IndexTTS-2](https://github.com/index-tts/index-tts) (Bilibili Index) | 1.5B | default | 7065ms | 5966ms | 1.1× | 7.2 GB | source-clone install; ~5 GB weights; emotion-conditioning support |
| IndexTTS-2 | 1.5B | cloning | 7486ms | 6072ms | 1.1× | 7.3 GB | |
| [Qwen3-TTS-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | default | 11253ms | 10096ms | 0.6× | 4.5 GB | 10-lang cloning; warm RTF below realtime — slower than vendor's claimed 97ms streaming TTFA |
| Qwen3-TTS Base | 1.7B | cloning (short) | 4420ms | 3247ms | 0.6× | 4.4 GB | **only completed 1/5 prompts** at 600s timeout — 15s reference + long text pushes generation past 5-10min per cell |
| [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) | 1B | default | 14563ms | 15925ms | 0.5× | 3.5 GB | conversational TTS via ChatML; HF manual-approval gated. Mimi codec 24kHz |
| Sesame CSM-1B | 1B | cloning | 14866ms | 12321ms | 0.5× | 3.5 GB | |
| [MARS5-TTS](https://github.com/Camb-ai/MARS5-TTS) (CAMB.AI) | 1.2B | default | 32717ms | 32115ms | 0.2× | 6.0 GB | unexpectedly slow on CUDA — torch.hub model load + AR generation. AGPL-3.0 |
| MARS5-TTS | 1.2B | cloning | 43620ms | 39235ms | 0.1× | 6.9 GB | |
| [Pocket-TTS](https://github.com/kyutai-labs/pocket-tts) | 100M | n/a | — | — | — | — | CPU-only (no CUDA path in the harness) |

**Top-line takeaway on CUDA:**

- *Predefined voices:* **Kokoro on the 5090 is the deploy-target answer** — 69ms warm TTFA, 101× RTF on short prompts, 0.7 GB VRAM. Lower latency than Piper/cpu and roughly 2× the RTF.
- *Cloning, fastest:* **OmniVoice** — 869ms warm TTFA at 9.2× RTF on the Chris Hemsworth reference, scaling to 20× on long prompts. F5-TTS is the runner-up (5.3× warm RTF, smallest VRAM at 0.8 GB).
- *Cloning, slowest:* MARS5 at 0.1× RTF and Qwen3-TTS Base at 0.5-0.6× — both genuinely sub-realtime on a 5090, with Qwen also hitting the 10-min per-cell wall on the 15-second reference (only the shortest prompt completed for cloning).
- *VRAM budget:* the heaviest cloning models (IndexTTS-2 at 7.3 GB, MARS5 at 6.9 GB, VoxCPM2 at 5.7 GB) all comfortably fit a 16 GB GPU; the 32 GB on this rig is overkill.

Raw CSVs live alongside their reports — Mac runs are in `_gh-pages/2026-05-23_*` (since `results/` is gitignored and per-machine).

Caveats: one machine, one run. Re-bench on your own hardware before committing — see [known-issues.md](known-issues.md) for examples of model README claims that didn't survive contact with a real install.
