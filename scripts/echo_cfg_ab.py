"""Echo-TTS independent-CFG vs joint-CFG A/B (audio.cpp issue #739). Experiment only.

Joint sampler = minimal diff of upstream sample_euler_cfg_independent_guidances:
same masked-KV "uncond" construction, 2 lanes (cond | text+speaker masked) instead
of 3, v = c + s*(c - u). Everything else (schedule, cfg window, truncation, seed ->
x_t) is identical, so prompt x seed pairs share initial noise across modes.

Run with the echo venv:  venvs/echo/Scripts/python scripts/echo_cfg_ab.py --out DIR
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "runners"))
import echo_runner as er  # noqa: E402
import _meminfo  # noqa: E402

er._stub_uninstalled_audio_libs()
sys.path.insert(0, str(er.ECHO_SRC))
import torch  # noqa: E402
from inference import (  # noqa: E402
    load_model_from_hf, load_fish_ae_from_hf, load_pca_state_from_hf, sample_pipeline,
    sample_euler_cfg_independent_guidances, _concat_kv_caches,
)

PROMPTS = {1: "Open the browser and read my email.",
           2: "I'll start a new git branch, push the changes, and open a pull request when the tests pass.",
           3: ("The Parakeet TDT zero point six billion parameter model achieves "
               "one point six nine percent word error rate on LibriSpeech test-clean, "
               "beating Whisper Large V3 at two point seven percent while running at "
               "over two thousand times realtime on a single GPU."),
           4: "Run pytest tests slash test underscore voice dot py with verbose flag and capture flag set to no."}


@torch.inference_mode()
def sample_euler_cfg_joint(model, speaker_latent, speaker_mask, text_input_ids, text_mask,
                           rng_seed, num_steps, cfg_scale, cfg_min_t, cfg_max_t,
                           truncation_factor, sequence_length=640):
    INIT_SCALE = 0.999
    device, dtype = model.device, model.dtype
    batch_size = text_input_ids.shape[0]
    rng = torch.Generator(device=device).manual_seed(rng_seed)
    t_schedule = torch.linspace(1., 0., num_steps + 1, device=device) * INIT_SCALE

    kv_text_cond = model.get_kv_cache_text(text_input_ids, text_mask)
    kv_speaker_cond = model.get_kv_cache_speaker(speaker_latent.to(dtype))
    kv_text_full = _concat_kv_caches(kv_text_cond, kv_text_cond)
    kv_speaker_full = _concat_kv_caches(kv_speaker_cond, kv_speaker_cond)
    full_text_mask = torch.cat([text_mask, torch.zeros_like(text_mask)], dim=0)
    full_speaker_mask = torch.cat([speaker_mask, torch.zeros_like(speaker_mask)], dim=0)

    x_t = torch.randn((batch_size, sequence_length, 80), device=device, dtype=torch.float32, generator=rng)
    if truncation_factor is not None:
        x_t = x_t * truncation_factor

    for i in range(num_steps):
        t, t_next = t_schedule[i], t_schedule[i + 1]
        if ((t >= cfg_min_t) * (t <= cfg_max_t)).item():
            v_cond, v_uncond = model(
                x=torch.cat([x_t, x_t], dim=0).to(dtype),
                t=(torch.ones((batch_size * 2,), device=device) * t).to(dtype),
                text_mask=full_text_mask, speaker_mask=full_speaker_mask,
                kv_cache_text=kv_text_full, kv_cache_speaker=kv_speaker_full,
            ).float().chunk(2, dim=0)
            v_pred = v_cond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = model(
                x=x_t.to(dtype), t=(torch.ones((batch_size,), device=device) * t).to(dtype),
                text_mask=text_mask, speaker_mask=speaker_mask,
                kv_cache_text=kv_text_cond, kv_cache_speaker=kv_speaker_cond,
            ).float()
        x_t = x_t + v_pred * (t_next - t)
    return x_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--joint-scales", default="2,3,4,6,8,10")
    ap.add_argument("--prompts", default="1,2,3,4")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    prompts = [int(p) for p in args.prompts.split(",")]

    common = dict(num_steps=40, cfg_min_t=0.5, cfg_max_t=1.0, truncation_factor=0.8, sequence_length=640)
    configs = {"indep": partial(sample_euler_cfg_independent_guidances, cfg_scale_text=3.0,
                                cfg_scale_speaker=8.0, rescale_k=None, rescale_sigma=None,
                                speaker_kv_scale=None, speaker_kv_max_layers=None,
                                speaker_kv_min_t=None, **common)}
    for s in args.joint_scales.split(","):
        configs[f"joint{s}"] = partial(sample_euler_cfg_joint, cfg_scale=float(s), **common)

    model = load_model_from_hf(device="cuda", delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf(device="cuda")
    pca_state = load_pca_state_from_hf(device="cuda")
    spk = er._load_ref_audio(er.DEFAULT_REF, "cuda")

    # warmup (not recorded)
    sample_pipeline(model=model, fish_ae=fish_ae, pca_state=pca_state, sample_fn=configs["indep"],
                    text_prompt=er._tag(PROMPTS[1]), speaker_audio=spk, rng_seed=99)

    import soundfile as sf
    rows = open(out / "runs.jsonl", "a", encoding="utf-8")
    for pid in prompts:
        for seed in seeds:
            for name, fn in configs.items():
                _meminfo.reset_peak("cuda")
                torch.cuda.synchronize(); t0 = time.perf_counter()
                audio, _ = sample_pipeline(model=model, fish_ae=fish_ae, pca_state=pca_state, sample_fn=fn,
                                           text_prompt=er._tag(PROMPTS[pid]), speaker_audio=spk, rng_seed=seed)
                torch.cuda.synchronize(); gen_s = time.perf_counter() - t0
                clip = audio[0].detach().cpu().float()
                clip = clip.unsqueeze(0) if clip.dim() == 1 else clip
                wav = out / f"{name}_p{pid}_s{seed}.wav"
                sf.write(str(wav), clip.numpy().T, er.SAMPLERATE)
                row = {"config": name, "prompt": pid, "seed": seed, "wav": wav.name, "gen_s": gen_s,
                       "audio_s": clip.shape[-1] / er.SAMPLERATE, **_meminfo.sample("cuda")}
                rows.write(json.dumps(row) + "\n"); rows.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
