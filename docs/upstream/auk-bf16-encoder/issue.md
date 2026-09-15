# Inference casts the frozen Qwen text encoder back to fp32, causing unnecessary GPU OOM

## Cause

[`src/auk/infer/infer_auk.py`](https://github.com/Tencent-Hunyuan/AuK/blob/main/src/auk/infer/infer_auk.py) builds `CFMEdit` with a `text_encoder` loaded via `torch_dtype=torch.bfloat16`, then calls `model = model.to(torch.float32)`. This recursively re-casts the frozen Qwen encoder to fp32, doubling its parameter storage.

The [README.md](https://github.com/Tencent-Hunyuan/AuK/blob/main/README.md) VRAM table is captioned "with bf16 inference", but reports memory numbers consistent with this fp32 encoder path. The merged PR titled `cpu_offload to save memory` (PR #9) provides a workaround for this cause by moving modules between CPU and GPU; it does not correct the encoder dtype.

## Reproduction

```python
from auk.infer.infer_auk import AukInfer
engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors",
                  qwen_path="ckpts/Qwen2.5-Omni-3B")
te = engine.model.text_encoder
print(next(te.parameters()).dtype,
      sum(p.numel() * p.element_size() for p in te.parameters()) / 2**30, "GiB")
```

Prints `torch.float32` and the fp32 figure in the table below, despite the `torch_dtype=torch.bfloat16`
used to load it. On a 24 GB card, generation then fails during the shortest prompt rather than on a
long one, which is what makes it read as a hard capacity limit rather than a dtype issue.

## Measurements and affected hardware

All measurements below come from `baseline.json` and `bf16.json`, taken on an RTX 5090 with 32 GB. The original OOM was observed separately on an RTX 3090 with 24 GB. That GPU was not re-tested after the fix.

| Measurement | Baseline fp32 encoder | Restored bf16 encoder |
| --- | --- | --- |
| Encoder parameters | 4034780160 | 4034780160 |
| Encoder parameter storage, GiB | 15.031 | 7.515 |
| Peak after load, GiB | 21.443 | 13.966 |
| Overall peak, GiB | 26.054 | 18.582 |

Restoring only the encoder dtype brings the measured workload below the affected GPU's capacity. This identifies the encoder upcast as the cause of the capacity overrun for this workload; a successful post-fix run on the OOM GPU remains unverified.

## Proposed fix

Immediately after the fp32 cast, while the model is still on CPU, add:

```python
model.text_encoder.to(torch.bfloat16)
```

This restores the intended encoder dtype before GPU transfer. The transformer and VAE remain fp32.

## Why the cast is safe where it sits

The EMA checkpoint carries no `text_encoder.*` tensors. `_load_ema_weights` already accounts for
this: it counts them separately and logs them as expected-missing, warning only about
non-text-encoder gaps. So `load_state_dict` does not touch the encoder and cannot upcast it back,
and the bf16 line placed before that call survives it.

The encoder's hidden states are consumed in `CFMEdit.encode_text`, where they are layer-normed,
weighted by the fp32 `layer_weights`/`layer_scale` parameters, and summed before reaching the
fp32 transformer. That path was exercised end-to-end for every measurement below; no dtype
handling elsewhere needed changing.

## Quality control

`wer_all.json` reports WER as fractions. Prompts 1 and 2 have WER 0.0 in all runs; prompt 3 has 0.0889. Prompt 4 differs: baseline fp32 at seed 0 gives 0.1176, while bf16 gives 0.1765. However, fp32 at seeds 1 and 2 also gives 0.1765. Seed 0 is the outlier in these comparisons; the observed delta is consistent with take-to-take variance rather than evidence of bf16-specific degradation. These checks do not establish bit-identical or unchanged output.
