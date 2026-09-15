# Restore the Qwen text encoder to bf16 before GPU transfer

## Change

In [`src/auk/infer/infer_auk.py`](https://github.com/Tencent-Hunyuan/AuK/blob/main/src/auk/infer/infer_auk.py), the model-wide fp32 cast re-casts the frozen Qwen encoder that was loaded with `torch_dtype=torch.bfloat16`.

Add one line immediately after that cast, while the model is still on CPU and before checkpoint loading and device transfer:

```diff
 model = model.to(torch.float32)
+model.text_encoder.to(torch.bfloat16)
```

The fp32 encoder therefore never reaches the GPU. The transformer and VAE remain fp32. This addresses the encoder upcast responsible for the reported GPU capacity overrun.

## Why the cast is safe where it sits

The EMA checkpoint carries no `text_encoder.*` tensors. `_load_ema_weights` already accounts for
this: it counts them separately and logs them as expected-missing, warning only about
non-text-encoder gaps. So `load_state_dict` does not touch the encoder and cannot upcast it back,
and the bf16 line placed before that call survives it.

The encoder's hidden states are consumed in `CFMEdit.encode_text`, where they are layer-normed,
weighted by the fp32 `layer_weights`/`layer_scale` parameters, and summed before reaching the
fp32 transformer. That path was exercised end-to-end for every measurement below; no dtype
handling elsewhere needed changing.

## Validation

Measurements in `baseline.json` and `bf16.json` were taken on an RTX 5090 with 32 GB. The fix was then verified on the RTX 3090 with 24 GB where the OOM was originally observed: that card failed at 23.48 GiB of 23.56 usable on canonical prompt 1 before the change, and after it completes prompt 1 at a **17.36 GiB** peak and a 23-second prompt at **17.91 GiB**, with `text_encoder` confirmed `torch.bfloat16`. WER on that card is 0.0 for prompt 1 and 0.1212/0.0606 across two seeds for prompt 3, straddling the 0.0889 measured on the 32 GB card. Figures in `linux3090.json`.

| Measurement | Baseline | Fix |
| --- | --- | --- |
| Encoder dtype | torch.float32 | torch.bfloat16 |
| Encoder parameter storage, GiB | 15.031 | 7.515 |
| Peak after load, GiB | 21.443 | 13.966 |
| Overall peak, GiB | 26.054 | 18.582 |

Quality control in `wer_all.json`: prompts 1 and 2 have WER 0.0 throughout, and prompt 3 has 0.0889. Prompt 4 changes from 0.1176 for baseline fp32 at seed 0 to 0.1765 for bf16. The fp32 control runs at seeds 1 and 2 also produce WER 0.1765. Seed 0 is the outlier; the delta is consistent with take-to-take variance. This does not demonstrate bit-identical or unchanged output, nor does this limited WER check establish general quality equivalence.
