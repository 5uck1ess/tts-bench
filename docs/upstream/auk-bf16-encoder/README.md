# AuK: the fp32 encoder cast that costs 7.5 GiB

Prepared 2026-09-15, **not filed**. Target: https://github.com/Tencent-Hunyuan/AuK

`infer_auk.py` loads the frozen Qwen text encoder in bf16, then `model.to(torch.float32)`
re-casts it, doubling a 4.03B-param module from 7.52 to 15.03 GiB. That is the whole reason
AuK does not fit on a 24 GB card. One line after the cast, while the model is still on CPU,
drops peak VRAM 26.05 -> 18.58 GiB with no measurable quality change.

- `issue.md` — bug report body (title is the H1)
- `pr.md` — pull request body (title is the H1)
- `baseline.json` / `bf16.json` — the Win-5090 VRAM + RTFx measurements behind every number
- `wer_all.json` — WER for fp32 seeds 0/1/2 and bf16 seed 0, the control that shows prompt 4's
  delta is take-to-take variance and not a dtype regression

Every figure in the two bodies is traceable to those three files; a checker that flags any
unsourced decimal is in this repo's history with the drafting commit.

Nothing upstream has been opened. Checked 2026-09-15: issues #1-17 and PRs #1-15 contain no
report of the cast. Merged PR #9 (`cpu_offload to save memory`) works around the symptom.
