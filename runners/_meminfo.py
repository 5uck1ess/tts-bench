"""Tiny memory-sampling helper shared across runners.

Each runner imports this once and calls:
    _meminfo.reset_peak(device)            # before generation
    json_out.update(_meminfo.sample(device))  # after generation

Graceful degradation: if psutil or torch are missing, the corresponding
field comes back as None rather than raising. Runners that don't import
torch still get a valid sample() — peak_vram_mb just stays None.
"""

import os


# Memory sampling is intentionally best-effort: we don't want a missing
# psutil dep or an uninitialized CUDA context to break a perfectly good
# generation. Failures here become `None` in the JSON output, which the
# CSV / report treat as "no measurement" rather than as an error.
_SAFE = (ImportError, RuntimeError, AttributeError, OSError)


def reset_peak(device):
    """Reset the CUDA peak-memory counter before a generate() call.

    No-op for CPU/MPS devices, or when torch isn't importable. Safe to
    call before every run so the per-cell peak isn't polluted by prior runs.
    """
    if device != "cuda":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except _SAFE:
        pass


def sample(device):
    """Return {peak_mem_mb, peak_vram_mb} after a generate() call.

    peak_mem_mb is the current process RSS in MiB (None if psutil missing).
    peak_vram_mb is torch.cuda.max_memory_allocated() in MiB on CUDA,
    None for CPU/MPS or when torch is missing.
    """
    out = {"peak_mem_mb": None, "peak_vram_mb": None}
    try:
        import psutil
        out["peak_mem_mb"] = round(
            psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1
        )
    except _SAFE:
        pass
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                out["peak_vram_mb"] = round(
                    torch.cuda.max_memory_allocated() / (1024 * 1024), 1
                )
        except _SAFE:
            pass
    return out
