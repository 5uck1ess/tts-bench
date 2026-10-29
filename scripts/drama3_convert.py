"""Repack fishaudio/drama-3 into the single FP8 file the drama3 runner loads.

    venvs/drama3/<python> scripts/drama3_convert.py <hf_dir> <out.safetensors>

Runs the pinned port's own tools/convert.py with `--format fp8` (lossless: the checkpoint's
FP8 expert bytes and block scales are copied, nothing is re-rounded). convert.py imports
ComfyUI internals, so the venv's ComfyUI checkout goes on sys.path first; setting PYTHONPATH
from a Git-Bash shell instead segfaulted at startup on Windows. Holds the whole ~40 GB in RAM
before writing.
"""
import runpy
import sys
from pathlib import Path

COMFY = Path(__file__).resolve().parents[1] / "venvs" / "drama3" / "ComfyUI"
CONVERT = COMFY / "custom_nodes" / "ComfyUI-Drama3" / "tools" / "convert.py"

if len(sys.argv) != 3:
    sys.exit(__doc__)
sys.path.insert(0, str(COMFY))
sys.argv = [str(CONVERT), "model", sys.argv[1], sys.argv[2], "--format", "fp8"]
runpy.run_path(str(CONVERT), run_name="__main__")
