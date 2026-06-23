"""MioTTS runner (Aratako, LLM-codec TTS) — SERVER-BACKED, runs on Linux-3090.

MioTTS is an LLM-codec TTS: a small base LLM (Falcon-H1 / Qwen3 / LFM2 depending on
size) emits MioCodec audio tokens that a MioCodec torch model decodes to a 44.1 kHz wav
(default codec `MioCodec-25Hz-44.1kHz-v2`). Like higgs_v3, this runner loads NO model —
inference lives in TWO standing servers and this runner is a thin HTTP client.

  Server standup (Linux-3090, manual — see docs/known-issues.md):
    # 1) LLM server (reuse the box's existing llama.cpp). One GGUF at a time:
    llama-server -hf Aratako/MioTTS-GGUF -hff MioTTS-0.6B-BF16.gguf -c 8192 \
        --cont-batching --batch_size 8 --port 8000
    # 2) MioTTS REST orchestrator (clone Aratako/MioTTS-Inference; `uv sync`):
    python run_server.py --llm-base-url http://localhost:8000/v1 --port 8001
    #    (downloads MioCodec-25Hz-44.1kHz-v2 on first run; --device cuda)

The two MioTTS sizes (0.1b / 0.6b) share THIS runner + venv + servers. The variant is
carried for labeling only — the actual model is whichever GGUF the LLM server loaded, so
to bench the other size you (re)start the LLM server on its GGUF (a single GPU hosts one
at a time) and run one variant per server session.

Cloning is base64-in-band: MioTTS's `/v1/tts` takes `reference.type="base64"` +
`reference.data=<base64 wav>` — NO server-visible path, NO bind-mount (the win over
higgs_v3), and NO sibling transcript (unlike cosyvoice/f5tts; the codec handles the ref
internally). MioTTS is a pure zero-shot cloner with no model-native preset voice, so the
no-`--reference` (default-lens) path clones the house reference (chris_hemsworth_15s.wav),
the bench's default-voice convention → MioTTS is NO_PRESET_VOICE (cloning board only).

VRAM: the model lives in the servers, NOT this process, so `peak_vram_mb` is None; a coarse
whole-GPU figure comes from nvidia-smi as `gpu_used_mb` (includes both servers + anything
else on the card, not just this generation).

Licenses: inference code MIT; weights — 0.6B = Apache-2.0 (Qwen3-0.6B-Base), 0.1B =
Falcon-LLM License (permissive, benchmark redistribution OK). 44.1 kHz, EN/JA.
"""

import argparse
import base64
import io
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import _meminfo  # kept for parity; client-side VRAM is meaningless here (model is remote)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REF = REPO_ROOT / "reference" / "chris_hemsworth_15s.wav"

DEFAULT_SERVER_URL = "http://localhost:8001"
TTS_PATH = "/v1/tts"

_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _server_reachable(server_url: str, timeout: float = 3.0) -> bool:
    """Quick TCP connect to the server's host:port — never hang on a dead server."""
    parsed = urlparse(server_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _gpu_used_mb() -> float | None:
    """Coarse whole-GPU memory.used via nvidia-smi. Includes both servers, not just this
    generation — reported for rough sizing only, NOT as a per-call peak."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        first = out.stdout.strip().splitlines()[0]
        return float(first.strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None,
                   help="Wav path for zero-shot cloning (sent inline as base64; no transcript).")
    p.add_argument("--variant", default=None, help="0.1b / 0.6b — labeling only (see header).")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                   help="Base URL of the MioTTS REST orchestrator (default http://localhost:8001).")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    # Device gate: only cuda is meaningful (the servers run on the GPU). A non-cuda cell
    # skips cleanly, exactly like higgs_v3.
    if args.device != "cuda":
        _emit({"ok": False, "run_index": 0,
               "error": f"MioTTS is server-backed CUDA-only here; device={args.device}"})
        return 1

    # Pure cloner: no-reference (default lens) clones the house ref; --reference => cloning lens.
    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    if not ref_wav.exists():
        _emit({"ok": False, "run_index": 0, "error": f"reference wav not found: {ref_wav}"})
        return 1
    try:
        ref_b64 = base64.b64encode(ref_wav.read_bytes()).decode("ascii")
    except OSError as e:
        _emit({"ok": False, "run_index": 0, "error": f"cannot read reference {ref_wav}: {e}"})
        return 1

    try:
        import numpy as np
        import requests
        import soundfile as sf
    except Exception as e:
        _emit({"ok": False, "run_index": 0,
               "error": f"client deps missing (requests/soundfile/numpy): {type(e).__name__}: {e}"})
        return 1

    server_url = args.server_url.rstrip("/")
    if not _server_reachable(server_url):
        _emit({"ok": False, "run_index": 0,
               "error": f"MioTTS server not reachable at {server_url} — start the LLM + REST "
                        f"servers first (see runners/miotts_runner.py header)"})
        return 1

    tts_url = server_url + TTS_PATH

    def _payload(target_text: str) -> dict:
        # reference is always base64 (pure cloner): default lens clones the house ref,
        # cloning lens clones the supplied wav. output.format="wav" => raw wav bytes back.
        return {
            "text": target_text,
            "reference": {"type": "base64", "data": ref_b64},
            "output": {"format": "wav"},
        }

    def _decode_audio(resp) -> tuple:
        """Return (clip_float32_mono, sr). Handle both raw-wav and JSON-base64 responses
        (the Linux smoke test confirms which; we accept either defensively)."""
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            data = resp.json()
            b64 = data.get("audio") or data.get("data") or (data.get("output") or {}).get("data")
            raw = base64.b64decode(b64)
        else:
            raw = resp.content
        clip, sr = sf.read(io.BytesIO(raw), dtype="float32")
        return np.asarray(clip, dtype=np.float32).reshape(-1), int(sr)

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            t0 = time.perf_counter()
            resp = requests.post(tts_url, json=_payload(text), timeout=600)
            t_end = time.perf_counter()
            if resp.status_code != 200:
                _emit({"ok": False, "run_index": run_index,
                       "error": f"server {resp.status_code}: {resp.text[:200]}"})
                return False

            clip, sr = _decode_audio(resp)
            audio_s = float(len(clip) / sr) if sr else 0.0

            if write_wav:
                sf.write(str(out_path), clip, sr)

            # Non-streaming POST: the whole request IS time-to (and through) first audio,
            # so TTFA == gen_s == wall — same convention as other non-streaming models here.
            _emit({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                "peak_vram_mb": None,           # model is in the servers, not this process
                "gpu_used_mb": _gpu_used_mb(),  # whole-GPU incl. servers; coarse, not a peak
            })
            return True
        except requests.RequestException as e:
            _emit({"ok": False, "run_index": run_index,
                   "error": f"request failed: {type(e).__name__}: {e}"})
            return False
        except Exception as e:
            _emit({"ok": False, "run_index": run_index,
                   "error": f"{type(e).__name__}: {e}"})
            return False

    if args.stdin:
        idx = 0
        _emit({"ready": True})
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError as e:
                _emit({"ok": False, "run_index": idx, "error": f"json parse: {e}"})
                idx += 1
                continue
            _one(job["text"], job["out"], idx, write_wav=True)
            idx += 1
        return 0

    for i in range(args.runs):
        if not _one(args.text, args.out, i, write_wav=(i == 0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
