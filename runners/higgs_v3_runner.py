"""Higgs Audio v3 TTS runner (Boson AI, Research/Non-Commercial, ~4B) — SERVER-BACKED.

Higgs Audio v3 (`bosonai/higgs-audio-v3-tts-4b`) is a `higgs_multimodal_qwen3` model:
a Qwen3 ~4B backbone (36L, hidden 2560, GQA 32/8) + a Higgs audio tokenizer (8 codebooks
x 1026 vocab, 25 fps, delay pattern) emitting **24 kHz** audio. 100 languages, zero-shot
in-context cloning (ref audio + transcript), inline control tokens.

UNLIKE every other runner in this bench, this one loads NO model. The v3 repo ships only
`config.json` + `tokenizer.*` + `model.safetensors` — no `modeling_*.py`, no `auto_map`,
and the class isn't in stock transformers; raw `.generate()` would emit audio *tokens*
that still need the Higgs tokenizer decoder (wired up only by the SGLang-Omni stack). So
the one supported inference path is a Docker container running `sgl-omni serve`, exposing
an OpenAI-style HTTP `/v1/audio/speech`. This runner is a thin HTTP client against it.

  Server standup (manual, Linux-only, see install.sh header + the model-card cookbook
  https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html):
    docker run -it --gpus all --shm-size 32g --ipc host --network host --privileged \
      -v "$(pwd)":"$(pwd)" -w "$(pwd)" -e HF_TOKEN=hf_xxx lmsysorg/sglang-omni:dev /bin/zsh
    # inside: clone sglang-omni, `uv pip install -e .`, accept HF terms, then
    sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000

Cloning is in-context: the reference wav AND its transcript (a same-basename `.txt`, like
f5tts / neutts / higgs v2) are handed to the server via the CreateSpeechRequest top-level
`ref_audio` (a server-visible file path) + `ref_text` (the transcript) fields. The
`-v "$(pwd)":"$(pwd)"` bind-mount makes the repo's real path valid inside the container so
the absolute path resolves. (The HTTP schema exposes no base64 reference field over
`/v1/audio/speech` — `SpeechReference` is audio_path/text/vq_codes only — so a
server-visible path is the only file route; the bind-mount is required, not optional.)

VRAM: the model lives in the container, NOT in this process, so `peak_vram_mb` is None. A
coarse whole-GPU figure is reported via nvidia-smi as `gpu_used_mb` — but it includes the
whole server (and anything else on the card), not just this generation.

License: "Boson Higgs Audio v3 Research and Non-Commercial License" — explicitly permits
non-production benchmarking/evaluation by commercial entities and publication of benchmark
outputs / illustrative samples in non-commercial repositories. Model download is HF-gated
(accept terms on the model page + export HF_TOKEN before `sgl-omni serve`).
"""

import argparse
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

DEFAULT_SERVER_URL = "http://localhost:8000"
SPEECH_PATH = "/v1/audio/speech"

_REAL_STDOUT = sys.stdout


def _emit(obj) -> None:
    print(json.dumps(obj), file=_REAL_STDOUT, flush=True)


def _read_transcript(ref_wav: Path) -> str | None:
    txt = ref_wav.with_suffix(".txt")
    if txt.exists():
        return txt.read_text(encoding="utf-8").strip()
    return None


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
    """Coarse whole-GPU memory.used via nvidia-smi. Includes the whole server, not just
    this generation — reported for rough sizing only, NOT as a per-call peak."""
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
                   help="Wav path for in-context cloning; needs a same-basename .txt transcript.")
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                   help="Base URL of the sgl-omni server (default http://localhost:8000).")
    args = p.parse_args()

    if not args.stdin and (args.text is None or args.out is None):
        _emit({"ok": False, "run_index": 0,
               "error": "either --stdin or both --text and --out are required"})
        return 1

    # Device gate: only cuda is meaningful (the server runs on the GPU). A non-cuda cell
    # is fine to skip cleanly, exactly like the v2 runner does.
    if args.device != "cuda":
        _emit({"ok": False, "run_index": 0,
               "error": f"Higgs Audio v3 is server-backed CUDA-only here; device={args.device}"})
        return 1

    # Cloning path needs the reference wav + its transcript; default-voice path uses the
    # model's own native voice (no references array). Presence of --reference => cloning.
    cloning = args.reference is not None
    ref_wav = Path(args.reference) if args.reference else DEFAULT_REF
    ref_transcript = None
    if cloning:
        if not ref_wav.exists():
            _emit({"ok": False, "run_index": 0, "error": f"reference wav not found: {ref_wav}"})
            return 1
        ref_transcript = _read_transcript(ref_wav)
        if ref_transcript is None:
            _emit({"ok": False, "run_index": 0,
                   "error": f"in-context cloning needs a transcript: "
                            f"{ref_wav.with_suffix('.txt')} not found"})
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
               "error": f"sgl-omni server not reachable at {server_url} — start it first "
                        f"(see runners/higgs_v3_runner.py header)"})
        return 1

    speech_url = server_url + SPEECH_PATH

    def _payload(target_text: str) -> dict:
        # No-reference path relies on the server's voice="default" (its native voice).
        body = {
            "input": target_text,
            "temperature": 0.8,
            "top_k": 50,
            "max_new_tokens": 1024,
        }
        if cloning:
            # sgl-omni CreateSpeechRequest top-level fields: ref_audio (a server-visible
            # file path or URL) + ref_text (the reference transcript, for in-context
            # cloning). The bind-mount (-v "$(pwd)":"$(pwd)") makes this absolute path
            # resolve inside the container. The HTTP schema's SpeechReference exposes only
            # audio_path/text/vq_codes — there is NO base64 field over /v1/audio/speech —
            # so a server-visible path is the only file route (hence the bind-mount).
            body["ref_audio"] = str(ref_wav.resolve())
            body["ref_text"] = ref_transcript
        return body

    def _one(text: str, out_path: str, run_index: int, write_wav: bool) -> bool:
        try:
            t0 = time.perf_counter()
            resp = requests.post(speech_url, json=_payload(text), timeout=600)
            t_end = time.perf_counter()
            if resp.status_code != 200:
                _emit({"ok": False, "run_index": run_index,
                       "error": f"server {resp.status_code}: {resp.text[:200]}"})
                return False

            clip, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
            clip = np.asarray(clip, dtype=np.float32).reshape(-1)
            sr = int(sr)
            audio_s = float(len(clip) / sr) if sr else 0.0

            if write_wav:
                sf.write(str(out_path), clip, sr)

            # Non-streaming POST: the whole request IS the time to (and through) first
            # audio, so TTFA == gen_s == wall — same convention as other non-streaming
            # models in this bench. (Streaming SSE for true TTFA is a later enhancement.)
            _emit({
                "ok": True, "run_index": run_index,
                "ttfa_ms": (t_end - t0) * 1000,
                "gen_s": t_end - t0, "audio_s": audio_s,
                "peak_vram_mb": None,          # model is in the container, not this process
                "gpu_used_mb": _gpu_used_mb(),  # whole-GPU incl. server; coarse, not a peak
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
