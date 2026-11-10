"""ZONOS3-Mini runner (Zyphra; native C++/ggml runtime, no torch in the venv).

What is benched (name it on the board as such):
  * source  Zyphra/ZONOS3-Mini @ d5c266cfb96533445cc65041eda2f8a9b0d4d3b6 (MIT runtime),
            built from servers/ggml with patches/zonos3_mini_ggml.patch (see below)
  * weights Zyphra/ZONOS3-Mini @ 45deee5735477ed11a327e248294c466194c89e9 (Apache-2.0)
  * backbone gguf/zonos3-ptq.gguf -- the ggml runtime's DEFAULT backbone, a post-training
            quantized copy (Q4_0 experts, Q8_0 mixers, 0.66 GB) of the bf16 lm/zonos3_mini
            (800M total / ~100M active MoE). Zyphra's own evals run the bf16 LM on the
            Linux-only SGLang path, so this row is "ZONOS3-Mini (ggml Q4)", not the bf16 model.
  * decoder gguf/zac-keep2.gguf -- the ggml runtime's default, pruned multi-resolution iSTFT
            head (keep2_plain). The card's PyTorch default is the full head (full_mr,
            gguf/zac.gguf). Both decode the SAME codes (identical WER); keep2 is ~2x faster on
            CPU (prompt 3 warm, 9950X3D: 3.9 s vs 7.2 s) and ~20% faster on CUDA.
  * output  44.1 kHz mono, 15 Hz codec frames.

Why the server, not zonos3-cli: the CLI has no multi-utterance mode (every call reloads,
so no warm runs), and it skips the soft output limiter every server applies
(generation.json output_limit_dbfs=-1): CLI prompt-3 wavs peak at 1.25 and would clip.
Each runner call launches ONE `zonos3-server` on a free localhost port, waits for
/v2/capabilities (load + upstream's own backbone warmup -- excluded, like model load in the
other runners), then POSTs /v2/generate `--runs` times, non-streaming
(stream=false, format=wav). ttfa_ms == gen_s: this is NOT a streaming latency. The server
streams by default with a partial-codebook first chunk ("prefix" decode), whose audio is
not the whole-utterance decode, so we do not use it.

Cap: the CLI defaults to --max 450 frames (30 s); cloned/preset prompt 3 measured 324-340
frames. The server default is 2000 (133 s); we pass --max 2000 explicitly.

Voice / cloning: speaker_mode = "reference" in BOTH lenses -- zonos3-server's own
speaker_mode_default (capabilities). Not chosen by outcome: three-seed tests flip between the
two modes depending on voice and language, so the row uses the server default and publishes
what it does. (The CLI defaults to "embedding" and Zyphra's evals use it; that is a
different configuration from this row.)
  * default voice: upstream's shipped presets (gguf/voices, with their shipped .codes.npy, so
    no encoder runs), one per bench language like piper's per-language voices:
    en -> Wyatt_en-American_M_1, fr -> Gaspard_fr_M_1, es -> Rodrigo_es_M_1 (the male voice
    of each set, matching Wyatt; Rodrigo was pinned before listening). Without a voice
    upstream generation is UNCONDITIONED, so a voice is always pinned. Why per-language:
    in reference mode an ENGLISH voice prefix is fragile on French/Spanish text -- Wyatt on
    prompt 5 collapses to a 1.3-1.5 s clip on seeds 1/2/3 (shipped codes AND a live-encoded
    Wyatt clip alike), while Gaspard reads p5 at WER 0/0.07/0.07 and Rodrigo p6 at 0/0/0.
    That fragility is measured on this ggml Q4 build in reference mode only.
  * --reference: the clip is cached once per runner (POST /v2/speakers) and encoded by the
    server (codec encoder + SSL teacher -> raw reference prefix, up to 60 s) BEFORE timing,
    via a 12-frame throwaway generate (1 frame fails: the 10-codebook delay must drain);
    timed runs reuse the cached codes. No transcript is used. The house ref is English, so
    the cloning lens's p5/p6 are cross-lingual and hit the same fragility sometimes: on
    CUDA, seed 1, cloned p5 is a 1.33 s non-speech clip (Whisper: "Musique"), published as
    measured; CPU cloned p5/p6 read word-perfect.
  * LACI (upstream's skip/early-EOS rollback) is off: request default laci=false, and this
    server loads no LACI heads (capabilities laci_languages=[]).
  * Speaker CFG off (upstream default 1.0).

Seed pinned to 1 (the upstream README's example seed); reruns are bit-identical per device
(checked on CUDA and CPU). CPU and CUDA give different audio for the same seed.

Text normalization: ggml normalizes only with --worker-python (NeMo); without it text is
tokenized raw. The bench prompts are already spelled out (numbers as words), so this is not
needed; `language` is still sent (en_us / fr_fr / es, from the server language registry) as the API expects.

Threads (CPU): upstream hardcodes the backbone CPU backend at ggml's default of 4 threads.
patches/zonos3_mini_ggml.patch adds a ZONOS3_THREADS override, and we set it (and
--zac-threads) to the physical core count, as for the other CPU rows. Output is
bit-identical at 4/8/16/32 threads. The patch also renames a nested struct that MSVC
rejects (`zonos3_align_probe::head` has a member named `head`); no numeric change.

Memory: peak_mem_mb is the server child's peak working set; peak_vram_mb is the server
process's usage from nvidia-smi, sampled after each request. On Windows (WDDM reports no
per-process figure) it is the whole-GPU memory.used delta vs a baseline taken before the
server starts -- approximate if something else allocates VRAM during the cell. The server
pre-sizes its KV pool for --batch slots (upstream default 8), so this is not per-request.
"""

import argparse
import base64
import secrets
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import _meminfo  # noqa: F401  (shared contract; the child's memory is sampled here)

ROOT = Path(__file__).resolve().parent.parent
HOME = ROOT / "venvs" / "zonos3"
WEIGHTS = HOME / "weights" / "gguf"
EXE = ".exe" if os.name == "nt" else ""
BACKBONE = WEIGHTS / "zonos3-ptq.gguf"
DECODER = WEIGHTS / "zac-keep2.gguf"
VOICES = WEIGHTS / "voices"
PRESET = {"en": "Wyatt_en-American_M_1", "fr": "Gaspard_fr_M_1", "es": "Rodrigo_es_M_1"}
SEED = 1
MAX_FRAMES = 2000
# Per-run secrets: the upstream server allows any CORS origin and, without
# ZONOS3_ACCESS_TOKEN, no auth -- so a webpage could otherwise read the cached reference
# clip from the loopback server via a guessable session id. With the token set, the server
# rejects every request (CORS preflights included) lacking `Authorization: Bearer <token>`
# with 403 before routing.
SESSION = secrets.token_urlsafe(16)
TOKEN = secrets.token_urlsafe(32)
LANG = {"en": "en_us", "fr": "fr_fr", "es": "es"}
READY_TIMEOUT_S = 300
MAX_REF_BYTES = 64 * 1024 * 1024


class _KillOnCloseJob:
    """Windows: put the server in a Job object killed when this runner's handle closes
    (including TerminateProcess from the harness cell timeout), so an orphan never keeps the
    port and the VRAM. No-op on POSIX (the server is killed in a finally + PDEATHSIG)."""

    def __init__(self):
        self.h = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int,
                                                    ctypes.c_void_p, wintypes.DWORD)

            class _Basic(ctypes.Structure):
                _fields_ = [("a", ctypes.c_int64), ("b", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("c", ctypes.c_size_t),
                            ("d", ctypes.c_size_t), ("e", wintypes.DWORD),
                            ("f", ctypes.c_size_t), ("g", wintypes.DWORD), ("h", wintypes.DWORD)]

            class _Ext(ctypes.Structure):
                _fields_ = [("Basic", _Basic), ("Io", ctypes.c_uint64 * 6),
                            ("i", ctypes.c_size_t), ("j", ctypes.c_size_t),
                            ("k", ctypes.c_size_t), ("l", ctypes.c_size_t)]

            h = k32.CreateJobObjectW(None, None)
            info = _Ext()
            info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if h and k32.SetInformationJobObject(h, 9, ctypes.byref(info), ctypes.sizeof(info)):
                self.h, self.k32 = h, k32
        except (OSError, AttributeError):
            self.h = None

    def adopt(self, proc):
        if self.h:
            self.k32.AssignProcessToJobObject(self.h, int(proc._handle))


_JOB = _KillOnCloseJob()


def _die_with_parent():
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG


def _watchdog(pgid: int):
    """POSIX (Linux + macOS): a detached sh that SIGKILLs the server's process group once
    this runner is gone (same pattern as ribt_runner).

    harness.run_cell() enforces the cell timeout by killing the runner only, so close() in
    the finally never runs. macOS has no PDEATHSIG at all; the watchdog is a separate child,
    so it survives the runner's SIGKILL, sees the parent pid vanish, and kills the group.
    On a normal exit close() kills the watchdog first. All its stdio is /dev/null so it never
    holds the harness's capture pipes open."""
    script = f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 0.2; done; kill -9 -{pgid} 2>/dev/null"
    return subprocess.Popen(["sh", "-c", script], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def _binary(device: str) -> Path:
    return HOME / f"bin-{device}" / f"zonos3-server{EXE}"


def _physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", "0"], capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _opener():
    """Loopback-only opener: an empty ProxyHandler, so an inherited http_proxy/HTTP_PROXY can
    never route these 127.0.0.1 requests (which carry the reference clip) through a proxy."""
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


_OPENER = None


def _http(method, url, body=None, timeout=600):
    import urllib.request
    global _OPENER
    if _OPENER is None:
        _OPENER = _opener()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-TTS-Session-ID": SESSION,
                                          "Authorization": f"Bearer {TOKEN}"})
    with _OPENER.open(req, timeout=timeout) as r:
        return r.read()


class Server:
    def __init__(self, device: str):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        th = _physical_cores()
        cmd = [str(_binary(device)), str(BACKBONE), "--gpu" if device == "cuda" else "--cpu",
               "--zac", str(DECODER), "--max", str(MAX_FRAMES), "--port", str(self.port),
               "--host", "127.0.0.1", "--tts-default-voices-dir", str(VOICES),
               "--zac-threads", str(th)]
        env = dict(os.environ)
        env["ZONOS3_THREADS"] = str(th)
        env["SPK_THREADS"] = str(th)
        env["ZONOS3_ACCESS_TOKEN"] = TOKEN
        self.gpu_base = _gpu_used_mb() if device == "cuda" else None
        self.log = tempfile.TemporaryFile()
        posix = os.name != "nt"
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, cwd=str(HOME),
                                     # own process group, so the watchdog can kill it as a unit
                                     start_new_session=posix,
                                     preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
        _JOB.adopt(self.proc)  # Windows: Job object kills it when this runner's handle closes
        self.dog = _watchdog(self.proc.pid) if posix else None  # POSIX incl. macOS (no PDEATHSIG)
        self.peak_rss = 0.0
        self._stop = False
        threading.Thread(target=self._poll_mem, daemon=True).start()

    def _poll_mem(self):
        try:
            import psutil
            p = psutil.Process(self.proc.pid)
            while not self._stop and self.proc.poll() is None:
                try:
                    mi = p.memory_info()
                    self.peak_rss = max(self.peak_rss, getattr(mi, "peak_wset", 0) or mi.rss)
                except psutil.Error:
                    break
                time.sleep(0.05)
        except ImportError:
            pass

    def wait_ready(self):
        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT_S:
            if self.proc.poll() is not None:
                raise RuntimeError(f"zonos3-server exited {self.proc.returncode}: {self.tail()}")
            try:
                _http("GET", self.base + "/v2/capabilities", timeout=5)
                return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError(f"zonos3-server not ready after {READY_TIMEOUT_S}s: {self.tail()}")

    def speaker_id(self, stem: str) -> str:
        sp = json.loads(_http("GET", self.base + "/v2/speakers"))["speakers"]
        for s in sp:
            if Path(s.get("original_name") or "").stem == stem:
                return s["id"]
        raise RuntimeError(f"preset voice {stem} not served (have {len(sp)} speakers)")

    def cache_reference(self, path: str) -> str:
        """Cache the clip in this runner's session and encode it ONCE, before timing.

        POST /v2/speakers stores only the PCM; the server encodes reference codes lazily on
        the first generate that uses the voice and remembers them in the session. A 1-token
        generate forces that encode here, so the timed runs reuse the cached codes, like
        the other cloners' encode-once-at-load (without it CPU paid ~25 s of codec-encoder +
        SSL-teacher work inside every run)."""
        p = Path(path)
        # Bounded read: only a regular file, at most MAX_REF_BYTES (a 60 s 48 kHz stereo
        # float32 wav is ~23 MB; the server keeps only the first 60 s anyway). A FIFO or a
        # huge file would otherwise block or exhaust memory before the server's limit applies.
        if not p.is_file():
            raise ValueError(f"reference is not a regular file: {path}")
        size = p.stat().st_size
        if size > MAX_REF_BYTES:
            raise ValueError(f"reference too large ({size} B > {MAX_REF_BYTES} B): {path}")
        with p.open("rb") as fh:
            raw = fh.read(MAX_REF_BYTES + 1)
        if len(raw) > MAX_REF_BYTES:
            raise ValueError(f"reference grew past {MAX_REF_BYTES} B while reading: {path}")
        ref = base64.b64encode(raw).decode()
        sp = json.loads(_http("POST", self.base + "/v2/speakers",
                              {"speaker_audio_base64": ref, "speaker_audio_name": Path(path).name}))
        _http("POST", self.base + "/v2/generate",
              {"text": "Hello.", "stream": False, "format": "wav", "seed": SEED, "max_tokens": 12,
               "speaker_mode": "reference", "speaker_embedding_id": sp["id"]})
        return sp["id"]

    def vram_mb(self):
        """Server VRAM: its per-process entry from nvidia-smi where the driver reports one
        (Linux); on Windows/WDDM that field is N/A, so fall back to the whole-GPU
        memory.used minus the baseline taken before the server started."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                pid, mb = [x.strip() for x in line.split(",")[:2]]
                if pid.isdigit() and int(pid) == self.proc.pid and mb.replace(".", "").isdigit():
                    return float(mb)
            used = _gpu_used_mb()
            if used is not None and self.gpu_base is not None:
                return max(0.0, used - self.gpu_base)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return None

    def tail(self, n=1500) -> str:
        try:
            self.log.seek(0)
            return self.log.read().decode("utf-8", "replace")[-n:].replace("\n", " | ")
        except Exception:
            return ""

    def close(self):
        self._stop = True
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        if self.dog is not None:
            self.dog.kill()
            self.dog.wait()


def _generate(srv: Server, text: str, language: str, speaker: dict, out: Path | None):
    import io
    import soundfile as sf
    body = {"text": " ".join(text.split()), "language": LANG.get(language, "en_us"),
            "stream": False, "format": "wav", "seed": SEED, "max_tokens": MAX_FRAMES,
            "speaker_mode": "reference", **speaker}
    t0 = time.perf_counter()
    wav = _http("POST", srv.base + "/v2/generate", body)
    gen_s = time.perf_counter() - t0
    info = sf.info(io.BytesIO(wav))
    if out is not None:
        Path(out).write_bytes(wav)
    return gen_s, info.frames / info.samplerate


def _emit(idx, gen_s, audio_s, srv, device, vram):
    print(json.dumps({"ok": True, "run_index": idx, "ttfa_ms": gen_s * 1000, "gen_s": gen_s,
                      "audio_s": audio_s,
                      "peak_mem_mb": round(srv.peak_rss / (1024 * 1024), 1) if srv.peak_rss else None,
                      "peak_vram_mb": vram if device == "cuda" else None}), flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    def fail(msg):
        print(json.dumps({"ok": False, "run_index": 0, "error": msg}), flush=True)
        return 1

    if not args.stdin and (args.text is None or args.out is None):
        return fail("either --stdin or both --text and --out are required")
    if args.device not in ("cpu", "cuda"):
        return fail(f"zonos3_mini runner supports cpu/cuda, not {args.device}")
    if args.language not in LANG:
        return fail(f"unsupported language {args.language}")
    for f in (_binary(args.device), BACKBONE, DECODER):
        if not f.is_file():
            return fail(f"missing {f} (run install for zonos3_mini)")

    srv = None
    try:
        srv = Server(args.device)
        srv.wait_ready()
        if args.reference:
            speaker = {"speaker_embedding_id": srv.cache_reference(args.reference)}
        else:
            speaker = {"speaker_embedding_id": srv.speaker_id(PRESET[args.language])}

        if args.stdin:
            print(json.dumps({"ready": True}), flush=True)
            idx = 0
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                    lang = job.get("language", args.language)
                    spk = speaker if args.reference else {"speaker_embedding_id": srv.speaker_id(PRESET[lang])}
                    gen_s, audio_s = _generate(srv, job["text"], lang, spk, Path(job["out"]))
                    _emit(idx, gen_s, audio_s, srv, args.device, srv.vram_mb())
                except Exception as e:
                    print(json.dumps({"ok": False, "run_index": idx,
                                      "error": f"{type(e).__name__}: {e}"}), flush=True)
                idx += 1
            return 0

        for i in range(args.runs):
            try:
                gen_s, audio_s = _generate(srv, args.text, args.language, speaker,
                                           Path(args.out) if i == 0 else None)
                _emit(i, gen_s, audio_s, srv, args.device, srv.vram_mb())
            except Exception as e:
                print(json.dumps({"ok": False, "run_index": i,
                                  "error": f"{type(e).__name__}: {e} | {srv.tail(600)}"}), flush=True)
        return 0
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    finally:
        if srv is not None:
            srv.close()


if __name__ == "__main__":
    sys.exit(main())
