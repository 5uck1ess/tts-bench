"""TontaubeV1.1-int4 runner (Cremer & Cremer / craitech, Tontaube Community Model License 1.0).

Multilingual expressive TTS: four causal codebook predictors (CB0 = semantic + duration
from character-level text, CB1-CB3 = acoustic detail) over DualCodec tokens, decoded
via a VibeVoice acoustic latent with Tontaube's own adapter. This row is the 4-bit
GPTQ release (`TontaubeAI/TontaubeV1.1-int4`), the inference server's default model,
on its "lite" backend. The full-precision TontaubeV1.1 (vLLM backend) is not benched.

LINUX-ONLY. Upstream: "Native Windows, macOS, and non-NVIDIA accelerators are not
supported" (lite backend builds native C++ kernels; CPU path needs AVX2/FMA/F16C).

API: there is no library entry point, only `tontaube serve` (FastAPI). We call the
same coroutine the /predict handler calls, in-process:
    from app.main import synthesize, TTSRequest, _get_engine
    await _get_engine()                       # what the server's lifespan does at start
    res = await synthesize(TTSRequest(...))   # == POST /predict body
`tontaube serve` adds only a torch-variant check, the FastAPI app and a request
semaphore on top of that (read app/cli.py + main.create_app, v1.1.0), so the audio
and timing are the served path minus HTTP.

Pinned: inference repo tag v1.1.0 (commit ef254ca), weights resolved to HF sha
2cff8331a357cdd50b05c823f5172eaa3ac6c121. The runner forces that repo + sha (overriding any
inherited TTS_MODEL_REPO_ID / MODEL_PATH / CB<n>_MODEL_PATH / LITE_QUANT) and refuses to
load unless the int4 bundle is beside the codebooks: upstream otherwise falls back to bf16
silently. A --device cpu run also hides CUDA (DualCodec / VibeVoice pick CUDA on their own).
References are validated before load (non-empty, <= upstream's 8 MiB-base64 cap) and
cloning requires ffmpeg/ffprobe, which upstream decodes every reference through.
DualCodec / W2V-BERT / VibeVoice / MossFormer2 download from upstream at revisions
the server pins.

Choices / gotchas (measured on linux-3090, 2026-11-19):
- Device is chosen by env at IMPORT time (app.config reads LITE_DEVICE and derives the
  CPU defaults), so LITE_DEVICE is set from --device before any `app.*` import.
- MossFormer2_SR_48K super-resolution is upstream's per-device default: ON with a GPU
  (48 kHz out), OFF on the lite CPU path (24 kHz out; upstream: it "costs about 3x real
  time" there). We pass the same value explicitly per device, so the CUDA wav (the one
  Listen / arena / scoring use) is 48 kHz and the CPU row times what CPU users get.
- Preset voices are bundled reference clips the server ships (voices/samples, 46 of
  them, per language); with no reference the server uses Marcus.mp3. Same footing as
  ribt's bundled ref -> _PRESET_AND_CLONE. Pinned per bench language to a native
  speaker so fr/es defaults aren't cross-lingual clones of an English voice:
  en Marcus (upstream default) / fr Adam / es Fernando (both tagged "audiobook").
- Style tag pinned to "audiobook" (upstream DEFAULT_TAG). Sliders left at model choice.
- Seed pinned (SEED) on every call; reruns regenerate the same take. CUDA: identical
  durations on every run. CPU: warm runs identical to each other, but run 0 (the one
  whose wav is written) can differ slightly from them -- clone p3 21.93 s cold vs 22.95 s
  warm -- so first-call numerics on the CPU kernels differ; no degradation across runs.
  Upstream sampling defaults (temp 0.7, freq penalty 0.1).
- The pipeline includes upstream's CB0-CTC guard: an English piece whose CTC CER > 0.25
  is regenerated (at most 2 guard retries) and a quality gate may resample. Those
  retries are inside gen_s — that's what the server does per request.
- Cloning: the reference wav is sent as voice_audio_b64 each call (as a /predict client
  would). The engine caches encoded voice prompts by content, so run 0 of a cell
  includes the W2V-BERT reference encode and warm runs don't. No transcript needed.
- Generation cap: 400 CB0 tokens per text chunk (12.5 Hz -> 32 s), chunks <= 350 chars.
  Prompt 3 (253 chars) is one chunk and measured 261 frames default / 220 cloned.
- Boundary silence trimmed to 250 ms padding (upstream default); kept.
- Non-streaming /predict path: ttfa_ms == gen_s (whole clip returned at once).
- CPU: first load JIT-compiles the lite C++ kernels (needs g++ + ninja from the venv's
  bin/, which the runner puts on PATH) and runs a one-time GEMV tuning pass; both are
  cached under ~/.cache/tontaube.
- Library noise (APEX notice, SyntaxWarnings, HF progress bars) is routed to stderr;
  stdout carries only the JSON lines.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import _meminfo


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "venvs" / "tontaube" / "src"
SEED = 1234
TAG = "audiobook"
LANGS = {"en": "english", "fr": "french", "es": "spanish"}
PRESETS = {"en": "Marcus.mp3", "fr": "Adam.mp3", "es": "Fernando.mp3"}
MODEL_REPO = "TontaubeAI/TontaubeV1.1-int4"
MODEL_REVISION = "2cff8331a357cdd50b05c823f5172eaa3ac6c121"   # = tag v1.1.0
# upstream MAX_VOICE_REFERENCE_B64_CHARS (8 MiB of base64) as raw bytes
MAX_REF_BYTES = 8 * 1024 * 1024 // 4 * 3


def _build_dir() -> Path:
    # Same expression upstream uses (app/native/__init__.py), resolved to an absolute
    # real path so it compares against /proc/<pid>/cwd even when the env var is relative.
    root = Path(os.getenv("TONTAUBE_CACHE_DIR",
                          Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "tontaube")).expanduser()
    return (root / "native-kernels").resolve()


def _builder_alive(build: Path) -> bool:
    """True while any process is compiling in `build` (ninja runs with it as cwd;
    the compiler/linker get its paths on the command line)."""
    needle = str(build)
    me = str(os.getpid())
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit() or pid_dir.name == me:
            continue
        try:
            if needle in (pid_dir / "cmdline").read_bytes().decode("utf-8", "replace"):
                return True
            cwd = os.readlink(pid_dir / "cwd")
            if cwd == needle or cwd.startswith(needle + os.sep):
                return True
        except OSError:
            continue
    return False


def _build_native_kernels_locked(note) -> None:
    """Build upstream's CPU C++ kernels under a runner-owned flock, recovering a
    PyTorch build lock orphaned by a killed run.

    torch.utils.cpp_extension guards the build dir with a bare `lock` file
    (FileBaton: no owner PID, removed in a `finally`). harness.run_cell kills the
    runner on cell timeout, which skips that `finally` and can leave ninja running;
    every later load would then wait on the orphaned file until it times out too.

    Ownership comes from an fcntl.flock on a sidecar file instead: the kernel drops
    it when its holder dies, so holding it proves no live *runner* owns `lock`. We
    then wait for any orphaned compiler from a killed run to exit (so two builds never
    share the dir), and only then remove a leftover `lock`. The build itself
    (app.native.kernels(), normally triggered lazily inside engine load) is forced
    while we hold the flock, so a second runner blocks on the flock rather than
    racing the lock file. Covers runner-vs-runner; a `tontaube serve` building in the
    same cache dir at that moment is not coordinated (not a bench scenario).
    """
    import fcntl

    build = _build_dir()
    build.mkdir(parents=True, exist_ok=True)
    with open(build / ".tontaube_runner.flock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)          # waits for a live runner's build
        lock = build / "lock"
        if lock.exists():                       # only a leftover lock needs the /proc scan
            waited = False
            while _builder_alive(build):        # orphaned ninja/c++ from a killed run
                waited = True
                time.sleep(0.5)
            if waited:
                note("tontaube: waited for an orphaned native-kernels build to exit")
            lock.unlink(missing_ok=True)
            note(f"tontaube: removed stale build lock {lock} (no live owner)")
        from app.native import kernels
        kernels()                               # build (or load the cached .so) now
        # flock released on close; the extension is cached in-process from here.


def _private_tmpdir(note) -> Path:
    """Per-process temp dir under the Tontaube cache, sweeping dirs of dead runners.

    Upstream decodes each new reference to a NamedTemporaryFile(delete=False) wav and
    removes it in a `finally`; harness.run_cell's timeout kill skips that, leaving a
    decoded copy of the reference behind. Pointing tempfile at runner-tmp/<pid> lets
    the next run delete whatever a killed run left (normal runs clean up on exit).
    """
    base = _build_dir().parent / "runner-tmp"
    base.mkdir(parents=True, exist_ok=True)
    for d in base.iterdir():
        if not d.name.isdigit() or d.name == str(os.getpid()):
            continue
        try:
            os.kill(int(d.name), 0)
            continue                            # that runner is still alive
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        shutil.rmtree(d, ignore_errors=True)
        note(f"tontaube: removed temp files left by killed runner {d.name}")
    mine = base / str(os.getpid())
    mine.mkdir(exist_ok=True)
    return mine


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--stdin", action="store_true")
    args = p.parse_args()

    # Everything the libraries print goes to stderr; JSON goes to the saved stdout.
    out = sys.stdout
    sys.stdout = sys.stderr

    def emit(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    if not args.stdin and (args.text is None or args.out is None):
        emit({"ok": False, "run_index": 0,
              "error": "either --stdin or both --text and --out are required"})
        return 1
    if args.language not in LANGS:
        emit({"ok": False, "run_index": 0,
              "error": f"tontaube runner supports {sorted(LANGS)}, got {args.language}"})
        return 1
    if args.device not in ("cpu", "cuda"):
        emit({"ok": False, "run_index": 0, "error": f"unsupported device {args.device}"})
        return 1

    os.environ["TTS_BACKEND"] = "lite"
    os.environ["LITE_DEVICE"] = args.device
    if args.device == "cpu":
        # LITE_DEVICE only moves the predictors; the DualCodec / VibeVoice / W2V-BERT
        # loaders pick CUDA on their own whenever it is visible. Hide it so a CPU run is
        # CPU end to end (harness.run_cell does the same; this covers direct runs).
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VOICE_PATH"] = str(SRC / "voices" / "samples")
    # Pin the checkpoint this row is labelled with. Upstream resolves the model from
    # inheritable env (TTS_MODEL_REPO_ID / _REVISION, MODEL_PATH, CB<n>_MODEL_PATH,
    # LITE_QUANT) and silently falls back to bf16 when no int4 bundle sits next to the
    # codebooks, so a stray override would bench a different model under the int4 name.
    os.environ["TTS_MODEL_REPO_ID"] = MODEL_REPO
    os.environ["TTS_MODEL_REVISION"] = MODEL_REVISION
    os.environ["LITE_QUANT"] = "auto"
    for var in ("MODEL_PATH", "CB0_MODEL_PATH", "CB1_MODEL_PATH", "CB2_MODEL_PATH", "CB3_MODEL_PATH"):
        os.environ.pop(var, None)
    # The CPU path JIT-builds C++ kernels via torch.utils.cpp_extension, which shells
    # out to `ninja`. The ninja wheel's binary lives in the venv's bin/, which is only
    # on PATH when the venv is activated (upstream runs under `uv run`); harness calls
    # the venv python directly, so put its bin/ on PATH ourselves.
    bindir = str(Path(sys.executable).parent)
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message=".*weight_norm", category=FutureWarning)
    sys.path.insert(0, str(SRC))

    voice_b64 = None
    if args.reference:
        # Validate before model load. An empty file would encode to "" and upstream
        # would quietly use its default voice (an ok:true "clone" of Marcus); an
        # oversized one would be read whole before upstream's base64 cap rejects it.
        ref = Path(args.reference)
        try:
            size = ref.stat().st_size
            if size == 0:
                raise ValueError(f"reference {ref} is empty")
            if size > MAX_REF_BYTES:
                raise ValueError(f"reference {ref} is {size} bytes; upstream accepts at most "
                                 f"{MAX_REF_BYTES} (8 MiB of base64)")
            with ref.open("rb") as f:
                data = f.read(MAX_REF_BYTES + 1)
            if not data or len(data) > MAX_REF_BYTES:
                raise ValueError(f"reference {ref} changed size while reading")
            voice_b64 = base64.b64encode(data).decode("ascii")
        except (OSError, ValueError) as e:
            emit({"ok": False, "run_index": 0, "error": f"reference rejected: {e}"})
            return 1
        # Upstream decodes reference audio through FFmpeg (ffprobe + ffmpeg); without
        # it even a valid PCM wav is rejected as invalid audio.
        if not (shutil.which("ffprobe") and shutil.which("ffmpeg")):
            emit({"ok": False, "run_index": 0,
                  "error": "cloning needs ffmpeg/ffprobe on PATH (upstream decodes references with FFmpeg)"})
            return 1
    mossformer = args.device == "cuda"   # upstream's per-device default, made explicit

    import atexit
    import tempfile
    tmp = _private_tmpdir(lambda m: print(m, file=sys.stderr))
    os.environ["TMPDIR"] = str(tmp)
    tempfile.tempdir = str(tmp)
    atexit.register(shutil.rmtree, tmp, True)

    async def run() -> int:
        try:
            import logging
            logging.getLogger("VibeVoice.vibevoice.modular.modular_vibevoice_tokenizer").setLevel(logging.ERROR)
            import soundfile as sf
            from app.main import synthesize, TTSRequest, _get_engine
            from app.model_sources import resolve_codebook_model_paths
            paths = resolve_codebook_model_paths()
            if MODEL_REVISION not in paths[0] or not (Path(paths[0]).parent / "int4" / "bundle.json").is_file():
                raise RuntimeError(f"codebooks resolved to {paths[0]}, not the pinned int4 snapshot "
                                   f"{MODEL_REPO}@{MODEL_REVISION}")
            if args.device == "cpu":
                _build_native_kernels_locked(lambda m: print(m, file=sys.stderr))
            await _get_engine()
        except Exception as e:
            emit({"ok": False, "run_index": 0, "error": f"load failed: {type(e).__name__}: {e}"})
            return 1

        async def one(text, out_path, run_index, write_wav, language):
            try:
                if language not in LANGS:
                    raise ValueError(f"unsupported language {language}")
                kw = dict(text=text, language=LANGS[language], tag=TAG, seed=SEED,
                          format="wav", mossformer2_postprocess=mossformer)
                if voice_b64 is not None:
                    kw["voice_audio_b64"] = voice_b64
                else:
                    kw["voice_path"] = PRESETS[language]
                req = TTSRequest(**kw)
                _meminfo.reset_peak(args.device)
                t0 = time.perf_counter()
                res = await synthesize(req)
                t_end = time.perf_counter()
                audio, sr = sf.read(io.BytesIO(base64.b64decode(res["audio_b64"])))
                if write_wav:
                    sf.write(out_path, audio, sr)
                emit({
                    "ok": True, "run_index": run_index,
                    "ttfa_ms": (t_end - t0) * 1000,
                    "gen_s": t_end - t0, "audio_s": float(len(audio) / sr),
                    **_meminfo.sample(args.device),
                })
                return True
            except Exception as e:
                emit({"ok": False, "run_index": run_index, "error": f"{type(e).__name__}: {e}"})
                return False

        if args.stdin:
            emit({"ready": True})
            loop = asyncio.get_running_loop()
            idx = 0
            while True:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                except json.JSONDecodeError as e:
                    emit({"ok": False, "run_index": idx, "error": f"json parse: {e}"})
                    idx += 1
                    continue
                await one(job["text"], job["out"], idx, True, job.get("language", args.language))
                idx += 1
            return 0

        for i in range(args.runs):
            if not await one(args.text, args.out, i, i == 0, args.language):
                return 1
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
