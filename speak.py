"""Interactive TTS speed feel-test.

Loads a model once, then opens a REPL. Type prompts, hear them spoken.
First prompt = cold (model warm-up). Subsequent = warm — that's what an
always-on interactive agent would feel like.

Usage:
    python speak.py pocket
    python speak.py neutts_air
    python speak.py neutts_nano --language fr
    python speak.py neutts_nano --reference reference/myvoice.wav

Type '/exit' or Ctrl+C to quit. WAVs go to results/speak/<timestamp>/.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parent


MODELS = {
    "pocket":      ("pocket",     "runners/pocket_runner.py",     None),
    "neutts_air":  ("neutts",     "runners/neutts_runner.py",     "air"),
    "neutts_nano": ("neutts",     "runners/neutts_runner.py",     "nano"),
    "luxtts":      ("luxtts",     "runners/luxtts_runner.py",     None),
    "kokoro":      ("kokoro",     "runners/kokoro_runner.py",     None),
    "kittentts":   ("kittentts",  "runners/kittentts_runner.py",  None),
    "piper":       ("piper",      "runners/piper_runner.py",      None),
    "chatterbox":  ("chatterbox", "runners/chatterbox_runner.py", None),
    "f5tts":       ("f5tts",      "runners/f5tts_runner.py",      None),
    "coqui":       ("coqui",      "runners/coqui_runner.py",      None),
    "vibevoice":   ("vibevoice",  "runners/vibevoice_runner.py",  None),
}


def venv_python(venv_dir: str) -> Path:
    root = REPO / "venvs" / venv_dir
    if sys.platform.startswith("win"):
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _play(wav_path: Path) -> None:
    if sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
            return
        except Exception as e:
            print(f"  (playback failed: {e})")
    elif sys.platform == "darwin":
        subprocess.run(["afplay", str(wav_path)], check=False)
    else:
        for tool in ("aplay", "paplay"):
            try:
                subprocess.run([tool, str(wav_path)], check=False)
                return
            except FileNotFoundError:
                continue


def main() -> int:
    p = argparse.ArgumentParser(description="Interactive TTS speed feel-test.")
    p.add_argument("model", choices=list(MODELS),
                   help=" | ".join(MODELS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", default=None,
                   help="Reference wav for cloning (omit to use model default voice).")
    p.add_argument("--language", default="en")
    args = p.parse_args()

    venv_dir, runner_rel, variant = MODELS[args.model]
    py = venv_python(venv_dir)
    runner = REPO / runner_rel
    if not py.exists():
        installer = "install.ps1" if sys.platform.startswith("win") else "install.sh"
        print(f"ERROR: venv not installed at {py}. Run {installer} first.")
        return 2

    out_dir = REPO / "results" / "speak" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(py), str(runner), "--stdin",
           "--device", args.device, "--language", args.language]
    if variant:
        cmd += ["--variant", variant]
    if args.reference:
        cmd += ["--reference", args.reference]

    print(f"Loading {args.model} (device={args.device}, lang={args.language})...", flush=True)
    t_load_start = time.perf_counter()
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    # Wait for {"ready": true}
    while True:
        line = proc.stdout.readline()
        if not line:
            stderr = proc.stderr.read() if proc.stderr else ""
            print(f"ERROR: runner died before becoming ready.\nstderr (last 500): {stderr[-500:]}")
            return 2
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("ready"):
            break
        if msg.get("ok") is False:
            print(f"ERROR during load: {msg.get('error')}")
            return 2

    t_load = time.perf_counter() - t_load_start
    print(f"Loaded in {t_load:.1f}s.")
    print(f"Type a prompt and press Enter. '/exit' or Ctrl+C to quit. WAVs: {out_dir}\n")

    turn = 0
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in ("/exit", "/quit"):
                break

            turn += 1
            wav = out_dir / f"turn_{turn:03d}.wav"
            job = {"text": text, "out": str(wav)}

            t0 = time.perf_counter()
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()

            response = None
            while True:
                line = proc.stdout.readline()
                if not line:
                    print("  ERROR: runner exited.")
                    return 2
                line = line.strip()
                if line.startswith("{"):
                    try:
                        response = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            wall = time.perf_counter() - t0

            if not response.get("ok"):
                print(f"  FAIL: {response.get('error')}")
                continue

            ttfa = response.get("ttfa_ms")
            gen_s = response.get("gen_s")
            audio_s = response.get("audio_s")
            rtf = audio_s / gen_s if (audio_s and gen_s) else None
            tag = "cold" if turn == 1 else "warm"
            parts = [f"[{tag}]"]
            if ttfa is not None:
                parts.append(f"ttfa={ttfa:.0f}ms")
            if gen_s is not None:
                parts.append(f"gen={gen_s:.2f}s")
            if audio_s is not None:
                parts.append(f"audio={audio_s:.2f}s")
            if rtf is not None:
                parts.append(f"rtf={rtf:.2f}x")
            parts.append(f"wall={wall:.2f}s")
            print("  " + " ".join(parts))

            _play(wav)

    except KeyboardInterrupt:
        print()
    finally:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{turn} turns. WAVs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
