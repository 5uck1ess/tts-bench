"""Collect hosted-API samples on the canonical bench prompts (raw material for a
separate hosted pool, issue #14). NOT a bench row: nothing here touches results/,
merge.py, the speed tables or the arena.

    venvs/hosted/Scripts/python.exe scripts/collect_hosted.py --probe
    venvs/hosted/Scripts/python.exe scripts/collect_hosted.py --model drama-3

Fish Audio only for now. Key comes from FISH_API_KEY; only a fingerprint is stored.

Traps:
- An omitted or UNRECOGNIZED `model` header silently falls back to s2.1-pro with a
  200. The API reference enum says `drama-3`; launch coverage said `drama-3-preview`.
  Run --probe first and check that drama-3 audio actually differs from s2.1-pro.
- Timings are network + hosted GPU. Never compare them with the local speed tables.
- Clearing the model license (docs/considered.md) does not clear the API ToS.
  Check the ToS before putting any of this output in a public gallery.

Layout: hosted/<model>/<lens>/p<N>[_t<take>].wav + a .json sidecar per clip.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ormsgpack
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring.prompts import PROMPTS  # noqa: E402

URL = "https://api.fish.audio/v1/tts"
REF = ROOT / "reference" / "chris_hemsworth_15s"
OUT = ROOT / "hosted"


def synth(key, model, text, ref=None, reference_id=None):
    body = {"text": text, "format": "wav", "sample_rate": 44100, "latency": "normal"}
    if reference_id:
        body["reference_id"] = reference_id
    logged = dict(body)
    if ref:
        audio = ref.with_suffix(".wav").read_bytes()
        transcript = ref.with_suffix(".txt").read_text(encoding="utf-8").strip()
        body["references"] = [{"audio": audio, "text": transcript}]
        logged["references"] = [{"audio": f"{ref.name}.wav sha256="
                                 + hashlib.sha256(audio).hexdigest()[:16],
                                 "text": transcript}]
    headers = {"Authorization": f"Bearer {key}", "model": model,
               "Content-Type": "application/msgpack"}
    t0 = time.perf_counter()
    r = requests.post(URL, data=ormsgpack.packb(body), headers=headers,
                      stream=True, timeout=300)
    ttfb = None
    chunks = []
    for chunk in r.iter_content(8192):
        if ttfb is None:
            ttfb = time.perf_counter() - t0
        chunks.append(chunk)
    total = time.perf_counter() - t0
    data = b"".join(chunks)
    if r.status_code != 200:
        raise RuntimeError(f"{model}: HTTP {r.status_code}: {data[:500]!r}")
    return data, logged, dict(r.headers), ttfb, total


def save(path, data, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    info = sf.info(str(path))
    meta.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                sample_rate=info.samplerate, channels=info.channels,
                duration_s=round(info.duration, 3))
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def cell(key, model, lens, pid, text, take, reference_id):
    ref = REF if lens == "cloning" else None
    data, body, resp_headers, ttfb, total = synth(key, model, text, ref, reference_id)
    name = f"p{pid}" + (f"_t{take}" if take > 1 else "") + ".wav"
    meta = {
        "provider": "fish.audio", "endpoint": URL, "model_header_sent": model,
        "lens": lens, "prompt_id": pid, "take": take,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "key_fingerprint": hashlib.sha256(key.encode()).hexdigest()[:8],
        "request_body": body, "response_headers": resp_headers,
        "network_plus_hosted_ttfb_s": round(ttfb or 0, 3),
        "network_plus_hosted_total_s": round(total, 3),
    }
    meta = save(OUT / model / lens / name, data, meta)
    print(f"{model:>10} {lens:<8} p{pid} t{take}  {meta['duration_s']:6.2f}s "
          f"{meta['sample_rate']}Hz  total {total:5.1f}s  {meta['sha256'][:10]}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="drama-3")
    ap.add_argument("--probe", action="store_true",
                    help="p1 on drama-3 x2 and s2.1-pro, default lens; check they differ")
    ap.add_argument("--lens", choices=["default", "cloning", "both"], default="both")
    ap.add_argument("--prompts", default="1,2,3,4,5,6")
    ap.add_argument("--takes", type=int, default=1)
    ap.add_argument("--reference-id", help="pin a Fish library voice for the default lens")
    args = ap.parse_args()

    key = os.environ.get("FISH_API_KEY")
    if not key:
        sys.exit("FISH_API_KEY not set")

    by_id = {pid: text for pid, _lang, text in PROMPTS}
    if args.probe:
        text = by_id[1]
        a = cell(key, "drama-3", "default", 1, text, 1, args.reference_id)
        b = cell(key, "drama-3", "default", 1, text, 2, args.reference_id)
        c = cell(key, "s2.1-pro", "default", 1, text, 1, args.reference_id)
        # The reference enum says drama-3; launch coverage said drama-3-preview.
        # One of them may be the silent s2.1-pro fallback, so collect both.
        d = cell(key, "drama-3-preview", "default", 1, text, 1, args.reference_id)
        print("\nresponse headers (drama-3):", json.dumps(a["response_headers"], indent=1))
        print("drama-3 take1 vs take2 identical:", a["sha256"] == b["sha256"])
        print("drama-3 vs s2.1-pro identical:  ", a["sha256"] == c["sha256"])
        print("drama-3 vs drama-3-preview identical:", a["sha256"] == d["sha256"])
        print("Sampling is random, so hashes differ even on the same model. Listen to "
              "hosted/{drama-3,drama-3-preview,s2.1-pro}/default/p1*.wav before sweeping.")
        return

    lenses = ["default", "cloning"] if args.lens == "both" else [args.lens]
    for lens in lenses:
        for pid in (int(p) for p in args.prompts.split(",")):
            for take in range(1, args.takes + 1):
                cell(key, args.model, lens, pid, by_id[pid], take, args.reference_id)


if __name__ == "__main__":
    main()
