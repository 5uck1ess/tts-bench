"""Multi-day speed sampler for the hosted board (docs/hosted.md, "Speed").

One pass = for every model, a fresh client, one untimed warmup, then CALLS timed warm calls of one
fixed English sentence. Vendors with both transports are measured on both, in alternating calls
so drift hits both equally. Rows append to hosted/speed/samples.jsonl; the board takes medians
across passes. Run a few times a day for 3+ days (Task Scheduler, see docs/hosted.md):

    venvs/hosted/Scripts/python.exe -m scripts.hosted.speed_sample
    venvs/hosted/Scripts/python.exe -m scripts.hosted.speed_sample --model cartesia_sonic36

Keys come from the DPAPI cache even past its 12 h TTL (keys.get(..., stale_ok=True)), so an
unattended pass never opens 1Password. No cache at all -> the pass exits without calling anything.
"""

import argparse
import datetime as dt
import json
import sys
import time
import traceback

from . import core, keys
from .providers import ADAPTERS
from .providers.batch_a import CartesiaSonic36, InworldTTS2
from .providers.batch_b import (AssemblyTTS1, DeepgramFluxTTS, SmallestLightning31Pro,
                                SonioxTTS2, StepAudio3TTS, XaiTTS)
from .providers.batch_c import MurfFalcon21
from .providers.ws import (AssemblyTTS1WS, CartesiaSonic36WS, DeepgramFluxTTSWS, InworldTTS2WS,
                           MurfFalcon21WS, SmallestLightning31ProWS, SonioxTTS2WS,
                           StepAudio3TTSWS, XaiTTSWS)

TEXT = ("I'll start a new git branch, push the changes, and open a pull request when the tests "
        "pass.")
CALLS = 3
OUT = core.OUT / "speed"
LOG = OUT / "samples.jsonl"
SKIP = {"speechmatics_tts"}  # built, not on the board

# slug -> (http class, websocket class) for vendors offering both.
BOTH = {c.slug: (c, w) for c, w in (
    (CartesiaSonic36, CartesiaSonic36WS), (AssemblyTTS1, AssemblyTTS1WS),
    (DeepgramFluxTTS, DeepgramFluxTTSWS), (XaiTTS, XaiTTSWS), (InworldTTS2, InworldTTS2WS),
    (SonioxTTS2, SonioxTTS2WS), (SmallestLightning31Pro, SmallestLightning31ProWS),
    (MurfFalcon21, MurfFalcon21WS), (StepAudio3TTS, StepAudio3TTSWS))}


def transport(ad):
    if not hasattr(ad, "synth_ws"):
        return "http"
    return "realtime-ws" if ad.route == "realtime-ws" else "websocket"


def close(ad):
    for ws in (getattr(ad, "_socks", None) or {}).values():
        try:
            ws.close()
        except Exception:  # noqa: BLE001
            pass


def row(pass_id, slug, ad, i, r=None, err=None):
    return {"pass": pass_id, "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "slug": slug, "transport": transport(ad), "call": i,
            "ttfa_speech_ms": r and r.get("ttfa_speech_ms"), "ttfa_ms": r and r.get("ttfa_ms"),
            "total_ms": r and r.get("total_ms"), "cold": r and r.get("cold"), "error": err}


def sample(slug, classes, pass_id, out):
    ads = [c(keys.get(c.provider, stale_ok=True)) for c in classes]
    clients = [core.new_client() for _ in ads]
    try:
        live = []
        for ad, cl in zip(ads, clients):
            try:
                core.synth(cl, ad, "Hello.", "en")  # warmup: connection + vendor session
                live.append((ad, cl))
            except Exception as e:  # noqa: BLE001
                out.append(row(pass_id, slug, ad, 0, err=f"warmup: {e!r}"[:300]))
        for i in range(1, CALLS + 1):
            for ad, cl in live:
                try:
                    out.append(row(pass_id, slug, ad, i, core.synth(cl, ad, TEXT, "en")))
                except Exception as e:  # noqa: BLE001
                    out.append(row(pass_id, slug, ad, i, err=repr(e)[:300]))
    finally:
        for ad, cl in zip(ads, clients):
            close(ad)
            cl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pass_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slugs = args.model or [s for s in ADAPTERS if s not in SKIP]
    n_ok = n_err = 0
    with open(OUT / "sampler.log", "a", encoding="utf-8") as log:
        log.write(f"{pass_id} start {len(slugs)} models\n")
        for slug in slugs:
            classes = BOTH.get(slug, (ADAPTERS[slug],))
            rows = []
            t = time.monotonic()
            try:
                sample(slug, classes, pass_id, rows)
            except SystemExit as e:  # keys.get: no cache / missing field
                log.write(f"{pass_id} {slug} ABORT {e}\n")
                return 1
            except Exception:  # noqa: BLE001
                log.write(f"{pass_id} {slug} CRASH {traceback.format_exc()[-400:]}\n")
            with open(LOG, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            ok = sum(1 for r in rows if not r["error"])
            n_ok, n_err = n_ok + ok, n_err + len(rows) - ok
            log.write(f"{pass_id} {slug} {ok}/{len(rows)} ok {time.monotonic() - t:.0f}s\n")
            log.flush()
        log.write(f"{pass_id} done ok={n_ok} err={n_err}\n")
    print(f"{pass_id}: ok={n_ok} err={n_err} -> {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
