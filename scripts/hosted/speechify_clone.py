"""Speechify voice clone with its live consent challenge (docs.speechify.ai, read 2026-11-01).

Speechify issues a one-time phrase; the speaker reads it aloud right then, and the recording goes
up with the voice sample. A challenge works once and expires, so run the two steps back to back:

    venvs/hosted/Scripts/python.exe -m scripts.hosted.speechify_clone challenge "Full Name"
    # read the printed phrase aloud, save it as reference/speechify_consent.wav (5-30 s)
    venvs/hosted/Scripts/python.exe -m scripts.hosted.speechify_clone create

`create` sends the 24.8 s cut (Speechify: 10-30 s, <5 MB) and records the voice id in
hosted/_private/clones.json, after which `--lens cloning --model speechify_simba35` runs as usual.
"""

import json
import sys
from datetime import datetime, timezone

import httpx

from . import core, keys

API = "https://api.speechify.ai/v1/voices"
STATE = core.PRIVATE / "speechify_challenge.json"
CONSENT = core.REF_DIR / "speechify_consent.wav"


def _h():
    return {"Authorization": f"Bearer {keys.get('speechify')}"}


def challenge(full_name):
    with core.new_client() as c:  # bounded transport
        r = c.post(f"{API}/consent-challenges", headers=_h(), json={"full_name": full_name})
    if r.status_code >= 300:
        sys.exit(f"HTTP {r.status_code}: {r.text[:400]}")
    j = r.json()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(j, indent=2), encoding="utf-8")
    print(f"Read this aloud now and save it as {CONSENT} (expires {j.get('expires_at')}):\n")
    print(f"    {j['phrase']}\n")


def create():
    ch = json.loads(STATE.read_text(encoding="utf-8"))
    ref = core.ref("30")
    with core.new_client() as c:
        r = c.post(API, headers=_h(),
                   data={"name": "tts-bench reference voice", "gender": "male",
                         "consent_challenge_id": ch["id"], "locale": "en-US"},
                   files={"sample": ("reference.wav", ref["wav"], "audio/wav"),
                          "consent_recording": ("consent.wav", CONSENT.read_bytes(),
                                                "audio/wav")})
    if r.status_code >= 300:
        sys.exit(f"HTTP {r.status_code}: {r.text[:400]}")
    vid = r.json()["id"]
    reg = json.loads(core.CLONES.read_text(encoding="utf-8")) if core.CLONES.exists() else {}
    reg["speechify_simba35"] = {"voice_id": vid, "cut": ref["cut"], "seconds": ref["seconds"],
                                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    core.CLONES.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    STATE.unlink(missing_ok=True)
    print(f"speechify voice {vid}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["challenge"]:
        challenge(" ".join(sys.argv[2:]))
    elif sys.argv[1:2] == ["create"]:
        create()
    else:
        sys.exit(__doc__)
