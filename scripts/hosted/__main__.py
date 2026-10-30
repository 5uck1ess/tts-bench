"""Hosted-API collector (docs/hosted.md).

    venvs/hosted/Scripts/python.exe -m scripts.hosted --list
    venvs/hosted/Scripts/python.exe -m scripts.hosted --model rime_coda --probe
    venvs/hosted/Scripts/python.exe -m scripts.hosted --model rime_coda --runs 3
"""

import argparse
import sys

from scoring.prompts import PROMPTS

from . import core, keys
from .providers import ADAPTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="row slug; repeatable")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true", help="p1, take 1, default lens only")
    ap.add_argument("--prompts", default="1,2,3,4,5,6")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--refresh-keys", action="store_true", help="re-fetch keys from 1Password")
    ap.add_argument("--only-failed", action="store_true",
                    help="re-run only cells whose sidecar is missing or status=error")
    args = ap.parse_args()
    if args.refresh_keys:
        keys.refresh()

    if args.list or not args.model:
        for slug, a in ADAPTERS.items():
            print(f"{slug:<18} {a.vendor:<12} {a.model:<20} {a.route}")
        return

    pids = [1] if args.probe else [int(p) for p in args.prompts.split(",")]
    runs = 1 if args.probe else args.runs
    by_id = {pid: (lang, text) for pid, lang, text in PROMPTS}
    for slug in args.model:
        if slug not in ADAPTERS:
            sys.exit(f"unknown model {slug}; --list")
        cls = ADAPTERS[slug]
        ad = cls(keys.get(cls.provider))
        with core.new_client() as client:
            if not args.probe:
                core.warmup(client, ad)  # every timed call then runs warm; sidecars say if not
            for take in range(1, runs + 1):
                for pid in pids:
                    lang, text = by_id[pid]
                    if args.only_failed and core.cell_status(ad, "default", pid, take) not in (
                            None, "error"):
                        continue
                    core.run_cell(client, ad, "default", pid, lang, text, take)


if __name__ == "__main__":
    main()
