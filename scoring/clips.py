"""Discover scorable clips in the published gh-pages worktree (pure stdlib).

A clip filename is `<model>_<cuda|mps|cpu>_p<digits>.wav` inside a `*-default/`
or `*-cloning/` canonical dir. The mode is taken from the dir suffix.
"""

import os
import re
from dataclasses import dataclass

_WAV_RE = re.compile(r"^(?P<model>.+)_(?P<dev>cuda|mps|cpu)_p(?P<pid>\d+)\.wav$")


@dataclass(frozen=True)
class Clip:
    dir: str        # canonical dir basename, e.g. "windows-cloning"
    wav: str        # filename
    model: str
    dev: str        # cuda | mps | cpu
    prompt_id: str  # "1".."5"
    mode: str       # "default" | "cloning"


def _mode_of(dir_name):
    if dir_name.endswith("-default"):
        return "default"
    if dir_name.endswith("-cloning"):
        return "cloning"
    return None


def parse_wav_name(dir_name, fn):
    """Return a Clip for a valid (dir, filename) pair, else None."""
    mode = _mode_of(dir_name)
    if mode is None:
        return None
    m = _WAV_RE.match(fn)
    if not m:
        return None
    return Clip(dir=dir_name, wav=fn, model=m.group("model"),
               dev=m.group("dev"), prompt_id=m.group("pid"), mode=mode)


def discover_clips(gh_pages_root):
    """All scorable clips under <root>/*-default/ and <root>/*-cloning/."""
    clips = []
    for entry in sorted(os.listdir(gh_pages_root)):
        if _mode_of(entry) is None:
            continue
        ad = os.path.join(gh_pages_root, entry)
        if not os.path.isdir(ad):
            continue
        for fn in sorted(os.listdir(ad)):
            c = parse_wav_name(entry, fn)
            if c is not None:
                clips.append(c)
    return clips
