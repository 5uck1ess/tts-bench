"""Load a clips_manifest.json into an in-memory Inventory for one lens.

Opaque clip ids keep model names out of the page DOM and the audio URL the
browser sees (the Space 302-redirects /clip/<id> to the gh-pages wav).
"""

import hashlib
from dataclasses import dataclass, field


def clip_id(mode: str, model: str, prompt: int) -> str:
    """Stable opaque id for a (mode, model, prompt) clip — no model name leaks."""
    raw = f"{mode}|{model}|{prompt}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


@dataclass
class Inventory:
    mode: str
    models: list = field(default_factory=list)
    by_prompt: dict = field(default_factory=dict)      # prompt -> [models] (>=2)
    id_of: dict = field(default_factory=dict)          # (model, prompt) -> clip_id
    url_of: dict = field(default_factory=dict)         # clip_id -> gh-pages url
    prompts: dict = field(default_factory=dict)        # prompt -> (lang, text)
    reference_url: str | None = None


def load_inventory(manifest: dict, mode: str, langs: set | None) -> Inventory:
    """Build the Inventory for ``mode``. ``langs`` (e.g. {"en"}) restricts which
    prompts are votable; None means all languages."""
    mblock = manifest["modes"][mode]
    prompts_raw = manifest.get("prompts", {})
    prompts = {int(pid): (lang, text) for pid, (lang, text) in prompts_raw.items()}

    inv = Inventory(mode=mode, prompts=prompts,
                    reference_url=mblock.get("reference_url"))

    present = {}  # prompt -> set(models)
    for c in mblock["clips"]:
        model, prompt, url = c["model"], int(c["prompt"]), c["url"]
        cid = clip_id(mode, model, prompt)
        inv.id_of[(model, prompt)] = cid
        inv.url_of[cid] = url
        present.setdefault(prompt, set()).add(model)

    inv.models = sorted({m for (m, _) in inv.id_of})
    for prompt, models in present.items():
        if langs is not None and prompts.get(prompt, ("en", ""))[0] not in langs:
            continue
        if len(models) >= 2:
            inv.by_prompt[prompt] = sorted(models)
    return inv
