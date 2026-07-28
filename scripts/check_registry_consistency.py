#!/usr/bin/env python3
"""Check that every harness model has complete report metadata."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name
                   for target in node.targets):
                return node.value
    raise ValueError(f"assignment not found: {name}")


def _registry_slugs(node: ast.expr, name: str) -> set[str]:
    if isinstance(node, ast.Dict):
        entries = node.keys
    elif isinstance(node, ast.Set):
        entries = node.elts
    else:
        raise ValueError(f"{name} is not a dictionary or set literal")
    slugs = []
    for entry in entries:
        if not isinstance(entry, ast.Constant) or not isinstance(entry.value, str):
            raise ValueError(f"{name} has a non-literal slug")
        slugs.append(entry.value)
    return set(slugs)


def main() -> int:
    harness_tree = ast.parse((ROOT / "harness.py").read_text(encoding="utf-8"))
    report_tree = ast.parse((ROOT / "report.py").read_text(encoding="utf-8"))

    models = ast.literal_eval(_assignment(harness_tree, "MODELS"))
    harness_slugs = {row[0] for row in models}

    registry_node = _assignment(report_tree, "_REGISTRIES")
    if not isinstance(registry_node, ast.Dict):
        raise ValueError("_REGISTRIES is not a dictionary literal")

    registries = {}
    for key, value in zip(registry_node.keys, registry_node.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError("_REGISTRIES has a non-literal name")
        if not isinstance(value, ast.Name):
            raise ValueError(f"_REGISTRIES[{key.value!r}] is not a named registry")
        registries[key.value] = _registry_slugs(
            _assignment(report_tree, value.id), value.id)

    gaps = {
        name: sorted(harness_slugs - slugs)
        for name, slugs in registries.items()
        if harness_slugs - slugs
    }
    if gaps:
        print("Registry consistency check failed:")
        for name, missing in gaps.items():
            print(f"  {name}: {', '.join(missing)}")
        return 1

    print(
        f"Registry consistency: OK ({len(harness_slugs)} harness models, "
        f"{len(registries)} report registries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
