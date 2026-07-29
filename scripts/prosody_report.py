"""Build the temporal-F0 corpus table and human-vs-TTS report.

The external tts-bench corpus is read in place.  Only the two output paths in
this worktree are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import parselmouth
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scoring.prosody import (  # noqa: E402
    MIN_VOICED,
    ProsodyMetrics,
    analyze_sound,
    analyze_wav,
    infer_pitch_bounds,
)


DEFAULT_CORPUS_ROOT = Path("C:/Users/tymra/LocalDev/tts-bench")
DEFAULT_SCORES = REPO_ROOT / "scoring" / "scores.csv"
DEFAULT_CSV = REPO_ROOT / "scoring" / "prosody.csv"
DEFAULT_REPORT = REPO_ROOT / "docs" / "prosody-report.md"

MIN_SEGMENT_S = 1.2
ENERGY_FRAME_S = 0.025
ENERGY_HOP_S = 0.010
MIN_SILENCE_S = 0.35
MIN_SOUND_S = 0.10
SEGMENT_PAD_S = 0.08

METRIC_FIELDS = [
    "declination_slope",
    "terminal_slope",
    "f0_range_st",
    "f0_sd_st",
    "duration_s",
    "voiced_frac",
    "n_voiced",
    "n_octave_rejected",
]
CSV_FIELDS = [
    "set",
    "language",
    "source_path",
    "source_hash",
    "segment_id",
    "segment_start_s",
    "segment_end_s",
    "dir",
    "wav",
    "model",
    "mode",
    "prompt_id",
    "join_source",
    "health",
    "wer",
    *METRIC_FIELDS,
    "status",
    "error",
]

_DEVICE_PROMPT_RE = re.compile(
    r"^(?P<model>.+)_(?:cpu|cuda|mps)_p(?P<prompt>[1-5])$"
)
_DEVICE_RE = re.compile(r"^(?P<model>.+)_(?:cpu|cuda|mps)$")
_PROMPT_RE = re.compile(r"(?:^|_)p(?P<prompt>[1-5])(?:_|$)")
_MOSS_AB_RE = re.compile(
    r"^(?P<mode>default|cloning)_p(?P<prompt>[1-5])_v(?P<version>.+)$"
)


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_status(metrics: ProsodyMetrics) -> str:
    if metrics.n_voiced < MIN_VOICED:
        return "too_few_voiced"
    if not math.isfinite(metrics.declination_slope):
        return "invalid_f0"
    return "ok"


def _failed_metrics() -> dict[str, object]:
    return {
        "declination_slope": float("nan"),
        "terminal_slope": float("nan"),
        "f0_range_st": float("nan"),
        "f0_sd_st": float("nan"),
        "duration_s": float("nan"),
        "voiced_frac": float("nan"),
        "n_voiced": 0,
        "n_octave_rejected": 0,
    }


def _language_for_prompt(prompt_id: object) -> str:
    prompt = str(prompt_id).strip()
    if prompt in {"1", "2", "3", "4"}:
        return "en"
    if prompt == "5":
        return "fr"
    return "unknown"


def _known_human_language(path: Path) -> str:
    """Language labels grounded in the repository's named reference files."""
    name = path.name.lower()
    if "juliette" in name:
        return "fr"
    if any(token in name for token in ("chris_hemsworth", "dafoe", "jo.wav")):
        return "en"
    return "unknown"


def _mode_from_parts(parts: Iterable[str]) -> str:
    for part in parts:
        if part.endswith("-cloning") or part == "cloning":
            return "cloning"
        if part.endswith("-default") or part == "default":
            return "default"
    return "unknown"


def _fallback_keys(relative_path: Path) -> dict[str, str]:
    """Parse only facts encoded in a path; never infer from run chronology."""
    stem = relative_path.stem
    special = _MOSS_AB_RE.match(stem)
    if special:
        return {
            "model": f"moss_tts_v{special.group('version')}",
            "mode": special.group("mode"),
            "prompt_id": special.group("prompt"),
        }

    match = _DEVICE_PROMPT_RE.match(stem)
    prompt_id = match.group("prompt") if match else ""
    if match:
        model = match.group("model")
    else:
        device_match = _DEVICE_RE.match(stem)
        model = device_match.group("model") if device_match else ""
        prompt_match = _PROMPT_RE.search(stem)
        if prompt_match:
            prompt_id = prompt_match.group("prompt")
        if not model and prompt_match:
            model = stem[: prompt_match.start()].rstrip("_")

    if not model and relative_path.parts[0] == "_preserved_moss_tts_v1.0":
        model = "moss_tts_v1.0"
    return {
        "model": model or "unknown",
        "mode": _mode_from_parts(relative_path.parts),
        "prompt_id": prompt_id,
    }


def _read_scores(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["dir"], row["wav"]): row
            for row in csv.DictReader(handle)
        }


def _generation_drop_reasons(score: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    health = score.get("health", "").strip()
    if health:
        reasons.append(f"health:{health}")
    wer = _float_or_nan(score.get("wer", ""))
    if math.isfinite(wer) and wer > 0.5:
        reasons.append("wer>0.5")
    return reasons


def _audio_content_hash(sound: parselmouth.Sound) -> str:
    """Hash decoded audio content rather than RIFF container bytes."""
    samples = np.ascontiguousarray(sound.values, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(
        (
            f"{float(sound.sampling_frequency):.9f}|"
            f"{samples.shape[0]}|{samples.shape[1]}|"
        ).encode("ascii")
    )
    digest.update(samples.tobytes(order="C"))
    return digest.hexdigest()


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if mask.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, mask.size]
    return [
        (int(start), int(end), bool(mask[start]))
        for start, end in zip(starts, ends, strict=True)
    ]


def _segment_on_silence(
    sound: parselmouth.Sound,
    *,
    min_segment_s: float = MIN_SEGMENT_S,
) -> list[tuple[float, float]]:
    """Energy-based utterance segmentation with an adaptive dBFS threshold."""
    samples = np.mean(np.asarray(sound.values, dtype=np.float64), axis=0)
    sample_rate = float(sound.sampling_frequency)
    duration_s = float(sound.get_total_duration())
    frame_n = max(1, int(round(ENERGY_FRAME_S * sample_rate)))
    hop_n = max(1, int(round(ENERGY_HOP_S * sample_rate)))
    if samples.size < frame_n:
        return [(0.0, duration_s)] if duration_s >= min_segment_s else []

    starts = np.arange(0, samples.size - frame_n + 1, hop_n, dtype=np.int64)
    squared = samples * samples
    cumulative = np.r_[0.0, np.cumsum(squared, dtype=np.float64)]
    mean_square = (
        cumulative[starts + frame_n] - cumulative[starts]
    ) / frame_n
    dbfs = 10.0 * np.log10(np.maximum(mean_square, 1.0e-12))
    threshold_db = max(-45.0, float(np.percentile(dbfs, 90)) - 30.0)
    sounding = dbfs >= threshold_db

    max_short_silence = max(1, int(round(MIN_SILENCE_S / ENERGY_HOP_S)))
    for start, end, is_sounding in _boolean_runs(sounding):
        if (
            not is_sounding
            and start > 0
            and end < sounding.size
            and end - start < max_short_silence
        ):
            sounding[start:end] = True

    min_sound_frames = max(1, int(round(MIN_SOUND_S / ENERGY_HOP_S)))
    for start, end, is_sounding in _boolean_runs(sounding):
        if is_sounding and end - start < min_sound_frames:
            sounding[start:end] = False

    intervals: list[tuple[float, float]] = []
    for start, end, is_sounding in _boolean_runs(sounding):
        if not is_sounding:
            continue
        start_s = max(0.0, starts[start] / sample_rate - SEGMENT_PAD_S)
        end_frame_start = starts[end - 1] / sample_rate
        end_s = min(
            duration_s, end_frame_start + ENERGY_FRAME_S + SEGMENT_PAD_S
        )
        if end_s - start_s >= min_segment_s:
            intervals.append((float(start_s), float(end_s)))
    return intervals


def _sound_part(
    sound: parselmouth.Sound, start_s: float, end_s: float
) -> parselmouth.Sound:
    sample_rate = float(sound.sampling_frequency)
    first = max(0, int(math.floor(start_s * sample_rate)))
    last = min(sound.n_frames, int(math.ceil(end_s * sample_rate)))
    return parselmouth.Sound(
        np.ascontiguousarray(sound.values[:, first:last]),
        sampling_frequency=sample_rate,
    )


def _deduplicate_human_sources(
    candidates: list[Path],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    groups: dict[str, dict[str, object]] = {}
    load_errors = 0
    for path in candidates:
        try:
            sound = parselmouth.Sound(str(path))
            content_hash = _audio_content_hash(sound)
        except Exception:
            load_errors += 1
            continue
        language = _known_human_language(path)
        if content_hash not in groups:
            groups[content_hash] = {
                "path": path,
                "hash": content_hash,
                "language": language,
                "copies": [path],
            }
            continue

        group = groups[content_hash]
        group["copies"].append(path)
        old_language = str(group["language"])
        if old_language == "unknown" and language != "unknown":
            group["language"] = language
            group["path"] = path
        elif (
            language != "unknown"
            and old_language != "unknown"
            and language != old_language
        ):
            group["language"] = "unknown"

    stats = {
        "candidates": len(candidates),
        "unique": len(groups),
        "duplicates": len(candidates) - len(groups) - load_errors,
        "load_errors": load_errors,
    }
    return list(groups.values()), stats


def _analyze_synthetic(
    results_root: Path,
    scores: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    wav_paths = sorted(
        path
        for path in results_root.rglob("*.wav")
        if path.name != "_reference.wav"
    )
    rows: list[dict[str, object]] = []
    synthetic_keys: set[tuple[str, str]] = set()
    matched_keys: set[tuple[str, str]] = set()
    drop_reason_counts: Counter[str] = Counter()
    failed_unique = 0

    for index, path in enumerate(wav_paths, start=1):
        relative = path.relative_to(results_root)
        dir_name = relative.parts[0]
        key = (dir_name, path.name)
        synthetic_keys.add(key)
        score = scores.get(key)
        if score is not None:
            matched_keys.add(key)
            reasons = _generation_drop_reasons(score)
            if reasons:
                failed_unique += 1
                drop_reason_counts.update(reasons)
                if len(reasons) > 1:
                    drop_reason_counts["both_health_and_wer"] += 1
                continue
            model = score.get("model", "").strip() or "unknown"
            mode = score.get("mode", "").strip() or "unknown"
            prompt_id = score.get("prompt_id", "").strip()
            join_source = "scores_csv"
            health = score.get("health", "").strip()
            wer = _float_or_nan(score.get("wer", ""))
        else:
            parsed = _fallback_keys(relative)
            model = parsed["model"]
            mode = parsed["mode"]
            prompt_id = parsed["prompt_id"]
            join_source = "filename_fallback"
            health = ""
            wer = float("nan")

        try:
            metrics = analyze_wav(path)
            metric_values = metrics.as_dict()
            status = _metric_status(metrics)
            error = ""
        except Exception as exc:
            metric_values = _failed_metrics()
            status = "analysis_error"
            error = f"{type(exc).__name__}: {exc}"

        rows.append(
            {
                "set": "synthetic",
                "language": _language_for_prompt(prompt_id),
                "source_path": relative.as_posix(),
                "source_hash": "",
                "segment_id": "",
                "segment_start_s": "",
                "segment_end_s": "",
                "dir": dir_name,
                "wav": path.name,
                "model": model,
                "mode": mode,
                "prompt_id": prompt_id,
                "join_source": join_source,
                "health": health,
                "wer": wer,
                **metric_values,
                "status": status,
                "error": error,
            }
        )
        if index % 100 == 0:
            print(f"synthetic: visited {index}/{len(wav_paths)} WAVs")

    stats: dict[str, object] = {
        "discovered": len(wav_paths),
        "score_rows": len(scores),
        "matched": len(matched_keys),
        "unmatched_wavs": len(wav_paths) - len(matched_keys),
        "score_rows_without_wav": len(set(scores) - synthetic_keys),
        "failed_unique": failed_unique,
        "retained": len(rows),
        "drop_reasons": dict(drop_reason_counts),
    }
    return rows, stats


def _analyze_human(
    corpus_root: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    candidates = sorted((corpus_root / "reference").glob("*.wav"))
    candidates += sorted(
        (corpus_root / "results").rglob("_reference.wav")
    )
    sources, stats = _deduplicate_human_sources(candidates)
    rows: list[dict[str, object]] = []
    short_segments = 0
    source_errors = 0
    segment_count = 0

    for source in sources:
        path = Path(source["path"])
        try:
            sound = parselmouth.Sound(str(path))
            bounds = infer_pitch_bounds(sound)
            intervals = _segment_on_silence(sound)
        except Exception:
            source_errors += 1
            continue

        source_duration = float(sound.get_total_duration())
        all_sounding = _segment_on_silence(sound, min_segment_s=0.0)
        short_segments += sum(
            (end_s - start_s) < MIN_SEGMENT_S
            for start_s, end_s in all_sounding
        )
        relative = path.relative_to(corpus_root)
        for source_segment_id, (start_s, end_s) in enumerate(
            intervals, start=1
        ):
            segment_count += 1
            segment = _sound_part(sound, start_s, end_s)
            try:
                metrics = analyze_sound(
                    segment,
                    pitch_floor=bounds[0],
                    pitch_ceiling=bounds[1],
                )
                metric_values = metrics.as_dict()
                status = _metric_status(metrics)
                error = ""
            except Exception as exc:
                metric_values = _failed_metrics()
                metric_values["duration_s"] = end_s - start_s
                status = "analysis_error"
                error = f"{type(exc).__name__}: {exc}"

            rows.append(
                {
                    "set": "human",
                    "language": source["language"],
                    "source_path": relative.as_posix(),
                    "source_hash": source["hash"],
                    "segment_id": source_segment_id,
                    "segment_start_s": start_s,
                    "segment_end_s": end_s,
                    "dir": relative.parts[0],
                    "wav": path.name,
                    "model": "human",
                    "mode": "human",
                    "prompt_id": "",
                    "join_source": "audio_hash_deduped",
                    "health": "",
                    "wer": float("nan"),
                    **metric_values,
                    "status": status,
                    "error": error,
                }
            )
        print(
            f"human: {relative.as_posix()} -> {len(intervals)} segments "
            f"from {source_duration:.1f}s"
        )

    stats.update(
        {
            "source_errors": source_errors,
            "segments": segment_count,
            "short_segments_ignored": short_segments,
        }
    )
    return rows, stats


def _fmt(value: object, digits: int = 3) -> str:
    number = _float_or_nan(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def _fmt_p(value: object) -> str:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return "NA"
    return "<0.001" if number < 0.001 else f"{number:.3f}"


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["status"] == "ok")
        & np.isfinite(df["declination_slope"])
        & np.isfinite(df["terminal_slope"])
    ].copy()


def _language_label(language: str) -> str:
    return {"en": "English (prompts 1-4)", "fr": "French (prompt 5)"}.get(
        language, language
    )


def _summary_values(
    valid: pd.DataFrame, set_name: str, language: str
) -> dict[str, float]:
    subset = valid[
        (valid["set"] == set_name) & (valid["language"] == language)
    ]
    return {
        "n": float(len(subset)),
        "declination_slope": float(subset["declination_slope"].median()),
        "terminal_slope": float(subset["terminal_slope"].median()),
    }


def _pattern_label(human: dict[str, float], synthetic: dict[str, float]) -> str:
    if not human["n"] or not synthetic["n"]:
        return "not estimable"
    less_negative_declination = (
        synthetic["declination_slope"] > human["declination_slope"]
    )
    weaker_terminal = abs(synthetic["terminal_slope"]) < abs(
        human["terminal_slope"]
    )
    if less_negative_declination and weaker_terminal:
        return "yes"
    if not less_negative_declination and not weaker_terminal:
        return "no"
    return "mixed"


def _per_model_frame(
    valid: pd.DataFrame, language: str
) -> pd.DataFrame:
    subset = valid[
        (valid["set"] == "synthetic") & (valid["language"] == language)
    ]
    grouped = (
        subset.groupby("model", as_index=False)
        .agg(
            n=("terminal_slope", "size"),
            terminal_slope=("terminal_slope", "median"),
            declination_slope=("declination_slope", "median"),
            fallback_n=(
                "join_source",
                lambda values: int((values == "filename_fallback").sum()),
            ),
        )
        .sort_values("model")
    )
    grouped["terminal_rank"] = (
        grouped["terminal_slope"].abs().rank(method="min", ascending=True)
    )
    grouped["declination_rank"] = grouped["declination_slope"].rank(
        method="min", ascending=False
    )
    return grouped.sort_values(
        ["terminal_rank", "declination_rank", "model"]
    ).reset_index(drop=True)


def _model_table_markdown(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| terminal rank | model | n | terminal median | "
        "|terminal| | declination median | declination rank | fallback n |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        model = str(row.model).replace("|", r"\|")
        lines.append(
            f"| {int(row.terminal_rank)} | {model} | {int(row.n)} | "
            f"{_fmt(row.terminal_slope)} | {_fmt(abs(row.terminal_slope))} | "
            f"{_fmt(row.declination_slope)} | "
            f"{int(row.declination_rank)} | {int(row.fallback_n)} |"
        )
    return lines


def _quantiles(series: pd.Series) -> tuple[float, float, float]:
    return (
        float(series.quantile(0.25)),
        float(series.median()),
        float(series.quantile(0.75)),
    )


def _iqr_overlap(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> bool:
    return max(left[0], right[0]) <= min(left[2], right[2])


def _dispersion_rows(
    valid: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, bool]]]:
    lines = [
        "| language | metric | human n | human median [IQR] | "
        "synthetic n | synthetic median [IQR] | IQR overlap? |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    overlap_results: dict[str, dict[str, bool]] = {}
    for language in ("en", "fr"):
        overlap_results[language] = {}
        for metric in ("f0_range_st", "f0_sd_st"):
            human = valid[
                (valid["set"] == "human")
                & (valid["language"] == language)
            ][metric]
            synthetic = valid[
                (valid["set"] == "synthetic")
                & (valid["language"] == language)
            ][metric]
            if human.empty or synthetic.empty:
                human_q = (float("nan"),) * 3
                synthetic_q = (float("nan"),) * 3
                overlap = False
            else:
                human_q = _quantiles(human)
                synthetic_q = _quantiles(synthetic)
                overlap = _iqr_overlap(human_q, synthetic_q)
            overlap_results[language][metric] = overlap
            lines.append(
                f"| {_language_label(language)} | {metric} | {len(human)} | "
                f"{_fmt(human_q[1])} [{_fmt(human_q[0])}, "
                f"{_fmt(human_q[2])}] | {len(synthetic)} | "
                f"{_fmt(synthetic_q[1])} [{_fmt(synthetic_q[0])}, "
                f"{_fmt(synthetic_q[2])}] | {'yes' if overlap else 'no'} |"
            )
    return lines, overlap_results


def _spearman_rows(valid: pd.DataFrame) -> list[str]:
    lines = [
        "| language | set | n | declination vs duration rho | p | "
        "terminal vs duration rho | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for language in ("en", "fr"):
        for set_name in ("human", "synthetic"):
            subset = valid[
                (valid["language"] == language)
                & (valid["set"] == set_name)
            ]
            if len(subset) < 3:
                decl_rho = decl_p = term_rho = term_p = float("nan")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    decl = spearmanr(
                        subset["duration_s"],
                        subset["declination_slope"],
                        nan_policy="omit",
                    )
                    term = spearmanr(
                        subset["duration_s"],
                        subset["terminal_slope"],
                        nan_policy="omit",
                    )
                decl_rho, decl_p = float(decl.statistic), float(decl.pvalue)
                term_rho, term_p = float(term.statistic), float(term.pvalue)
            lines.append(
                f"| {_language_label(language)} | {set_name} | "
                f"{len(subset)} | {_fmt(decl_rho)} | {_fmt_p(decl_p)} | "
                f"{_fmt(term_rho)} | {_fmt_p(term_p)} |"
            )
    return lines


def _report_markdown(
    df: pd.DataFrame,
    synthetic_stats: dict[str, object],
    human_stats: dict[str, int],
    corpus_root: Path,
    scores_path: Path,
) -> tuple[str, str]:
    valid = _valid(df)
    comparisons: dict[str, tuple[dict[str, float], dict[str, float], str]] = {}
    for language in ("en", "fr"):
        human = _summary_values(valid, "human", language)
        synthetic = _summary_values(valid, "synthetic", language)
        comparisons[language] = (
            human,
            synthetic,
            _pattern_label(human, synthetic),
        )

    labels = [comparisons[language][2] for language in ("en", "fr")]
    if all(label == "yes" for label in labels):
        headline_class = "YES"
    elif all(label == "no" for label in labels):
        headline_class = "NO"
    else:
        headline_class = "MIXED"
    headline = (
        f"{headline_class}: English={labels[0]}, French={labels[1]} for "
        "less-negative declination plus weaker absolute terminal movement."
    )

    human_valid_total = len(valid[valid["set"] == "human"])
    synthetic_valid_total = len(valid[valid["set"] == "synthetic"])
    human_underpowered = human_valid_total < 30
    failed_reasons = synthetic_stats["drop_reasons"]
    health_detail = ", ".join(
        f"{key.removeprefix('health:')}={value}"
        for key, value in sorted(failed_reasons.items())
        if key.startswith("health:")
    ) or "none"
    status_counts = (
        df.groupby(["set", "status"]).size().to_dict()
    )
    unknown_language = (
        df.groupby("set")["language"]
        .apply(lambda values: int((values == "unknown").sum()))
        .to_dict()
    )

    lines = [
        "# F0 temporal-structure report",
        "",
        f"**Headline: {headline}**",
        "",
        (
            "This is a descriptive corpus result, not a replication of the "
            "paper's experiment. “Weaker terminal movement” is operationalized "
            "as a smaller absolute median `terminal_slope`; declination is "
            "compared as a signed slope (higher means less negative)."
        ),
        "",
    ]
    if human_underpowered:
        lines += [
            (
                f"**UNDERPOWERED HUMAN BASELINE:** only {human_valid_total} "
                "metric-valid human segments survived. Every human-vs-synthetic "
                "statement below is underpowered."
            ),
            "",
        ]
    else:
        lines += [
            (
                f"The human side has {human_valid_total} metric-valid segments, "
                f"but only {human_stats['unique']} unique source clips; segments "
                "from a source are not independent. Human-vs-synthetic claims "
                "therefore remain weak."
            ),
            "",
        ]

    lines += [
        "## Human baseline and aggregate synthetic comparison",
        "",
        "| language | set | n | declination median | terminal median | "
        "|terminal| | paper pattern? |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for language in ("en", "fr"):
        human, synthetic, pattern = comparisons[language]
        for set_name, values in (("human", human), ("synthetic", synthetic)):
            lines.append(
                f"| {_language_label(language)} | {set_name} | "
                f"{int(values['n'])} | {_fmt(values['declination_slope'])} | "
                f"{_fmt(values['terminal_slope'])} | "
                f"{_fmt(abs(values['terminal_slope']))} | "
                f"{pattern if set_name == 'synthetic' else 'baseline'} |"
            )
    lines += [
        "",
        (
            "English prompts 1-4 and French prompt 5 are never pooled. Human "
            "language labels come only from named repository references: "
            "`juliette.wav` is French; `chris_hemsworth*`, `dafoe*`, and "
            "`jo.wav` are English. Distinct `_reference.wav` content without a "
            "named match is labeled unknown and excluded from language "
            "comparisons."
        ),
        "",
        "## Per-model ranking (primary result)",
        "",
        (
            "Terminal rank orders the smallest absolute median movement first "
            "(weakest movement). Declination rank orders the highest signed "
            "median first (least negative/most positive). `fallback n` counts "
            "clips whose keys were parsed from filenames because scores.csv "
            "did not join."
        ),
        "",
    ]
    for language in ("en", "fr"):
        frame = _per_model_frame(valid, language)
        human = comparisons[language][0]
        meets_both = int(
            (
                (frame["declination_slope"] > human["declination_slope"])
                & (
                    frame["terminal_slope"].abs()
                    < abs(human["terminal_slope"])
                )
            ).sum()
        ) if human["n"] else 0
        lines += [
            f"### {_language_label(language)}",
            "",
            (
                f"{meets_both}/{len(frame)} model medians meet both descriptive "
                "directions relative to the language-matched human baseline."
            ),
            "",
            *_model_table_markdown(frame),
            "",
        ]

    dispersion_lines, overlap_results = _dispersion_rows(valid)
    dispersion_labels = {
        language: all(overlap_results[language].values())
        for language in ("en", "fr")
    }
    lines += [
        "## Dispersion control",
        "",
        (
            "Overlap is defined before inspection as overlap between human and "
            "synthetic interquartile intervals. Metrics are computed from "
            "cleaned voiced frames before time interpolation."
        ),
        "",
        *dispersion_lines,
        "",
        (
            "Dispersion-control result: "
            f"English={'reproduced' if dispersion_labels['en'] else 'not reproduced'}; "
            f"French={'reproduced' if dispersion_labels['fr'] else 'not reproduced'}. "
            "This criterion requires IQR overlap for both `f0_range_st` and "
            "`f0_sd_st`."
        ),
        "",
        "## Duration control",
        "",
        (
            "Spearman correlations are utterance-level and descriptive. They do "
            "not account for repeated prompts, voices, models, or multiple "
            "segments from one human source."
        ),
        "",
        *_spearman_rows(valid),
        "",
        "## Counts and exclusions",
        "",
        f"- Synthetic WAVs discovered: {synthetic_stats['discovered']}.",
        (
            f"- Joined to `{scores_path.as_posix()}` on `(dir, wav)`: "
            f"{synthetic_stats['matched']}; filename fallback: "
            f"{synthetic_stats['unmatched_wavs']}."
        ),
        (
            f"- Score rows with no matching WAV under `{corpus_root.as_posix()}/"
            f"results`: {synthetic_stats['score_rows_without_wav']} of "
            f"{synthetic_stats['score_rows']}."
        ),
        (
            f"- Failed generations removed before F0 analysis: "
            f"{synthetic_stats['failed_unique']} unique WAVs; "
            f"health flags={health_detail} "
            f"(total {sum(value for key, value in failed_reasons.items() if key.startswith('health:'))}), "
            f"WER > 0.5={failed_reasons.get('wer>0.5', 0)}, "
            f"both={failed_reasons.get('both_health_and_wer', 0)}."
        ),
        (
            f"- Synthetic retained/analyzed: {synthetic_stats['retained']}; "
            f"metric-valid={synthetic_valid_total}; too few voiced frames="
            f"{status_counts.get(('synthetic', 'too_few_voiced'), 0)}; "
            f"analysis errors={status_counts.get(('synthetic', 'analysis_error'), 0)}."
        ),
        (
            f"- Human source WAV candidates: {human_stats['candidates']}; unique "
            f"decoded-audio hashes={human_stats['unique']}; duplicate copies="
            f"{human_stats['duplicates']}; source load errors="
            f"{human_stats['load_errors'] + human_stats['source_errors']}."
        ),
        (
            f"- Human segments >= {MIN_SEGMENT_S:.1f}s analyzed: "
            f"{human_stats['segments']}; metric-valid={human_valid_total}; too "
            f"few voiced frames={status_counts.get(('human', 'too_few_voiced'), 0)}; "
            f"analysis errors={status_counts.get(('human', 'analysis_error'), 0)}."
        ),
        (
            f"- Rows with unknown language (excluded from EN/FR comparisons): "
            f"synthetic={unknown_language.get('synthetic', 0)}, "
            f"human={unknown_language.get('human', 0)}."
        ),
        "",
        "## Method",
        "",
        (
            "Praat autocorrelation F0 uses a broad 50-800 Hz pass to estimate "
            "each clip/source voice, followed by an adaptive median/2.5 to "
            "median*2.5 floor/ceiling. Unvoiced frames are dropped. Contiguous "
            "voiced runs receive a five-frame median filter; remaining adjacent "
            "jumps over 12 semitones reject the member farther from the run "
            "median. Each valid contour is converted to semitones relative to "
            "its voiced-frame median and linearly interpolated to 100 points "
            "from the first through last cleaned voiced frame (internal "
            "unvoiced gaps retain their timing; container-edge padding is not "
            "extrapolated). OLS slopes use normalized time 0-1 globally and "
            "0.80-1.00 terminally. Dispersion uses the cleaned, uninterpolated "
            "frames."
        ),
        "",
        (
            f"Human sources are deduplicated by SHA-256 over decoded float32 "
            f"audio content and segmented with {ENERGY_FRAME_S * 1000:.0f} ms "
            f"energy frames/{ENERGY_HOP_S * 1000:.0f} ms hops, an adaptive "
            f"max(-45 dBFS, p90-30 dB) threshold, and {MIN_SILENCE_S:.2f}s "
            f"minimum silence. Segments shorter than {MIN_SEGMENT_S:.1f}s are "
            "ignored."
        ),
        "",
        "## Limitations",
        "",
        (
            "- The human baseline is a handful of source recordings. Silence-"
            "derived segments from the same clip are correlated and do not "
            "create independent speakers."
        ),
        (
            "- Human segments are not text- or duration-matched to the five "
            "synthetic prompts. The duration correlations diagnose, but do not "
            "remove, this confound."
        ),
        (
            f"- The `(dir, wav)` join covered only {synthetic_stats['matched']}/"
            f"{synthetic_stats['discovered']} discovered synthetic WAVs. "
            "Filename-fallback rows have no health/WER gate, and mode is unknown "
            "when it is not encoded in the path."
        ),
        (
            "- The recursive synthetic set includes historical runs, smoke/"
            "comparison artifacts, and potentially repeated audio. It was not "
            "deduplicated because the requested set was every non-reference WAV."
        ),
        (
            "- Per-model medians with small n are unstable. Rows are not "
            "statistically independent across devices, modes, prompts, repeated "
            "runs, or shared cloning references."
        ),
        (
            "- Terminal slope is especially sensitive to utterance duration, "
            "endpoint voicing, and silence segmentation; normalized time does "
            "not eliminate those effects."
        ),
        (
            "- Automatic pitch bounds and median filtering reduce octave errors "
            "but cannot guarantee that every tracker error is removed, especially "
            "for creaky voice, music, noise, or non-speech artifacts."
        ),
        (
            "- The result is descriptive medians without uncertainty intervals, "
            "mixed-effects modeling, or a prompt-matched inferential test. It "
            "cannot establish that model architecture causes over-smoothing."
        ),
        "",
    ]
    return "\n".join(lines), headline


def build_report(
    corpus_root: Path,
    scores_path: Path,
    csv_path: Path,
    report_path: Path,
) -> tuple[pd.DataFrame, str]:
    results_root = corpus_root / "results"
    reference_root = corpus_root / "reference"
    for required in (results_root, reference_root, scores_path):
        if not required.exists():
            raise FileNotFoundError(f"required input is missing: {required}")

    scores = _read_scores(scores_path)
    synthetic_rows, synthetic_stats = _analyze_synthetic(results_root, scores)
    human_rows, human_stats = _analyze_human(corpus_root)
    df = pd.DataFrame(synthetic_rows + human_rows, columns=CSV_FIELDS)
    df = df.sort_values(
        ["set", "language", "model", "source_path", "segment_id"],
        kind="stable",
    ).reset_index(drop=True)

    report, headline = _report_markdown(
        df, synthetic_stats, human_stats, corpus_root, scores_path
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, na_rep="")
    report_path.write_text(report, encoding="utf-8")
    return df, headline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df, headline = build_report(
        args.corpus_root.resolve(),
        args.scores.resolve(),
        args.csv.resolve(),
        args.report.resolve(),
    )
    print(headline)
    print(f"wrote {len(df)} rows to {args.csv}")
    print(f"wrote report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
