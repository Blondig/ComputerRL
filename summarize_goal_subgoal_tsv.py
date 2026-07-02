#!/usr/bin/env python3
"""
Summarize task_subgoal_sequences.tsv outputs from analyze_goal_subgoal_gap.py.

The goal is to decide whether goal/subgoal clusters support reusable procedure
memory, progress monitoring, or only stall diagnostics. This script reads TSV
details, not the markdown report, and it does not depend on v3 action_sig.

Typical usage:

  python summarize_goal_subgoal_tsv.py /path/to/probe_out \
      --out-dir /tmp/goal_subgoal_tsv_summary

Inputs:
  - task_subgoal_sequences.tsv (required; files or dirs accepted)
  - task_goal_table.tsv (optional; auto-loaded from sibling dirs when present)

Outputs:
  - tsv_summary_report.md
  - goal_candidates.jsonl
  - goal_domain_matrix.tsv
  - subgoal_gap_table.tsv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


LOWLEVEL_ROLES = {"lowlevel", "lowlevel_select", "lowlevel_input"}
CORE_ROLES = {"inspect", "locate", "transform", "edit", "format", "create", "configure", "commit", "media"}
NOISY_STALL_KEYS = {"n_steps", "n_roles"}


def safe_json(text: Any, default: Optional[dict] = None) -> dict:
    if default is None:
        default = {}
    if text in (None, ""):
        return dict(default)
    try:
        value = json.loads(str(text))
    except Exception:
        return dict(default)
    return value if isinstance(value, dict) else dict(default)


def parse_bool(text: Any) -> Optional[bool]:
    if text in (None, ""):
        return None
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return None


def parse_float(text: Any) -> Optional[float]:
    if text in (None, ""):
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_int(text: Any, default: int = 0) -> int:
    if text in (None, ""):
        return default
    try:
        return int(float(str(text)))
    except Exception:
        return default


def split_seq(text: Any) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(">") if part.strip()]


def pct(value: float) -> str:
    if math.isnan(value):
        return "-"
    return "{:.0f}%".format(value * 100.0)


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return "{:.4f}".format(value)


def relpath(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def collect_tsv_files(paths: Sequence[str], basename: str) -> List[str]:
    out: List[str] = []
    for path in paths:
        if not path:
            continue
        if os.path.isfile(path):
            if os.path.basename(path) == basename or path.endswith(".tsv"):
                out.append(path)
        elif os.path.isdir(path):
            out.extend(glob.glob(os.path.join(path, "**", basename), recursive=True))
    return sorted(set(out))


@dataclass
class SeqRow:
    tsv_path: str
    run_id: str
    task_id: str
    domain: str
    app: str
    source: str
    goal_key: str
    metadata_strength: str
    score: Optional[float]
    baseline_score: Optional[float]
    success: Optional[bool]
    label: str
    n_steps: int
    n_errors: int
    injected_count: int
    l1_seq: List[str]
    l2_seq: List[str]
    source_counts: Dict[str, Any]
    stall_signals: Dict[str, Any]
    artifact_paths: str

    @property
    def key(self) -> Tuple[str, str]:
        return self.domain, self.goal_key


@dataclass
class GoalTaskInfo:
    task_id: str
    instruction: str = ""
    task_cluster_key: str = ""
    ops: str = ""
    objs: str = ""


@dataclass
class FeatureGap:
    feature: str
    pass_rate: float
    fail_rate: float
    gap: float
    direction: str
    level: str


@dataclass
class GoalSummary:
    domain: str
    goal_key: str
    rows: List[SeqRow]
    n: int
    pass_n: int
    fail_n: int
    unknown_n: int
    pass_rate: float
    l1_missing: List[FeatureGap] = field(default_factory=list)
    l2_missing: List[FeatureGap] = field(default_factory=list)
    l1_fail_excess: List[FeatureGap] = field(default_factory=list)
    stall_gaps: List[FeatureGap] = field(default_factory=list)
    pass_prefix: List[str] = field(default_factory=list)
    pass_prefix_support: float = 0.0
    pass_patterns: Counter = field(default_factory=Counter)
    fail_patterns: Counter = field(default_factory=Counter)
    stall_rates: Dict[str, float] = field(default_factory=dict)
    decision: str = "no_gap"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    sample_tasks: List[str] = field(default_factory=list)


def read_sequences(tsv_files: Sequence[str]) -> List[SeqRow]:
    rows: List[SeqRow] = []
    for path in tsv_files:
        try:
            f = open(path, encoding="utf-8", errors="replace", newline="")
        except OSError:
            continue
        with f:
            reader = csv.DictReader(f, delimiter="\t")
            for raw in reader:
                if not raw.get("task_id"):
                    continue
                rows.append(
                    SeqRow(
                        tsv_path=path,
                        run_id=raw.get("run_id", ""),
                        task_id=raw.get("task_id", ""),
                        domain=raw.get("domain", "") or "unknown",
                        app=raw.get("app", "") or "unknown",
                        source=raw.get("source", ""),
                        goal_key=raw.get("goal_key", "") or "unknown",
                        metadata_strength=raw.get("metadata_strength", ""),
                        score=parse_float(raw.get("score")),
                        baseline_score=parse_float(raw.get("baseline_score")),
                        success=parse_bool(raw.get("success")),
                        label=raw.get("label", ""),
                        n_steps=parse_int(raw.get("n_steps")),
                        n_errors=parse_int(raw.get("n_errors")),
                        injected_count=parse_int(raw.get("injected_count")),
                        l1_seq=split_seq(raw.get("l1_seq")),
                        l2_seq=split_seq(raw.get("l2_seq")),
                        source_counts=safe_json(raw.get("source_counts")),
                        stall_signals=safe_json(raw.get("stall_signals")),
                        artifact_paths=raw.get("artifact_paths", ""),
                    )
                )
    return rows


def read_task_info(goal_tables: Sequence[str]) -> Dict[str, GoalTaskInfo]:
    info: Dict[str, GoalTaskInfo] = {}
    for path in goal_tables:
        try:
            f = open(path, encoding="utf-8", errors="replace", newline="")
        except OSError:
            continue
        with f:
            reader = csv.DictReader(f, delimiter="\t")
            for raw in reader:
                task_id = raw.get("task_id", "")
                if not task_id:
                    continue
                info[task_id] = GoalTaskInfo(
                    task_id=task_id,
                    instruction=raw.get("instruction", ""),
                    task_cluster_key=raw.get("task_cluster_key", ""),
                    ops=raw.get("ops", ""),
                    objs=raw.get("objs", ""),
                )
    return info


def sibling_goal_tables(tsv_files: Sequence[str]) -> List[str]:
    out = []
    for path in tsv_files:
        sibling = os.path.join(os.path.dirname(path), "task_goal_table.tsv")
        if os.path.isfile(sibling):
            out.append(sibling)
    return sorted(set(out))


def rate(rows: Sequence[SeqRow], pred: Callable[[SeqRow], bool]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if pred(row)) / len(rows)


def feature_gap(
    pass_rows: Sequence[SeqRow],
    fail_rows: Sequence[SeqRow],
    level: str,
    feature_fn: Callable[[SeqRow], Iterable[str]],
) -> Tuple[List[FeatureGap], List[FeatureGap]]:
    all_features = sorted(set(f for row in list(pass_rows) + list(fail_rows) for f in set(feature_fn(row))))
    missing: List[FeatureGap] = []
    fail_excess: List[FeatureGap] = []
    for feature in all_features:
        pr = rate(pass_rows, lambda row, feature=feature: feature in set(feature_fn(row)))
        fr = rate(fail_rows, lambda row, feature=feature: feature in set(feature_fn(row)))
        if math.isnan(pr) or math.isnan(fr):
            continue
        gap = pr - fr
        if gap >= 0:
            missing.append(FeatureGap(feature, pr, fr, gap, "pass_minus_fail", level))
        else:
            fail_excess.append(FeatureGap(feature, pr, fr, -gap, "fail_minus_pass", level))
    missing.sort(key=lambda g: (g.gap, g.pass_rate, -g.fail_rate), reverse=True)
    fail_excess.sort(key=lambda g: (g.gap, g.fail_rate, -g.pass_rate), reverse=True)
    return missing, fail_excess


def stall_features(row: SeqRow) -> List[str]:
    out = []
    for key, value in row.stall_signals.items():
        if key in NOISY_STALL_KEYS:
            continue
        try:
            active = float(value) > 0
        except Exception:
            active = bool(value)
        if active:
            out.append(key)
    return out


def stable_prefix(rows: Sequence[SeqRow], max_len: int, support: float) -> Tuple[List[str], float]:
    pass_rows = [row for row in rows if row.success is True and row.l1_seq]
    if not pass_rows:
        return [], 0.0
    prefix: List[str] = []
    for idx in range(max_len):
        counts = Counter(row.l1_seq[idx] for row in pass_rows if len(row.l1_seq) > idx)
        if not counts:
            break
        token, count = counts.most_common(1)[0]
        if count / len(pass_rows) < support:
            break
        prefix.append(token)
    if not prefix:
        return [], 0.0
    support_count = sum(1 for row in pass_rows if row.l1_seq[: len(prefix)] == prefix)
    return prefix, support_count / len(pass_rows)


def sequence_patterns(rows: Sequence[SeqRow], max_len: int = 6) -> Counter:
    patterns = Counter()
    for row in rows:
        seq = row.l1_seq[:max_len]
        if seq:
            patterns[" > ".join(seq)] += 1
    return patterns


def top_filtered(gaps: Sequence[FeatureGap], min_gap: float, min_rate: float, pass_side: bool) -> List[FeatureGap]:
    out = []
    for gap in gaps:
        anchor = gap.pass_rate if pass_side else gap.fail_rate
        if gap.gap >= min_gap and anchor >= min_rate:
            out.append(gap)
    return out


def summarize_goal(
    domain: str,
    goal_key: str,
    rows: Sequence[SeqRow],
    task_info: Dict[str, GoalTaskInfo],
    min_gap: float,
    min_stall_gap: float,
    prefix_support: float,
) -> GoalSummary:
    pass_rows = [row for row in rows if row.success is True]
    fail_rows = [row for row in rows if row.success is False]
    unknown_rows = [row for row in rows if row.success is None]
    n = len(rows)
    pass_n = len(pass_rows)
    fail_n = len(fail_rows)
    pass_rate = pass_n / n if n else float("nan")

    l1_missing_all, l1_fail_excess_all = feature_gap(pass_rows, fail_rows, "l1", lambda row: row.l1_seq)
    l2_missing_all, _l2_fail_excess_all = feature_gap(pass_rows, fail_rows, "l2", lambda row: row.l2_seq)
    stall_missing_all, stall_fail_excess_all = feature_gap(pass_rows, fail_rows, "stall", stall_features)

    prefix, prefix_rate = stable_prefix(pass_rows, max_len=5, support=prefix_support)
    pass_patterns = sequence_patterns(pass_rows)
    fail_patterns = sequence_patterns(fail_rows)

    stall_names = sorted(set(k for row in rows for k in row.stall_signals if k not in NOISY_STALL_KEYS))
    stall_rates = {
        key: rate(rows, lambda row, key=key: key in stall_features(row))
        for key in stall_names
    }

    strong_l1 = top_filtered(l1_missing_all, min_gap=min_gap, min_rate=0.5, pass_side=True)
    strong_l2 = top_filtered(l2_missing_all, min_gap=min_gap, min_rate=0.5, pass_side=True)
    strong_stall = top_filtered(stall_fail_excess_all, min_gap=min_stall_gap, min_rate=0.4, pass_side=False)

    content_l1 = [g for g in strong_l1 if g.feature not in LOWLEVEL_ROLES and g.feature != "setup"]
    content_prefix = [role for role in prefix if role in CORE_ROLES]
    lowlevel_loop_rate = stall_rates.get("lowlevel_loop", 0.0)
    no_core_rate = stall_rates.get("no_core_action", 0.0)

    notes: List[str] = []
    if pass_n == 0 or fail_n == 0:
        decision = "one_sided_or_no_gap"
        notes.append("needs both pass and fail examples")
    elif content_l1 or (pass_n >= 2 and len(prefix) >= 2 and content_prefix):
        decision = "progress_memory_candidate"
        if content_l1:
            notes.append("failures miss content subgoals")
        if pass_n >= 2 and len(prefix) >= 2:
            notes.append("successes share an L1 prefix")
    elif strong_stall:
        decision = "stall_monitor_only"
        notes.append("failure gap is mostly stall/noise")
    elif strong_l1 or strong_l2:
        decision = "weak_progress_candidate"
        notes.append("gap exists but is setup/lowlevel-heavy")
    else:
        decision = "no_clear_gap"

    if lowlevel_loop_rate >= 0.7 or no_core_rate >= 0.7:
        notes.append("lowlevel/no-core dominated")

    best_content_gap = max([g.gap for g in content_l1] + [0.0])
    best_stall_gap = max([g.gap for g in strong_stall] + [0.0])
    support_score = min(n / 10.0, 1.0) * 0.25 + min(pass_n / 3.0, 1.0) * 0.25 + min(fail_n / 3.0, 1.0) * 0.25
    signal_score = min(best_content_gap, 1.0) * 0.2 + min(best_stall_gap, 1.0) * 0.1 + min(prefix_rate, 1.0) * 0.1
    noise_penalty = 0.15 if ("lowlevel/no-core dominated" in notes and not content_l1) else 0.0
    confidence = max(0.0, min(1.0, support_score + signal_score - noise_penalty))

    samples = []
    for row in rows[:20]:
        info = task_info.get(row.task_id)
        if info and info.instruction:
            samples.append("{}: {}".format(row.task_id, info.instruction[:160]))
        elif row.task_id:
            samples.append(row.task_id)
        if len(samples) >= 3:
            break

    return GoalSummary(
        domain=domain,
        goal_key=goal_key,
        rows=list(rows),
        n=n,
        pass_n=pass_n,
        fail_n=fail_n,
        unknown_n=len(unknown_rows),
        pass_rate=pass_rate,
        l1_missing=l1_missing_all,
        l2_missing=l2_missing_all,
        l1_fail_excess=l1_fail_excess_all,
        stall_gaps=stall_fail_excess_all,
        pass_prefix=prefix,
        pass_prefix_support=prefix_rate,
        pass_patterns=pass_patterns,
        fail_patterns=fail_patterns,
        stall_rates=stall_rates,
        decision=decision,
        confidence=confidence,
        notes=notes,
        sample_tasks=samples,
    )


def summarize_all(
    rows: Sequence[SeqRow],
    task_info: Dict[str, GoalTaskInfo],
    min_goal_size: int,
    min_gap: float,
    min_stall_gap: float,
    prefix_support: float,
) -> List[GoalSummary]:
    grouped: Dict[Tuple[str, str], List[SeqRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    summaries = []
    for (domain, goal_key), vals in grouped.items():
        if len(vals) < min_goal_size:
            continue
        summaries.append(summarize_goal(domain, goal_key, vals, task_info, min_gap, min_stall_gap, prefix_support))
    summaries.sort(key=lambda s: (s.decision == "progress_memory_candidate", s.confidence, s.n), reverse=True)
    return summaries


def gap_rows(summaries: Sequence[GoalSummary], min_report_gap: float) -> Iterable[dict]:
    for s in summaries:
        for level, gaps in (
            ("l1", s.l1_missing + s.l1_fail_excess),
            ("l2", s.l2_missing),
            ("stall", s.stall_gaps),
        ):
            for gap in gaps:
                if gap.gap < min_report_gap:
                    continue
                yield {
                    "domain": s.domain,
                    "goal_key": s.goal_key,
                    "level": level,
                    "feature": gap.feature,
                    "direction": gap.direction,
                    "pass_rate": fmt_float(gap.pass_rate),
                    "fail_rate": fmt_float(gap.fail_rate),
                    "gap": fmt_float(gap.gap),
                    "n": s.n,
                    "pass_n": s.pass_n,
                    "fail_n": s.fail_n,
                    "decision": s.decision,
                }


def matrix_rows(summaries: Sequence[GoalSummary]) -> Iterable[dict]:
    for s in sorted(summaries, key=lambda x: (x.domain, x.goal_key)):
        best_l1 = s.l1_missing[0] if s.l1_missing else None
        best_stall = s.stall_gaps[0] if s.stall_gaps else None
        yield {
            "domain": s.domain,
            "goal_key": s.goal_key,
            "n": s.n,
            "pass_n": s.pass_n,
            "fail_n": s.fail_n,
            "unknown_n": s.unknown_n,
            "pass_rate": fmt_float(s.pass_rate),
            "decision": s.decision,
            "confidence": fmt_float(s.confidence),
            "best_missing_l1": "" if best_l1 is None else best_l1.feature,
            "best_missing_l1_gap": "" if best_l1 is None else fmt_float(best_l1.gap),
            "best_fail_stall": "" if best_stall is None else best_stall.feature,
            "best_fail_stall_gap": "" if best_stall is None else fmt_float(best_stall.gap),
            "pass_prefix": " > ".join(s.pass_prefix),
            "pass_prefix_support": fmt_float(s.pass_prefix_support),
            "notes": "; ".join(s.notes),
        }


def candidate_rows(summaries: Sequence[GoalSummary], top: int) -> Iterable[dict]:
    interesting = [s for s in summaries if s.decision in {"progress_memory_candidate", "weak_progress_candidate", "stall_monitor_only"}]
    interesting.sort(key=lambda s: (s.decision == "progress_memory_candidate", s.confidence, s.n), reverse=True)
    for s in interesting[:top]:
        yield {
            "domain": s.domain,
            "goal_key": s.goal_key,
            "decision": s.decision,
            "confidence": round(s.confidence, 4),
            "n": s.n,
            "pass_n": s.pass_n,
            "fail_n": s.fail_n,
            "pass_rate": round(s.pass_rate, 4) if not math.isnan(s.pass_rate) else None,
            "missing_l1": [
                {"feature": g.feature, "pass_rate": g.pass_rate, "fail_rate": g.fail_rate, "gap": g.gap}
                for g in s.l1_missing[:5]
            ],
            "missing_l2": [
                {"feature": g.feature, "pass_rate": g.pass_rate, "fail_rate": g.fail_rate, "gap": g.gap}
                for g in s.l2_missing[:5]
            ],
            "fail_stalls": [
                {"feature": g.feature, "pass_rate": g.pass_rate, "fail_rate": g.fail_rate, "gap": g.gap}
                for g in s.stall_gaps[:5]
            ],
            "pass_prefix": s.pass_prefix,
            "pass_prefix_support": s.pass_prefix_support,
            "top_pass_patterns": s.pass_patterns.most_common(5),
            "top_fail_patterns": s.fail_patterns.most_common(5),
            "notes": s.notes,
            "sample_tasks": s.sample_tasks,
        }


def write_tsv(path: str, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_report(rows: Sequence[SeqRow], tsv_files: Sequence[str], summaries: Sequence[GoalSummary], top: int) -> str:
    lines: List[str] = []
    lines.append("# TSV Goal/Subgoal Summary")
    lines.append("")
    lines.append("This report reads task_subgoal_sequences.tsv details, not the markdown summary.")
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append("- tsv files: {}".format(len(tsv_files)))
    lines.append("- observations: {}".format(len(rows)))
    lines.append("- grouped goals reported: {}".format(len(summaries)))
    lines.append("")
    if tsv_files:
        lines.append("### Input Files")
        lines.append("")
        for path in tsv_files[:20]:
            lines.append("- `{}`".format(relpath(path)))
        if len(tsv_files) > 20:
            lines.append("- ... {} more".format(len(tsv_files) - 20))
        lines.append("")

    by_domain = Counter(row.domain for row in rows)
    pass_by_domain = Counter(row.domain for row in rows if row.success is True)
    fail_by_domain = Counter(row.domain for row in rows if row.success is False)
    lines.append("## Domain Overview")
    lines.append("")
    lines.append("| domain | n | pass | fail | pass_rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for domain, n in by_domain.most_common():
        p = pass_by_domain[domain]
        f = fail_by_domain[domain]
        lines.append("| {} | {} | {} | {} | {} |".format(domain, n, p, f, pct(p / n if n else float("nan"))))
    lines.append("")

    decisions = Counter(s.decision for s in summaries)
    lines.append("## Decision Mix")
    lines.append("")
    for name, count in decisions.most_common():
        lines.append("- {}: {}".format(name, count))
    lines.append("")

    def top_for(decision: str) -> List[GoalSummary]:
        vals = [s for s in summaries if s.decision == decision]
        vals.sort(key=lambda s: (s.confidence, s.n), reverse=True)
        return vals[:top]

    for title, decision in (
        ("Progress Memory Candidates", "progress_memory_candidate"),
        ("Weak Progress Candidates", "weak_progress_candidate"),
        ("Stall Monitor Only", "stall_monitor_only"),
        ("Largest No-Clear-Gap Groups", "no_clear_gap"),
    ):
        vals = top_for(decision)
        if not vals:
            continue
        lines.append("## {}".format(title))
        lines.append("")
        for s in vals:
            lines.append("### `{}` / `{}`".format(s.domain, s.goal_key))
            lines.append("")
            lines.append("- n/pass/fail: {}/{}/{}".format(s.n, s.pass_n, s.fail_n))
            lines.append("- confidence: {:.2f}".format(s.confidence))
            if s.pass_prefix:
                lines.append("- pass prefix: `{}` ({})".format(" > ".join(s.pass_prefix), pct(s.pass_prefix_support)))
            if s.l1_missing:
                bits = [
                    "{} pass={} fail={} gap=+{:.0f}pp".format(g.feature, pct(g.pass_rate), pct(g.fail_rate), g.gap * 100.0)
                    for g in s.l1_missing[:4]
                ]
                lines.append("- pass-minus-fail L1: " + "; ".join(bits))
            if s.stall_gaps:
                bits = [
                    "{} pass={} fail={} gap=+{:.0f}pp".format(g.feature, pct(g.pass_rate), pct(g.fail_rate), g.gap * 100.0)
                    for g in s.stall_gaps[:4]
                ]
                lines.append("- fail-minus-pass stalls: " + "; ".join(bits))
            if s.pass_patterns:
                lines.append("- top pass patterns: `{}`".format(s.pass_patterns.most_common(3)))
            if s.fail_patterns:
                lines.append("- top fail patterns: `{}`".format(s.fail_patterns.most_common(3)))
            if s.notes:
                lines.append("- notes: {}".format("; ".join(s.notes)))
            lines.append("")

    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `progress_memory_candidate`: content subgoals or success prefixes may support goal-conditioned progress memory.")
    lines.append("- `stall_monitor_only`: signal is mainly lowlevel/error/no-core stall; useful for recovery diagnostics, not procedure replay.")
    lines.append("- `weak_progress_candidate`: gap exists but is setup/lowlevel-heavy or sparse.")
    lines.append("- `one_sided_or_no_gap`: lacks both pass and fail evidence.")
    return "\n".join(lines).rstrip() + "\n"


def print_summary(rows: Sequence[SeqRow], summaries: Sequence[GoalSummary], top: int) -> None:
    print("observations", len(rows))
    print("domains", dict(Counter(row.domain for row in rows).most_common()))
    print("reported_goal_groups", len(summaries))
    print("decisions", dict(Counter(s.decision for s in summaries).most_common()))
    print("")
    print("TOP_PROGRESS")
    progress = [s for s in summaries if s.decision == "progress_memory_candidate"]
    progress.sort(key=lambda s: (s.confidence, s.n), reverse=True)
    for s in progress[:top]:
        print(
            "{}\t{}\tn={}\tpass={}\tfail={}\tconf={:.2f}\tprefix={}".format(
                s.domain, s.goal_key, s.n, s.pass_n, s.fail_n, s.confidence, " > ".join(s.pass_prefix)
            )
        )
        if s.l1_missing:
            print("  missing_l1", [(g.feature, round(g.gap, 2), round(g.pass_rate, 2), round(g.fail_rate, 2)) for g in s.l1_missing[:4]])
        if s.stall_gaps:
            print("  stalls", [(g.feature, round(g.gap, 2), round(g.pass_rate, 2), round(g.fail_rate, 2)) for g in s.stall_gaps[:4]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="task_subgoal_sequences.tsv files or directories to scan.")
    parser.add_argument("--tsv", action="append", default=[], help="Explicit task_subgoal_sequences.tsv file or directory.")
    parser.add_argument("--goal-table", action="append", default=[], help="Optional task_goal_table.tsv file or directory.")
    parser.add_argument("--out-dir", default="", help="Write summary artifacts here.")
    parser.add_argument("--min-goal-size", type=int, default=3)
    parser.add_argument("--min-gap", type=float, default=0.35)
    parser.add_argument("--min-stall-gap", type=float, default=0.25)
    parser.add_argument("--prefix-support", type=float, default=0.5)
    parser.add_argument("--report-gap", type=float, default=0.20)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    input_paths = args.paths + args.tsv
    if not input_paths:
        input_paths = ["."]
    tsv_files = collect_tsv_files(input_paths, "task_subgoal_sequences.tsv")
    goal_tables = collect_tsv_files(args.goal_table, "task_goal_table.tsv") + sibling_goal_tables(tsv_files)

    rows = read_sequences(tsv_files)
    task_info = read_task_info(goal_tables)
    summaries = summarize_all(
        rows,
        task_info,
        min_goal_size=args.min_goal_size,
        min_gap=args.min_gap,
        min_stall_gap=args.min_stall_gap,
        prefix_support=args.prefix_support,
    )
    print_summary(rows, summaries, args.top)

    if args.out_dir:
        ensure_dir(args.out_dir)
        write_tsv(
            os.path.join(args.out_dir, "goal_domain_matrix.tsv"),
            [
                "domain", "goal_key", "n", "pass_n", "fail_n", "unknown_n", "pass_rate",
                "decision", "confidence", "best_missing_l1", "best_missing_l1_gap",
                "best_fail_stall", "best_fail_stall_gap", "pass_prefix",
                "pass_prefix_support", "notes",
            ],
            matrix_rows(summaries),
        )
        write_tsv(
            os.path.join(args.out_dir, "subgoal_gap_table.tsv"),
            [
                "domain", "goal_key", "level", "feature", "direction", "pass_rate",
                "fail_rate", "gap", "n", "pass_n", "fail_n", "decision",
            ],
            gap_rows(summaries, args.report_gap),
        )
        write_jsonl(os.path.join(args.out_dir, "goal_candidates.jsonl"), candidate_rows(summaries, args.top * 2))
        report = build_report(rows, tsv_files, summaries, args.top)
        with open(os.path.join(args.out_dir, "tsv_summary_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
