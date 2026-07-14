"""
Read-only feasibility probe for stall-triggered online procedure memory.

The existing divergence probe asks whether a failing trajectory made a wrong
choice after an exact shared prefix. This probe asks the complementary question:
once post-boundary behavior repeats or visibly stops progressing, does another
successful task contain a state-compatible subgoal transition that could supply
an alternative route?

The probe deliberately reads raw executed trajectories and screenshots first.
It does not use v3 action signatures to define procedures, and it never mutates
the run directories.

For every failed episode it:
  1. Finds the first mutually exclusive stall trigger: a repeated normalized
     post-boundary error, an exact action repeat, or a static low-level select run.
  2. Excludes network blocks, infrastructure failures, and pre-boundary errors.
  3. Ranks state-changing transitions from other successful tasks by app, goal,
     current subgoal/history, instruction, and pre-state screenshot similarity.
  4. Reports two retrieval upper bounds:
       oracle -- all other successful tasks are available;
       online -- only successful tasks completed earlier in the same run root
                 (or across roots with --online-scope all) are available.

Low-level typing is intentionally outside the no-progress rule because thumbnail
similarity can miss productive text edits. Exact repeats may trigger with unknown
state evidence; low-level select runs require positive static-screen evidence.

Pillow (already declared by ComputerRL) is used for fast screenshot decoding. A
small deterministic stdlib PNG fallback keeps the probe usable in bare shells.

Usage:
  python analyze_stall_retrieval_probe.py \
    --run-root /path/to/computerrl_omni_rec2 \
    --out-dir /path/to/stall_retrieval_probe

Outputs:
  stall_queries.jsonl
  retrieval_candidates.jsonl
  stall_retrieval_summary.tsv
  stall_retrieval_report.md
"""

import argparse
import glob
import hashlib
import io
import json
import os
import re
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from analyze_goal_subgoal_gap import (
    OBJ_RULES,
    OP_RULES,
    TaskInfo,
    app_family,
    collapse_repeats,
    ensure_dir,
    extract_calls_from_text,
    extract_subgoals_from_event,
    first_float,
    is_error_text,
    load_api_descriptions,
    load_tasks,
    tags,
    write_jsonl,
    write_tsv,
)
from mm_agents.error_ledger import classify_error_step

try:
    from PIL import Image
except ImportError:  # The repository declares Pillow; bare shells use the fallback.
    Image = None


NETWORK_MARKERS = (
    "could not resolve",
    "temporary failure resolving",
    "network is unreachable",
    "connection timed out",
    "failed to connect",
    "unable to connect",
    "unable to access",
    "site can't be reached",
    "site cannot be reached",
    "connection reset",
    "name or service not known",
)

INFRA_MARKERS = (
    "modulenotfounderror",
    "no module named",
    "connection refused",
    "failed to import",
    "cannot import",
    "dapiuno.cxx",
)

TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "your",
    "you", "please", "could", "would", "help", "want", "need", "using",
    "current", "make", "set", "can", "all", "are", "have", "has", "its",
}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value or "")


def _compact(value: Any, limit: int = 600) -> str:
    text = " ".join(_text(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _canonical_domain(domain: str) -> str:
    return domain[len("fix_rewrite_"):] if domain.startswith("fix_rewrite_") else domain


def _strip_adapter_calls(text: str) -> str:
    text = re.sub(r"\b[A-Za-z_]\w*Tools\.print_result\s*\([^)]*\)\s*;?", "", text)
    return text.strip(" ;")


def _action_key(action: Any) -> str:
    # Keep arguments here: Agent.click([10, 20]) and Agent.click([30, 40]) are
    # different executed actions even though the role/subgoal mapper intentionally
    # reduces both to Agent.click.
    text = _strip_adapter_calls(_text(action))
    return re.sub(r"\s+", "", text).lower()


def _action_template(action: Any) -> str:
    text = _strip_adapter_calls(_compact(action))
    text = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "<str>", text)
    text = re.sub(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?", "<num>", text)
    return re.sub(r"\s+", "", text)


def _error_template(text: Any) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    picked = next(
        (line for line in reversed(lines) if re.search(r"(Error|Exception)\b", line, re.IGNORECASE)),
        lines[-1],
    )
    picked = re.sub(r"'[^']*'", "'X'", picked)
    picked = re.sub(r'"[^"]*"', '"X"', picked)
    picked = re.sub(r"0x[0-9a-fA-F]+", "0xN", picked)
    picked = re.sub(r"\b\d+\b", "N", picked)
    return re.sub(r"\s+", " ", picked).lower()[:240]


def _timestamp(raw: Any) -> Optional[float]:
    text = str(raw or "")
    m = re.search(r"(\d{8})@(\d{6})", text)
    if not m:
        return None
    # Lexicographic calendar order is sufficient and avoids timezone assumptions.
    return float(m.group(1) + m.group(2))


def _tokens(text: str) -> set:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOPWORDS}


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _suffix_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    matched = 0
    for size in range(1, limit + 1):
        if list(left[-size:]) == list(right[-size:]):
            matched = size
    return matched / float(limit)


def _pct(n: int, d: int) -> str:
    return "{:.1f}%".format(100.0 * n / d) if d else "-"


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _md(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


@dataclass
class TraceStep:
    ordinal: int
    step_num: int
    timestamp: Optional[float]
    response: Any
    action: Any
    exe_result: str
    done: bool
    screenshot_path: str
    action_key: str
    action_template: str
    call: str = ""
    l1: str = ""
    l2: str = ""
    error_class: str = ""
    error_template: str = ""


@dataclass
class Episode:
    episode_id: str
    root_id: str
    root_path: str
    task_dir: str
    domain: str
    canonical_domain: str
    task_id: str
    task: TaskInfo
    score: Optional[float]
    success: Optional[bool]
    result_path: str
    traj_path: str
    runtime_path: str
    runtime_text: str
    steps: List[TraceStep] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class StallEvent:
    query_id: str
    episode: Episode
    step_pos: int
    window_start: int
    stall_type: str
    trigger_rule: str
    state_evidence: str
    state_similarity: Optional[float]
    eligible: bool
    exclusion_reason: str

    @property
    def step(self) -> TraceStep:
        return self.episode.steps[self.step_pos]


@dataclass
class DonorTransition:
    episode: Episode
    step_pos: int
    pre_screenshot: str
    post_screenshot: str
    transition_similarity: float
    context_l2: List[str]
    continuation_l2: List[str]
    continuation_actions: List[str]

    @property
    def step(self) -> TraceStep:
        return self.episode.steps[self.step_pos]


@dataclass(frozen=True)
class ImageState:
    digest: str
    thumbnail: Tuple[int, ...]
    error: str = ""


class ImageFingerprinter:
    """Deterministic screenshot -> grayscale thumbnail cache."""

    def __init__(self, width: int = 32, height: int = 18):
        self.width = width
        self.height = height
        self.cache: Dict[str, ImageState] = {}
        self.stats = Counter()

    def state(self, path: str) -> ImageState:
        path = os.path.abspath(path) if path else ""
        if path in self.cache:
            return self.cache[path]
        if not path or not os.path.isfile(path):
            state = ImageState("", (), "missing")
            self.stats["missing"] += 1
        else:
            try:
                if Image is not None:
                    state = self._decode_pillow(path)
                    self.stats["pillow_decoded"] += int(bool(state.thumbnail))
                else:
                    state = self._decode_png(path)
                    self.stats["stdlib_decoded"] += int(bool(state.thumbnail))
                self.stats["decoded"] += int(bool(state.thumbnail))
                self.stats["unsupported"] += int(not state.thumbnail)
            except (OSError, ValueError, struct.error, zlib.error) as exc:
                state = ImageState("", (), type(exc).__name__)
                self.stats["decode_error"] += 1
        self.cache[path] = state
        return state

    def similarity(self, left: str, right: str) -> Optional[float]:
        a, b = self.state(left), self.state(right)
        if a.digest and a.digest == b.digest:
            return 1.0
        if not a.thumbnail or len(a.thumbnail) != len(b.thumbnail):
            return None
        diff = sum(abs(x - y) for x, y in zip(a.thumbnail, b.thumbnail))
        return max(0.0, 1.0 - diff / float(255 * len(a.thumbnail)))

    def _decode_pillow(self, path: str) -> ImageState:
        with open(path, "rb") as handle:
            blob = handle.read()
        digest = hashlib.sha1(blob).hexdigest()
        with Image.open(io.BytesIO(blob)) as source:
            gray = source.convert("L").resize(
                (self.width, self.height),
                resample=Image.Resampling.BILINEAR,
            )
            return ImageState(digest, tuple(gray.getdata()), "")

    @staticmethod
    def _paeth(a: int, b: int, c: int) -> int:
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        return b if pb <= pc else c

    def _decode_png(self, path: str) -> ImageState:
        with open(path, "rb") as handle:
            blob = handle.read()
        digest = hashlib.sha1(blob).hexdigest()
        if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
            return ImageState(digest, (), "not_png")

        pos = 8
        width = height = bit_depth = color_type = interlace = None
        palette = b""
        idat: List[bytes] = []
        while pos + 12 <= len(blob):
            length = struct.unpack(">I", blob[pos:pos + 4])[0]
            kind = blob[pos + 4:pos + 8]
            payload = blob[pos + 8:pos + 8 + length]
            pos += 12 + length
            if kind == b"IHDR":
                width, height, bit_depth, color_type, _comp, _flt, interlace = struct.unpack(
                    ">IIBBBBB", payload
                )
            elif kind == b"PLTE":
                palette = payload
            elif kind == b"IDAT":
                idat.append(payload)
            elif kind == b"IEND":
                break

        if not width or not height or bit_depth != 8 or interlace != 0 or not idat:
            return ImageState(digest, (), "unsupported_png")
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type)
        if channels is None or (color_type == 3 and not palette):
            return ImageState(digest, (), "unsupported_color")

        raw = zlib.decompress(b"".join(idat))
        stride = width * channels
        expected = height * (stride + 1)
        if len(raw) < expected:
            raise ValueError("truncated PNG data")

        x_samples = [min(width - 1, int((i + 0.5) * width / self.width)) for i in range(self.width)]
        y_targets: Dict[int, List[int]] = defaultdict(list)
        for out_y in range(self.height):
            src_y = min(height - 1, int((out_y + 0.5) * height / self.height))
            y_targets[src_y].append(out_y)
        thumb = [0] * (self.width * self.height)

        prev = bytearray(stride)
        offset = 0
        for src_y in range(height):
            filter_type = raw[offset]
            scan = raw[offset + 1:offset + 1 + stride]
            offset += stride + 1
            row = bytearray(stride)
            for i, value in enumerate(scan):
                left = row[i - channels] if i >= channels else 0
                up = prev[i]
                up_left = prev[i - channels] if i >= channels else 0
                if filter_type == 0:
                    recon = value
                elif filter_type == 1:
                    recon = value + left
                elif filter_type == 2:
                    recon = value + up
                elif filter_type == 3:
                    recon = value + ((left + up) // 2)
                elif filter_type == 4:
                    recon = value + self._paeth(left, up, up_left)
                else:
                    raise ValueError("unsupported PNG filter")
                row[i] = recon & 0xFF

            if src_y in y_targets:
                values: List[int] = []
                for src_x in x_samples:
                    i = src_x * channels
                    if color_type == 0:
                        gray, alpha = row[i], 255
                    elif color_type == 2:
                        r, g, b = row[i:i + 3]
                        gray, alpha = (77 * r + 150 * g + 29 * b) >> 8, 255
                    elif color_type == 3:
                        pi = row[i] * 3
                        r, g, b = palette[pi:pi + 3]
                        gray, alpha = (77 * r + 150 * g + 29 * b) >> 8, 255
                    elif color_type == 4:
                        gray, alpha = row[i], row[i + 1]
                    else:
                        r, g, b, alpha = row[i:i + 4]
                        gray = (77 * r + 150 * g + 29 * b) >> 8
                    values.append((gray * alpha + 255 * (255 - alpha)) // 255)
                for out_y in y_targets[src_y]:
                    start = out_y * self.width
                    thumb[start:start + self.width] = values
            prev = row

        return ImageState(digest, tuple(thumb), "")


def _fallback_task(task_id: str, domain: str, instruction: str = "") -> TaskInfo:
    canonical = _canonical_domain(domain)
    ops = tags(instruction, OP_RULES)
    objs = tags(instruction, OBJ_RULES)
    app = app_family(canonical)
    return TaskInfo(
        task_id=task_id,
        path="",
        domain=canonical,
        app=app,
        source="unknown",
        instruction=instruction,
        eval_func="unknown",
        ops=ops,
        objs=objs,
        goal_key="{}.{}.{}".format(app, ops[0], objs[0]),
        task_cluster_key="unknown",
        metadata_strength="missing_task_json",
    )


def _parse_score(path: str) -> Optional[float]:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return first_float(handle.read())
    except OSError:
        return None


def _runtime_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def _load_steps(traj_path: str, task_dir: str, api_desc: Dict[str, str]) -> List[TraceStep]:
    steps: List[TraceStep] = []
    try:
        handle = open(traj_path, encoding="utf-8", errors="replace")
    except OSError:
        return steps
    with handle:
        for line in handle:
            try:
                raw = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict) or not (raw.get("step_num") or raw.get("step_idx")):
                continue
            action = raw.get("action") or raw.get("action_text") or ""
            response = raw.get("response") or ""
            subgoals = extract_subgoals_from_event("", action, action, "", api_desc)
            if not subgoals:
                subgoals = extract_subgoals_from_event(response, action, action, "", api_desc)
            primary = subgoals[0] if subgoals else None
            calls = extract_calls_from_text(action, prefer_answer=False)
            call = primary.call if primary else (calls[0] if calls else "")
            exe_result = str(raw.get("exe_result") or "")
            executed_key = _action_key(action)
            if is_error_text(exe_result):
                # Preserve the project's action-interface boundary: malformed
                # dispatched code is representation noise even if traj.jsonl kept
                # its non-empty text; a valid concrete action marks other failures
                # as post-boundary execution feedback.
                action_sig = (call or "executed") if executed_key else "unknown"
                error_class = classify_error_step(exe_result, action_sig)
            else:
                error_class = ""
            screenshot_file = str(raw.get("screenshot_file") or "")
            screenshot_path = os.path.join(task_dir, screenshot_file) if screenshot_file else ""
            steps.append(
                TraceStep(
                    ordinal=len(steps),
                    step_num=int(raw.get("step_num") or raw.get("step_idx") or len(steps) + 1),
                    timestamp=_timestamp(raw.get("action_timestamp") or screenshot_file),
                    response=response,
                    action=action,
                    exe_result=exe_result,
                    done=bool(raw.get("done")),
                    screenshot_path=screenshot_path,
                    action_key=executed_key,
                    action_template=_action_template(action),
                    call=call,
                    l1=primary.l1 if primary else "",
                    l2=primary.l2 if primary else "",
                    error_class=error_class,
                    error_template=_error_template(exe_result) if error_class else "",
                )
            )
    return steps


def discover_episodes(args) -> List[Episode]:
    tasks = load_tasks(args.task_root or [
        "evaluation_examples/examples",
        "evaluation_examples/examples_office",
    ])
    api_desc = load_api_descriptions(args.schema_root or ["mm_agents/autoglm_v/tools/apis"])
    episodes: List[Episode] = []
    domain_filter = set(getattr(args, "domain", []) or [])
    canonical_filter = set(getattr(args, "canonical_domain", []) or [])
    exclude_rewrite = bool(getattr(args, "exclude_rewrite", False))

    for root_index, raw_root in enumerate(args.run_root):
        root = os.path.abspath(raw_root)
        root_id = "{}#{}".format(os.path.basename(os.path.normpath(root)) or "run", root_index + 1)
        for result_path in sorted(glob.glob(os.path.join(root, "**", "result.txt"), recursive=True)):
            task_dir = os.path.dirname(result_path)
            task_id = os.path.basename(task_dir)
            domain = os.path.basename(os.path.dirname(task_dir))
            canonical = _canonical_domain(domain)
            if exclude_rewrite and domain.startswith("fix_rewrite_"):
                continue
            if domain_filter and domain not in domain_filter:
                continue
            if canonical_filter and canonical not in canonical_filter:
                continue
            task = tasks.get(task_id) or _fallback_task(task_id, canonical)
            traj_path = os.path.join(task_dir, "traj.jsonl")
            runtime_path = os.path.join(task_dir, "runtime.log")
            score = _parse_score(result_path)
            steps = _load_steps(traj_path, task_dir, api_desc)
            times = [s.timestamp for s in steps if s.timestamp is not None]
            rel = os.path.relpath(task_dir, root)
            episodes.append(
                Episode(
                    episode_id=root_id + "/" + rel,
                    root_id=root_id,
                    root_path=root,
                    task_dir=task_dir,
                    domain=domain,
                    canonical_domain=canonical,
                    task_id=task_id,
                    task=task,
                    score=score,
                    success=None if score is None else score > args.success_threshold,
                    result_path=result_path,
                    traj_path=traj_path,
                    runtime_path=runtime_path,
                    runtime_text=_runtime_text(runtime_path),
                    steps=steps,
                    start_time=min(times) if times else None,
                    end_time=max(times) if times else None,
                )
            )
    return sorted(episodes, key=lambda e: (e.root_id, e.start_time is None, e.start_time or 0, e.episode_id))


def _has_marker(text: str, markers: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in markers)


def _window_text(window: Sequence[TraceStep]) -> str:
    return "\n".join(s.exe_result for s in window)


def _context_l2(steps: Sequence[TraceStep], end_pos: int, limit: int) -> List[str]:
    seq = [s.l2 for s in steps[: end_pos + 1] if s.l2]
    return collapse_repeats(seq)[-limit:]


def _window_state_similarity(
    window: Sequence[TraceStep],
    images: ImageFingerprinter,
) -> Optional[float]:
    sims = [
        images.similarity(window[i - 1].screenshot_path, window[i].screenshot_path)
        for i in range(1, len(window))
    ]
    if not sims or any(sim is None for sim in sims):
        return None
    return min(sims)


def detect_first_stall(episode: Episode, images: ImageFingerprinter, args) -> Optional[StallEvent]:
    if episode.success is not False:
        return None
    for end in range(len(episode.steps)):
        match: Optional[Tuple[int, str, str, Optional[float], str]] = None

        if end + 1 >= args.error_repeat_threshold:
            start = end - args.error_repeat_threshold + 1
            window = episode.steps[start:end + 1]
            templates = [step.error_template for step in window]
            if templates[0] and len(set(templates)) == 1:
                classes = {step.error_class for step in window}
                if classes == {"execution"}:
                    match = (start, "execution", "error_repeat", None, "unknown")
                elif classes & {"representation", "no_action"}:
                    match = (start, "", "error_repeat", None, "unknown")

        if match is None and end + 1 >= args.exact_repeat_threshold:
            start = end - args.exact_repeat_threshold + 1
            window = episode.steps[start:end + 1]
            if not any(step.error_class for step in window):
                keys = [step.action_key for step in window]
                if keys[0] and len(set(keys)) == 1:
                    state_similarity = _window_state_similarity(window, images)
                    if (
                        state_similarity is None
                        or state_similarity >= args.no_progress_threshold
                    ):
                        state_evidence = "unknown" if state_similarity is None else "static"
                        match = (
                            start, "no_progress", "exact_repeat",
                            state_similarity, state_evidence,
                        )

        if match is None and end + 1 >= args.lowlevel_run_threshold:
            start = end - args.lowlevel_run_threshold + 1
            window = episode.steps[start:end + 1]
            if not any(step.error_class for step in window):
                keys = [step.action_key for step in window]
                if (
                    all(keys)
                    and len(set(keys)) >= 2
                    and all(step.l1 == "lowlevel_select" for step in window)
                ):
                    state_similarity = _window_state_similarity(window, images)
                    if (
                        state_similarity is not None
                        and state_similarity >= args.no_progress_threshold
                    ):
                        match = (
                            start, "no_progress", "lowlevel_static",
                            state_similarity, "static",
                        )

        if match is None:
            continue
        start, stall_type, trigger_rule, state_similarity, state_evidence = match
        window = episode.steps[start:end + 1]

        query_id = "{}@{}".format(episode.episode_id, window[-1].ordinal)
        text = _window_text(window)
        classes = {s.error_class for s in window if s.error_class}
        if classes & {"representation", "no_action"}:
            exclusion = "pre_boundary_error"
        elif _has_marker(text, NETWORK_MARKERS) or _has_marker(episode.runtime_text, NETWORK_MARKERS):
            exclusion = "network_block"
        elif _has_marker(text, INFRA_MARKERS) or _has_marker(episode.runtime_text, INFRA_MARKERS):
            exclusion = "infrastructure_failure"
        elif not images.state(window[-1].screenshot_path).thumbnail:
            exclusion = "missing_query_state"
        else:
            exclusion = ""
        return StallEvent(
            query_id=query_id,
            episode=episode,
            step_pos=end,
            window_start=start,
            stall_type="" if exclusion else stall_type,
            trigger_rule=trigger_rule,
            state_evidence=state_evidence,
            state_similarity=state_similarity,
            eligible=not exclusion,
            exclusion_reason=exclusion,
        )
    return None


def build_donors(episodes: Sequence[Episode], images: ImageFingerprinter, args) -> List[DonorTransition]:
    donors: List[DonorTransition] = []
    for episode in episodes:
        if episode.success is not True:
            continue
        for pos in range(1, len(episode.steps)):
            before, step = episode.steps[pos - 1], episode.steps[pos]
            if not step.action_key or step.error_class or _has_marker(step.exe_result, NETWORK_MARKERS + INFRA_MARKERS):
                continue
            similarity = images.similarity(before.screenshot_path, step.screenshot_path)
            if similarity is None or similarity >= args.no_progress_threshold:
                continue
            continuation = episode.steps[pos:pos + args.continuation_steps]
            donors.append(
                DonorTransition(
                    episode=episode,
                    step_pos=pos,
                    pre_screenshot=before.screenshot_path,
                    post_screenshot=step.screenshot_path,
                    transition_similarity=similarity,
                    context_l2=_context_l2(episode.steps, pos, args.context_length),
                    continuation_l2=[s.l2 for s in continuation if s.l2],
                    continuation_actions=[s.action_template for s in continuation if s.action_template],
                )
            )
    return donors


def _goal_similarity(query: TaskInfo, donor: TaskInfo) -> float:
    if query.goal_key and query.goal_key == donor.goal_key and "unknown" not in query.goal_key:
        return 1.0
    op = _jaccard(query.ops, donor.ops)
    obj = _jaccard(query.objs, donor.objs)
    return 0.5 * op + 0.5 * obj


def _subgoal_similarity(query: TraceStep, donor: TraceStep) -> float:
    if query.l2 and query.l2 == donor.l2:
        return 1.0
    if query.l1 and query.l1 == donor.l1:
        return 0.5
    return 0.0


def _online_available(query: Episode, donor: Episode, scope: str) -> Optional[bool]:
    if query.start_time is None or donor.end_time is None:
        return None
    if scope == "root" and query.root_id != donor.root_id:
        return False
    return donor.end_time < query.start_time


def rank_candidates(
    query: StallEvent,
    donors: Sequence[DonorTransition],
    images: ImageFingerprinter,
    args,
) -> List[dict]:
    query_step = query.step
    query_context = _context_l2(query.episode.steps, query.step_pos, args.context_length)
    out: List[dict] = []
    for donor in donors:
        if donor.episode.task_id == query.episode.task_id:
            continue
        if donor.episode.task.app != query.episode.task.app:
            continue
        goal_sim = _goal_similarity(query.episode.task, donor.episode.task)
        instruction_sim = _jaccard(
            _tokens(query.episode.task.instruction), _tokens(donor.episode.task.instruction)
        )
        subgoal_sim = _subgoal_similarity(query_step, donor.step)
        context_sim = _suffix_similarity(query_context, donor.context_l2)
        novel = donor.step.action_key != query_step.action_key
        novel_template = donor.step.action_template != query_step.action_template
        task_ok = goal_sim >= args.min_goal_similarity or instruction_sim >= args.min_instruction_similarity
        subgoal_ok = subgoal_sim >= 0.5 or context_sim >= args.min_context_similarity
        if not (novel and task_ok and subgoal_ok):
            continue
        state_sim = images.similarity(query_step.screenshot_path, donor.pre_screenshot)
        if state_sim is None:
            continue
        state_ok = state_sim >= args.candidate_state_threshold
        compatible = bool(state_ok)
        score = (
            0.35 * state_sim
            + 0.25 * goal_sim
            + 0.20 * context_sim
            + 0.10 * subgoal_sim
            + 0.10 * instruction_sim
        )
        score = max(0.0, min(1.0, score))
        out.append({
            "query_id": query.query_id,
            "score": round(score, 6),
            "compatible": compatible,
            "state_compatible": state_ok,
            "novel_action": novel,
            "novel_template": novel_template,
            "state_similarity": round(state_sim, 6),
            "goal_similarity": round(goal_sim, 6),
            "instruction_similarity": round(instruction_sim, 6),
            "subgoal_similarity": round(subgoal_sim, 6),
            "context_similarity": round(context_sim, 6),
            "online_available": _online_available(query.episode, donor.episode, args.online_scope),
            "donor": donor,
        })
    out.sort(key=lambda row: (
        -row["score"],
        row["donor"].episode.domain.startswith("fix_rewrite_"),
        row["donor"].episode.episode_id,
        row["donor"].step.ordinal,
    ))
    return out


def _mode_summary(ranked: Sequence[dict], top_k: int) -> dict:
    # One successful task contributes at most one candidate. This prevents
    # normal/fix_rewrite twins or several adjacent transitions from inflating
    # top-k coverage and agreement.
    unique: List[dict] = []
    seen_tasks = set()
    for row in ranked:
        task_id = row["donor"].episode.task_id
        if task_id in seen_tasks:
            continue
        seen_tasks.add(task_id)
        unique.append(row)
    top = unique[:top_k]
    compatible = [row for row in top if row["compatible"]]
    all_compatible = [row for row in unique if row["compatible"]]
    template_compatible = [row for row in compatible if row["novel_template"]]
    all_template_compatible = [row for row in all_compatible if row["novel_template"]]
    l2 = Counter(row["donor"].step.l2 or "unknown" for row in compatible)
    agreement = (l2.most_common(1)[0][1] / float(len(compatible))) if compatible else 0.0
    return {
        "candidate_count": len(unique),
        "compatible_count": len(all_compatible),
        "covered_at_1": bool(top and top[0]["compatible"]),
        "covered_at_k": bool(compatible),
        "covered_any": bool(all_compatible),
        "template_covered_at_1": bool(
            top and top[0]["compatible"] and top[0]["novel_template"]
        ),
        "template_covered_at_k": bool(template_compatible),
        "template_covered_any": bool(all_template_compatible),
        "support": len({row["donor"].episode.task_id for row in compatible}),
        "agreement": agreement,
        "top": top,
    }


def analyze(episodes: Sequence[Episode], images: ImageFingerprinter, args):
    stalls = [event for episode in episodes if (event := detect_first_stall(episode, images, args))]
    donors = build_donors(episodes, images, args)
    query_rows: List[dict] = []
    candidate_rows: List[dict] = []

    for event in stalls:
        ranked = rank_candidates(event, donors, images, args) if event.eligible else []
        oracle = _mode_summary(ranked, args.top_k)
        online_ranked = [row for row in ranked if row["online_available"] is True]
        online = _mode_summary(online_ranked, args.top_k)

        for mode, summary in (("oracle", oracle), ("online", online)):
            for rank, row in enumerate(summary["top"], 1):
                donor: DonorTransition = row["donor"]
                candidate_rows.append({
                    "query_id": event.query_id,
                    "mode": mode,
                    "rank": rank,
                    "score": row["score"],
                    "compatible": int(row["compatible"]),
                    "state_compatible": int(row["state_compatible"]),
                    "novel_action": int(row["novel_action"]),
                    "novel_template": int(row["novel_template"]),
                    "state_similarity": row["state_similarity"],
                    "goal_similarity": row["goal_similarity"],
                    "instruction_similarity": row["instruction_similarity"],
                    "subgoal_similarity": row["subgoal_similarity"],
                    "context_similarity": row["context_similarity"],
                    "online_available": row["online_available"],
                    "donor_episode_id": donor.episode.episode_id,
                    "donor_task_id": donor.episode.task_id,
                    "donor_domain": donor.episode.domain,
                    "donor_goal_key": donor.episode.task.goal_key,
                    "donor_step_num": donor.step.step_num,
                    "donor_l1": donor.step.l1,
                    "donor_l2": donor.step.l2,
                    "donor_action": _compact(donor.step.action),
                    "donor_action_template": donor.step.action_template,
                    "donor_pre_screenshot": donor.pre_screenshot,
                    "donor_post_screenshot": donor.post_screenshot,
                    "donor_state_delta": round(1.0 - donor.transition_similarity, 6),
                    "continuation_l2": donor.continuation_l2,
                    "continuation_actions": donor.continuation_actions,
                })

        query_rows.append({
            "query_id": event.query_id,
            "episode_id": event.episode.episode_id,
            "run_id": event.episode.root_id,
            "domain": event.episode.domain,
            "canonical_domain": event.episode.canonical_domain,
            "app": event.episode.task.app,
            "task_id": event.episode.task_id,
            "goal_key": event.episode.task.goal_key,
            "step_num": event.step.step_num,
            "step_ordinal": event.step.ordinal,
            "window_start_ordinal": event.window_start,
            "stall_type": event.stall_type,
            "trigger_rule": event.trigger_rule,
            "state_evidence": event.state_evidence,
            "eligible": int(event.eligible),
            "exclusion_reason": event.exclusion_reason,
            "query_state_available": int(
                bool(images.state(event.step.screenshot_path).thumbnail)
            ),
            "stall_state_similarity": "" if event.state_similarity is None else round(event.state_similarity, 6),
            "stall_l1": event.step.l1,
            "stall_l2": event.step.l2,
            "stall_action": _compact(event.step.action),
            "stall_action_template": event.step.action_template,
            "stall_screenshot": event.step.screenshot_path,
            "online_order_known": int(event.episode.start_time is not None),
            "oracle_candidate_count": oracle["candidate_count"],
            "oracle_compatible_count": oracle["compatible_count"],
            "oracle_covered_at_1": int(oracle["covered_at_1"]),
            "oracle_covered_at_k": int(oracle["covered_at_k"]),
            "oracle_covered_any": int(oracle["covered_any"]),
            "oracle_template_covered_at_1": int(oracle["template_covered_at_1"]),
            "oracle_template_covered_at_k": int(oracle["template_covered_at_k"]),
            "oracle_template_covered_any": int(oracle["template_covered_any"]),
            "oracle_support": oracle["support"],
            "oracle_agreement": round(oracle["agreement"], 6),
            "online_candidate_count": online["candidate_count"],
            "online_compatible_count": online["compatible_count"],
            "online_covered_at_1": int(online["covered_at_1"]),
            "online_covered_at_k": int(online["covered_at_k"]),
            "online_covered_any": int(online["covered_any"]),
            "online_template_covered_at_1": int(online["template_covered_at_1"]),
            "online_template_covered_at_k": int(online["template_covered_at_k"]),
            "online_template_covered_any": int(online["template_covered_any"]),
            "online_support": online["support"],
            "online_agreement": round(online["agreement"], 6),
        })
    return stalls, donors, query_rows, candidate_rows


def build_state_similarity_null(
    stalls: Sequence[StallEvent],
    donors: Sequence[DonorTransition],
    images: ImageFingerprinter,
    args,
) -> Dict[str, dict]:
    """Calibrate raw screenshot similarity against same-app cross-task states.

    Each pair combines an eligible stall with one successful pre-state from a
    different task in the same app. Stable hashes select at most one transition
    per donor task and cap work per app, so reruns are deterministic. The p95
    sensitivity threshold affects the decision gate only after enough null pairs
    are available.
    """
    queries_by_app: Dict[str, List[StallEvent]] = defaultdict(list)
    donors_by_app: Dict[str, List[DonorTransition]] = defaultdict(list)
    for event in stalls:
        if event.eligible:
            queries_by_app[event.episode.task.app].append(event)
    for donor in donors:
        donors_by_app[donor.episode.task.app].append(donor)

    summaries: Dict[str, dict] = {}
    for app, queries in sorted(queries_by_app.items()):
        per_query = max(1, (args.null_pairs_per_app + len(queries) - 1) // len(queries))
        sampled: List[Tuple[bytes, float]] = []
        for event in queries:
            best_by_task: Dict[str, Tuple[bytes, DonorTransition]] = {}
            for donor in donors_by_app.get(app, []):
                if donor.episode.task_id == event.episode.task_id:
                    continue
                pair_key = "{}\0{}\0{}".format(
                    event.query_id, donor.episode.episode_id, donor.step.ordinal
                ).encode("utf-8", errors="replace")
                stable_key = hashlib.sha1(pair_key).digest()
                previous = best_by_task.get(donor.episode.task_id)
                if previous is None or stable_key < previous[0]:
                    best_by_task[donor.episode.task_id] = (stable_key, donor)

            for stable_key, donor in sorted(best_by_task.values(), key=lambda item: item[0])[:per_query]:
                similarity = images.similarity(event.step.screenshot_path, donor.pre_screenshot)
                if similarity is not None:
                    sampled.append((stable_key, similarity))

        sampled.sort(key=lambda item: item[0])
        values = [similarity for _key, similarity in sampled[:args.null_pairs_per_app]]
        p95 = _quantile(values, 0.95)
        gate_applied = len(values) >= args.min_null_pairs_for_gate and p95 is not None
        summaries[app] = {
            "queries": len(queries),
            "pairs": len(values),
            "p50": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "p95": p95,
            "threshold_rate": (
                sum(value >= args.candidate_state_threshold for value in values) / float(len(values))
                if values else None
            ),
            "gate_applied": gate_applied,
            "calibrated_threshold": (
                max(args.candidate_state_threshold, p95) if gate_applied else None
            ),
        }
    return summaries


def apply_null_calibration(
    query_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    state_null: Dict[str, dict],
    args,
) -> None:
    """Attach null-p95 sensitivity coverage to each query row in place."""
    candidates: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for candidate in candidate_rows:
        candidates[(candidate["query_id"], candidate["mode"])].append(candidate)

    for row in query_rows:
        stats = state_null.get(row["app"], {})
        applied = bool(stats.get("gate_applied"))
        threshold = (
            stats["calibrated_threshold"] if applied else args.candidate_state_threshold
        )
        threshold = round(threshold, 6)
        row["null_calibration_pairs"] = int(stats.get("pairs", 0))
        row["null_calibration_applied"] = int(applied)
        row["null_calibrated_threshold"] = threshold
        for mode in ("oracle", "online"):
            covered = any(
                candidate["novel_template"]
                and candidate["state_similarity"] >= threshold
                for candidate in candidates.get((row["query_id"], mode), [])
            )
            row["{}_null_calibrated_template_covered_at_k".format(mode)] = int(covered)


def _coverage(rows: Sequence[dict], field: str, online_only: bool = False) -> Tuple[int, int]:
    eligible = [row for row in rows if row["eligible"]]
    if online_only:
        eligible = [row for row in eligible if row["online_order_known"]]
    return sum(int(row[field]) for row in eligible), len(eligible)


def build_report(
    episodes: Sequence[Episode],
    stalls: Sequence[StallEvent],
    donors: Sequence[DonorTransition],
    query_rows: Sequence[dict],
    state_null: Dict[str, dict],
    images: ImageFingerprinter,
    args,
) -> str:
    lines: List[str] = ["# Stall Retrieval Feasibility Probe", ""]
    passes = sum(e.success is True for e in episodes)
    fails = sum(e.success is False for e in episodes)
    unknown = len(episodes) - passes - fails
    eligible = [row for row in query_rows if row["eligible"]]
    excluded = [row for row in query_rows if not row["eligible"]]
    exclusions = Counter(row["exclusion_reason"] or "unknown" for row in excluded)
    order_known = sum(row["online_order_known"] for row in eligible)
    state_static = sum(row["state_evidence"] == "static" for row in eligible)
    state_unknown = sum(row["state_evidence"] == "unknown" for row in eligible)
    episode_domains: Dict[Tuple[str, str], set] = defaultdict(set)
    for episode in episodes:
        episode_domains[(episode.root_id, episode.task_id)].add(episode.domain)
    mixed_variant_tasks = sum(1 for domains in episode_domains.values() if len(domains) > 1)

    lines.extend([
        "## Corpus",
        "",
        "- episodes: {} (pass {}, fail {}, unknown {})".format(len(episodes), passes, fails, unknown),
        "- domains: {}".format(", ".join(sorted({e.domain for e in episodes}))),
        "- successful state-changing donor transitions: {} from {} unique tasks".format(
            len(donors), len({d.episode.task_id for d in donors})
        ),
        "- first stall trigger candidates: {} (eligible {}, excluded {})".format(
            len(stalls), len(eligible), len(excluded)
        ),
        "- eligible stall evidence: static {}, unknown {}".format(
            state_static, state_unknown
        ),
        "- eligible stalls with known online order: {}/{}".format(order_known, len(eligible)),
        "- task IDs present in multiple domain variants: {}".format(mixed_variant_tasks),
        "- image decode: {}".format(dict(images.stats)),
        "",
        "This is a retrieval-feasibility proxy, not counterfactual proof that replay would fix a task.",
        "Donor transitions begin at the second logged trajectory row because traj.jsonl has no initial pre-action screenshot; first-action entry routes are unobservable.",
        "",
        "## Stall Definition",
        "",
        "- execution-error repeat: {} consecutive identical normalized errors".format(
            args.error_repeat_threshold
        ),
        "- exact repeat: {} error-free identical action keys; visible change vetoes the trigger, missing state is allowed".format(
            args.exact_repeat_threshold
        ),
        "- low-level no progress: {} error-free lowlevel_select actions with at least two distinct keys and complete static evidence".format(
            args.lowlevel_run_threshold
        ),
        "- static evidence requires every adjacent screenshot similarity >= {:.4f}".format(
            args.no_progress_threshold
        ),
        "- representation/no_action, network, and infrastructure failures are routed out before retrieval",
        "- lowlevel_input is not a no-progress trigger; the same {:.4f} cutoff defines donor state change".format(
            args.no_progress_threshold
        ),
        "- candidate pre-state similarity: >= {:.4f}".format(args.candidate_state_threshold),
        "- online scope: `{}`".format(args.online_scope),
        "",
    ])

    trigger_rows: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in query_rows:
        trigger_rows[(row["stall_type"] or "routed_out", row["trigger_rule"])].append(row)
    lines.append(
        "| stall type | trigger rule | detected | eligible | static | unknown | "
        "oracle novel-template @{} | online novel-template @{} |".format(args.top_k, args.top_k)
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (stall_type, trigger_rule), rows in sorted(
        trigger_rows.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        trigger_eligible = [row for row in rows if row["eligible"]]
        trigger_online = [row for row in trigger_eligible if row["online_order_known"]]
        static_n = sum(row["state_evidence"] == "static" for row in trigger_eligible)
        unknown_n = sum(row["state_evidence"] == "unknown" for row in trigger_eligible)
        oracle_template_n = sum(row["oracle_template_covered_at_k"] for row in trigger_eligible)
        online_template_n = sum(row["online_template_covered_at_k"] for row in trigger_online)
        lines.append("| {} | {} | {} | {} | {} | {} | {}/{} ({}) | {}/{} ({}) |".format(
            _md(stall_type), _md(trigger_rule), len(rows), len(trigger_eligible),
            static_n, unknown_n,
            oracle_template_n, len(trigger_eligible), _pct(oracle_template_n, len(trigger_eligible)),
            online_template_n, len(trigger_online), _pct(online_template_n, len(trigger_online)),
        ))
    if exclusions:
        lines.extend(["", "Excluded: " + ", ".join("{}={}".format(k, v) for k, v in exclusions.most_common())])
    lines.append("")

    lines.extend([
        "## State Similarity Null",
        "",
        "Deterministic same-app, cross-task stall/donor pairs before goal or subgoal filtering. "
        "Apps with at least {} null pairs use max(base threshold, null p95) as a decision-gate sensitivity check.".format(
            args.min_null_pairs_for_gate
        ),
        "A high null pass rate means raw retrieval coverage may reflect shared app chrome rather than reusable task state.",
        "",
        "| app | eligible queries | sampled pairs | p50 | p90 | p95 | null >= {:.4f} | gate threshold | gate applied |".format(
            args.candidate_state_threshold
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    if not state_null:
        lines.append("| - | 0 | 0 | - | - | - | - | - | no |")
    for app, stats in sorted(state_null.items()):
        fmt = lambda value: "-" if value is None else "{:.4f}".format(value)
        rate = "-" if stats["threshold_rate"] is None else "{:.1f}%".format(100 * stats["threshold_rate"])
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _md(app), stats["queries"], stats["pairs"], fmt(stats["p50"]),
            fmt(stats["p90"]), fmt(stats["p95"]), rate,
            fmt(stats["calibrated_threshold"]), "yes" if stats["gate_applied"] else "no",
        ))
    lines.append("")

    oracle_1 = _coverage(query_rows, "oracle_covered_at_1")
    oracle_k = _coverage(query_rows, "oracle_covered_at_k")
    oracle_any = _coverage(query_rows, "oracle_covered_any")
    oracle_template_k = _coverage(query_rows, "oracle_template_covered_at_k")
    oracle_null_template_k = _coverage(
        query_rows, "oracle_null_calibrated_template_covered_at_k"
    )
    online_1 = _coverage(query_rows, "online_covered_at_1", online_only=True)
    online_k = _coverage(query_rows, "online_covered_at_k", online_only=True)
    online_any = _coverage(query_rows, "online_covered_any", online_only=True)
    online_template_k = _coverage(query_rows, "online_template_covered_at_k", online_only=True)
    online_null_template_k = _coverage(
        query_rows, "online_null_calibrated_template_covered_at_k", online_only=True
    )
    lines.extend([
        "## Retrieval Coverage",
        "",
        "| availability | compatible @1 | compatible @{} | novel-template @{} | null-p95 novel-template @{} | compatible anywhere |".format(
            args.top_k, args.top_k, args.top_k
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| oracle | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) |".format(
            oracle_1[0], oracle_1[1], _pct(*oracle_1),
            oracle_k[0], oracle_k[1], _pct(*oracle_k),
            oracle_template_k[0], oracle_template_k[1], _pct(*oracle_template_k),
            oracle_null_template_k[0], oracle_null_template_k[1], _pct(*oracle_null_template_k),
            oracle_any[0], oracle_any[1], _pct(*oracle_any),
        ),
        "| online | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) |".format(
            online_1[0], online_1[1], _pct(*online_1),
            online_k[0], online_k[1], _pct(*online_k),
            online_template_k[0], online_template_k[1], _pct(*online_template_k),
            online_null_template_k[0], online_null_template_k[1], _pct(*online_null_template_k),
            online_any[0], online_any[1], _pct(*online_any),
        ),
        "",
    ])

    lines.extend([
        "### By Domain", "",
        "| domain | eligible stalls | oracle @{} | oracle novel-template | oracle null-p95 | online @{} | online novel-template | online null-p95 |".format(
            args.top_k, args.top_k
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    by_domain: Dict[str, List[dict]] = defaultdict(list)
    for row in eligible:
        by_domain[row["domain"]].append(row)
    for domain in sorted(by_domain):
        rows = by_domain[domain]
        known = [r for r in rows if r["online_order_known"]]
        oc = sum(r["oracle_covered_at_k"] for r in rows)
        oct_ = sum(r["oracle_template_covered_at_k"] for r in rows)
        ocn = sum(r["oracle_null_calibrated_template_covered_at_k"] for r in rows)
        on = sum(r["online_covered_at_k"] for r in known)
        ont = sum(r["online_template_covered_at_k"] for r in known)
        onn = sum(r["online_null_calibrated_template_covered_at_k"] for r in known)
        lines.append("| {} | {} | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) | {}/{} ({}) |".format(
            domain, len(rows), oc, len(rows), _pct(oc, len(rows)),
            oct_, len(rows), _pct(oct_, len(rows)),
            ocn, len(rows), _pct(ocn, len(rows)),
            on, len(known), _pct(on, len(known)),
            ont, len(known), _pct(ont, len(known)),
            onn, len(known), _pct(onn, len(known)),
        ))
    lines.append("")

    null_gate_applied = any(
        row["eligible"] and row["null_calibration_applied"] for row in query_rows
    )
    raw_template_ready = bool(
        oracle_template_k[1]
        and oracle_template_k[0] / float(oracle_template_k[1]) >= args.min_coverage
    )
    calibrated_template_ready = bool(
        oracle_null_template_k[1]
        and oracle_null_template_k[0] / float(oracle_null_template_k[1]) >= args.min_coverage
    )
    if mixed_variant_tasks and not (args.domain or args.exclude_rewrite):
        verdict = "MIXED_VARIANTS_RERUN_SEPARATELY"
        explanation = "normal and fix_rewrite episodes share task IDs and are not independent queries"
    elif len(eligible) < args.min_queries:
        verdict = "INSUFFICIENT_SAMPLE"
        explanation = "too few eligible mutually exclusive stall triggers for a decision"
    elif null_gate_applied and raw_template_ready and not calibrated_template_ready:
        verdict = "STATE_SIGNAL_NULL_SATURATED"
        explanation = "raw template-novel coverage does not survive the same-app null-p95 state threshold"
    elif calibrated_template_ready:
        if not online_null_template_k[1]:
            verdict = "PROMISING_BUT_ONLINE_ORDER_IS_UNKNOWN"
            explanation = "the corpus contains null-calibrated template-novel experience, but trajectory timestamps are missing"
        elif (
            online_null_template_k[0] / float(online_null_template_k[1])
            >= args.min_coverage
        ):
            verdict = "READY_FOR_SMALL_ONLINE_ABLATION"
            explanation = "null-calibrated alternative action templates are available online often enough"
        else:
            verdict = "PROMISING_BUT_ONLINE_BANK_IS_COLD"
            explanation = "null-calibrated template-novel experience exists, but run order limits online availability"
    elif oracle_k[1] and oracle_k[0] / float(oracle_k[1]) >= args.min_coverage:
        verdict = "INSUFFICIENT_PROCEDURE_NOVELTY"
        explanation = "compatible donors exist, but too few offer a different action template"
    else:
        verdict = "NOT_SUPPORTED_AT_CURRENT_RETRIEVAL_GRANULARITY"
        explanation = "too few eligible stalls have a compatible cross-task successful transition"
    lines.extend([
        "## Decision Gate",
        "",
        "**{}**: {}.".format(verdict, explanation),
        "",
        "Gate used here: at least {} eligible queries and oracle novel-template@{} >= {:.0f}%, "
        "using null-p95 state thresholds for apps with at least {} calibration pairs.".format(
            args.min_queries, args.top_k, 100 * args.min_coverage,
            args.min_null_pairs_for_gate,
        ),
        "",
    ])

    covered = [row for row in eligible if row["oracle_covered_at_k"]]
    covered.sort(key=lambda row: (-row["oracle_support"], row["query_id"]))
    lines.extend(["## Top Covered Stalls", ""])
    if not covered:
        lines.append("None.")
    else:
        lines.append("| domain | task | goal | stall type | trigger | step | support | novel template | null-p95 | online | online null-p95 |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in covered[:args.top]:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _md(row["domain"]), _md(row["task_id"]), _md(row["goal_key"]),
                _md(row["stall_type"]), _md(row["trigger_rule"]),
                row["step_num"], row["oracle_support"],
                "yes" if row["oracle_template_covered_at_k"] else "no",
                "yes" if row["oracle_null_calibrated_template_covered_at_k"] else "no",
                "yes" if row["online_covered_at_k"] else "no",
                "yes" if row["online_null_calibrated_template_covered_at_k"] else "no",
            ))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", action="append", required=True, help="Recovery result root; repeatable.")
    parser.add_argument("--task-root", action="append", default=[])
    parser.add_argument("--schema-root", action="append", default=[])
    parser.add_argument("--domain", action="append", default=[], help="Exact result-folder domain; repeatable.")
    parser.add_argument("--canonical-domain", action="append", default=[],
                        help="Canonical domain filter (also matches fix_rewrite variants); repeatable.")
    parser.add_argument("--exclude-rewrite", action="store_true",
                        help="Skip result folders whose names start with fix_rewrite_.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--success-threshold", type=float, default=0.0)
    parser.add_argument("--error-repeat-threshold", type=int, default=2,
                        help="Consecutive identical normalized execution errors required for a loop.")
    parser.add_argument("--exact-repeat-threshold", type=int, default=2,
                        help="Consecutive identical action keys required for exact-repeat.")
    parser.add_argument("--lowlevel-run-threshold", type=int, default=3,
                        help="Consecutive lowlevel_select actions required for lowlevel-static.")
    parser.add_argument("--no-progress-threshold", type=float, default=0.995,
                        help="Static-screen evidence and donor state-change cutoff.")
    parser.add_argument("--candidate-state-threshold", type=float, default=0.90)
    parser.add_argument("--min-goal-similarity", type=float, default=0.50)
    parser.add_argument("--min-instruction-similarity", type=float, default=0.15)
    parser.add_argument("--min-context-similarity", type=float, default=0.50)
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--continuation-steps", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--online-scope", choices=["root", "all"], default="root")
    parser.add_argument("--min-coverage", type=float, default=0.20)
    parser.add_argument("--min-queries", type=int, default=5)
    parser.add_argument("--thumbnail-width", type=int, default=32)
    parser.add_argument("--thumbnail-height", type=int, default=18)
    parser.add_argument("--null-pairs-per-app", type=int, default=500,
                        help="Deterministic cross-task screenshot pairs sampled per app for calibration.")
    parser.add_argument("--min-null-pairs-for-gate", type=int, default=20,
                        help="Minimum same-app null pairs required before p95 affects the decision gate.")
    args = parser.parse_args()

    positive = {
        "error_repeat_threshold": args.error_repeat_threshold,
        "exact_repeat_threshold": args.exact_repeat_threshold,
        "lowlevel_run_threshold": args.lowlevel_run_threshold,
        "context_length": args.context_length,
        "continuation_steps": args.continuation_steps,
        "top_k": args.top_k,
        "thumbnail_width": args.thumbnail_width,
        "thumbnail_height": args.thumbnail_height,
        "null_pairs_per_app": args.null_pairs_per_app,
        "min_null_pairs_for_gate": args.min_null_pairs_for_gate,
    }
    for name, value in positive.items():
        minimum = 2 if name in {
            "error_repeat_threshold", "exact_repeat_threshold", "lowlevel_run_threshold"
        } else 1
        if value < minimum:
            parser.error("--{} must be >= {}".format(name.replace("_", "-"), minimum))
    for name in (
        "no_progress_threshold", "candidate_state_threshold", "min_goal_similarity",
        "min_instruction_similarity", "min_context_similarity", "min_coverage",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error("--{} must be in [0, 1]".format(name.replace("_", "-")))

    episodes = discover_episodes(args)
    images = ImageFingerprinter(args.thumbnail_width, args.thumbnail_height)
    stalls, donors, query_rows, candidate_rows = analyze(episodes, images, args)
    state_null = build_state_similarity_null(stalls, donors, images, args)
    apply_null_calibration(query_rows, candidate_rows, state_null, args)
    report = build_report(episodes, stalls, donors, query_rows, state_null, images, args)
    print(report)

    if args.out_dir:
        ensure_dir(args.out_dir)
        write_jsonl(os.path.join(args.out_dir, "stall_queries.jsonl"), query_rows)
        write_jsonl(os.path.join(args.out_dir, "retrieval_candidates.jsonl"), candidate_rows)
        fields = [
            "query_id", "episode_id", "run_id", "domain", "canonical_domain", "app", "task_id",
            "goal_key", "step_num", "step_ordinal", "stall_type", "trigger_rule",
            "state_evidence", "eligible", "exclusion_reason", "query_state_available",
            "stall_state_similarity", "stall_l1", "stall_l2", "stall_action_template",
            "stall_screenshot", "online_order_known", "oracle_candidate_count", "oracle_compatible_count",
            "oracle_covered_at_1", "oracle_covered_at_k", "oracle_covered_any", "oracle_support",
            "oracle_template_covered_at_1", "oracle_template_covered_at_k",
            "oracle_template_covered_any", "oracle_agreement",
            "online_candidate_count", "online_compatible_count",
            "online_covered_at_1", "online_covered_at_k", "online_covered_any", "online_support",
            "online_template_covered_at_1", "online_template_covered_at_k",
            "online_template_covered_any", "online_agreement",
            "null_calibration_pairs", "null_calibration_applied", "null_calibrated_threshold",
            "oracle_null_calibrated_template_covered_at_k",
            "online_null_calibrated_template_covered_at_k",
        ]
        write_tsv(os.path.join(args.out_dir, "stall_retrieval_summary.tsv"), fields, query_rows)
        with open(os.path.join(args.out_dir, "stall_retrieval_report.md"), "w", encoding="utf-8") as handle:
            handle.write(report)
        print("\nwrote", args.out_dir)


if __name__ == "__main__":
    main()
