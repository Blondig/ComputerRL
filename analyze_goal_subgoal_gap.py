#!/usr/bin/env python3
"""
Goal/subgoal gap probe over ComputerRL tasks and run artifacts.

This is a read-only analysis script for deciding whether dataset task similarity
corresponds to reusable subgoal similarity. It deliberately does NOT use v3
ledger action_sig as the primary subgoal definition: raw response / trajectory /
runtime-log calls are parsed first, and v3 action_sig is only a marked fallback.

Typical remote usage:

  python analyze_goal_subgoal_gap.py \
      --task-root evaluation_examples/examples \
      --task-root evaluation_examples/examples_office \
      --run-root /path/to/results_or_run \
      --out-dir /tmp/goal_subgoal_probe

Useful extras:

  --baseline-root /path/to/baseline_results
  --audit /path/to/ledger.v3.audit.jsonl
  --log /path/to/runtime.log
  --min-goal-size 4
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Generic helpers


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:#%+-]{2,}")
HIGH_CALL_RE = re.compile(r"\b((?:[A-Za-z_]\w*Tools|Agent)\.\w+)\s*\(")
PY_AUTO_RE = re.compile(r"\b(pyautogui\.\w+)\s*\(")
FENCE_RE = re.compile(r"```(?:\w+\s+)?(.*?)```", re.DOTALL)

ADAPTER_METHODS = {"print_result"}
TERMINAL_CALLS = {"DONE", "FAIL", "WAIT", "Agent.exit", "Agent.wait"}
LOWLEVEL_L1 = {"lowlevel_select", "lowlevel_input"}
CORE_L1 = {"create", "edit", "transform", "format", "configure", "media"}


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def safe_json_load(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def iter_jsonl(path: str) -> Iterable[Tuple[int, dict]]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    yield lineno, data
    except OSError:
        return


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def first_float(text: str) -> Optional[float]:
    if text is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def content_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "\n".join(content_text(x) for x in obj)
    if isinstance(obj, dict):
        if "text" in obj:
            return content_text(obj["text"])
        if "content" in obj:
            return content_text(obj["content"])
        return "\n".join(content_text(v) for v in obj.values())
    text = str(obj)
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return text
        parsed_text = content_text(parsed)
        return parsed_text or text
    return text


def last_answer_code(response: Any) -> str:
    """Extract the code the model most likely submitted, not its free-form plan."""
    text = content_text(response)
    if not text:
        return ""
    tail = text.rsplit("<answer>", 1)[-1]
    blocks = FENCE_RE.findall(tail)
    if blocks:
        return blocks[-1]
    blocks = FENCE_RE.findall(text)
    if blocks:
        return blocks[-1]
    return tail


def extract_calls_from_text(text: Any, prefer_answer: bool = False) -> List[str]:
    raw = last_answer_code(text) if prefer_answer else content_text(text)
    calls = HIGH_CALL_RE.findall(raw or "")
    # Grounded action strings often contain only pyautogui calls.
    calls.extend(PY_AUTO_RE.findall(raw or ""))
    out = []
    for call in calls:
        if call.split(".")[-1] in ADAPTER_METHODS:
            continue
        out.append(call)
    return list(dict.fromkeys(out))


def normalize_task_id(task_id: Any, fallback: str = "") -> str:
    text = str(task_id or fallback or "").strip()
    if not text:
        return ""
    return os.path.splitext(os.path.basename(text))[0]


def parent_task_id(path: str) -> str:
    return normalize_task_id(os.path.basename(os.path.dirname(path)))


def relpath(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def under_root(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def infer_run_label(path: str, roots: Sequence[str]) -> str:
    apath = os.path.abspath(path)
    best = ""
    for root in roots:
        if not root:
            continue
        aroot = os.path.abspath(root)
        if apath == aroot or apath.startswith(aroot + os.sep):
            best = os.path.basename(os.path.normpath(aroot)) or "run"
            break
    if best:
        return best
    parent = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return parent or stem or "run"


def pct(num: float, den: float) -> str:
    if den <= 0:
        return "-"
    return "{:.0f}%".format(num / den * 100.0)


# ---------------------------------------------------------------------------
# Task / goal metadata


OP_RULES = [
    ("formula_lookup", r"\b(formula|vba|macro|match|lookup|compare cells?|if\(|xlookup|vlookup|index|sumif|countif)\b"),
    ("aggregate_chart", r"\b(total|sum|average|count|pivot|chart|graph|annual changes|percentage|monthly)\b"),
    ("fill_populate", r"\b(fill|populate|write|type|enter|input|set value|update|copy .* to|blank cells?)\b"),
    ("transform_data", r"\b(transpose|split|merge|round|convert|extract|clean|remove duplicates?|sort|filter|reorder columns?)\b"),
    ("insert_create", r"\b(add|insert|create|new|duplicate|blank)\b"),
    ("delete_remove", r"\b(delete|remove|clear|hide)\b"),
    ("move_reorder", r"\b(move|reorder|rearrange|relocate|switch it with)\b"),
    ("format_style", r"\b(font|bold|italic|underline|color|highlight|align|style|spacing|border|format|background|theme|size|orientation|strikethrough|capitalize|case|width|height|zoom)\b"),
    ("configure_setting", r"\b(setting|settings|preference|configure|enable|disable|install|extension|default|autosave|sync|warning|safe browsing|password)\b"),
    ("search_web", r"\b(search|find|look up|lookup|navigate|website|url|browser|chrome|google|flight|forecast|eligibility)\b"),
    ("save_export", r"\b(save|export|download|print|pdf|csv|png|jpg|jpeg|gif|mp4|rename|compress|zip)\b"),
    ("email_ops", r"\b(email|mail|reply|forward|send|inbox|message|attachment|contacts|thunderbird)\b"),
    ("media_ops", r"\b(play|pause|video|audio|vlc|playlist|fullscreen|recording|snapshot|scene)\b"),
    ("code_project", r"\b(code|python|script|vscode|vs code|project|terminal|json|markdown|merge conflict|extension)\b"),
    ("file_ops", r"\b(file|folder|directory|desktop|home path|move files|copy files|attachments)\b"),
]

OBJ_RULES = [
    ("spreadsheet_range", r"\b(spreadsheet|excel|xlsx|workbook|worksheet|sheet|cell|cells|range|row|column|formula|[A-Z]+\d+)\b"),
    ("table", r"\b(table|pivot|rows|columns|tabular)\b"),
    ("slide", r"\b(slide|slides|ppt|pptx|presentation|powerpoint)\b"),
    ("page", r"\b(page|pages)\b"),
    ("text_title", r"\b(text|textbox|title|paragraph|word|font|footnote|endnote|comment|highlighted words?)\b"),
    ("document", r"\b(docx|document|writer|word)\b"),
    ("image", r"\b(image|picture|png|jpg|jpeg|logo|gimp|pixel|crop|brightness|contrast|saturation|avatar)\b"),
    ("media", r"\b(video|audio|mp4|gif|vlc|playlist|media)\b"),
    ("browser_state", r"\b(browser|chrome|website|web|url|tab|bookmark|history|password|privacy|profile|extension)\b"),
    ("settings", r"\b(setting|settings|preference|extension|plugin|sync|theme|volume|notifications)\b"),
    ("file_artifact", r"\b(file|folder|directory|desktop|home|pdf|csv|zip|attachment|path)\b"),
    ("email", r"\b(email|mail|thunderbird|attachment|contact|inbox|message)\b"),
    ("code", r"\b(code|python|script|vscode|project|json|terminal|extension)\b"),
]


def tags(text: str, rules: Sequence[Tuple[str, str]]) -> List[str]:
    s = (text or "").lower()
    found = [name for name, pat in rules if re.search(pat, s)]
    return found or ["other"]


def as_evals(evaluator: Any) -> List[dict]:
    if isinstance(evaluator, dict):
        return [evaluator]
    if isinstance(evaluator, list):
        return [x for x in evaluator if isinstance(x, dict)]
    return []


def eval_func_sig(evaluator: Any) -> str:
    vals = []
    for ev in as_evals(evaluator):
        f = ev.get("func")
        if isinstance(f, list):
            vals.extend(str(x) for x in f)
        elif f:
            vals.append(str(f))
    return "+".join(sorted(set(vals))) or "none"


def evaluator_options(evaluator: Any) -> dict:
    out: Dict[str, Any] = {}
    for ev in as_evals(evaluator):
        opt = ev.get("options")
        if isinstance(opt, dict):
            out.update(opt)
    return out


def answer_shape(answer_position: Any) -> str:
    if not answer_position:
        return "none"
    text = str(answer_position)
    if "!" in text and ":" in text:
        return "sheet_range"
    if "!" in text:
        return "sheet_cell"
    if ":" in text:
        return "range"
    if re.match(r"^[A-Z]+\d+$", text):
        return "cell"
    return "named_or_other"


def app_family(domain: str) -> str:
    table = {
        "ow-xlsx": "spreadsheet",
        "libreoffice_calc": "spreadsheet",
        "ow-pptx": "presentation",
        "libreoffice_impress": "presentation",
        "ow-docx": "document",
        "libreoffice_writer": "document",
        "chrome": "browser",
        "google_chrome": "browser",
        "vs_code": "code",
        "vlc": "media",
        "gimp": "image",
        "thunderbird": "email",
        "os": "os",
        "multi_apps": "multi_app",
    }
    return table.get(domain, domain or "unknown")


def source_family(path: str, data: dict) -> str:
    if "{}examples_office{}".format(os.sep, os.sep) in path:
        return "officeworld"
    return str(data.get("source") or data.get("snapshot") or "osworld")


def metadata_strength(task: "TaskInfo") -> str:
    if task.source != "officeworld":
        return "mixed_source_struct"
    if task.domain == "ow-xlsx" and task.instruction_type:
        return "officeworld_partial_struct"
    return "officeworld_text_weak"


@dataclass
class TaskInfo:
    task_id: str
    path: str
    domain: str
    app: str
    source: str
    instruction: str
    eval_func: str
    instruction_type: str = ""
    answer_shape: str = "none"
    ops: List[str] = field(default_factory=list)
    objs: List[str] = field(default_factory=list)
    goal_key: str = ""
    task_cluster_key: str = ""
    metadata_strength: str = ""


def load_tasks(task_roots: Sequence[str]) -> Dict[str, TaskInfo]:
    tasks: Dict[str, TaskInfo] = {}
    for root in task_roots:
        for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            data = safe_json_load(path)
            if not isinstance(data, dict) or not data.get("instruction"):
                continue
            task_id = normalize_task_id(data.get("id"), os.path.basename(path))
            domain = os.path.basename(os.path.dirname(path))
            instruction = " ".join(str(data.get("instruction", "")).split())
            opt = evaluator_options(data.get("evaluator"))
            ops = tags(instruction, OP_RULES)
            objs = tags(instruction, OBJ_RULES)
            task = TaskInfo(
                task_id=task_id,
                path=path,
                domain=domain,
                app=app_family(domain),
                source=source_family(path, data),
                instruction=instruction,
                eval_func=eval_func_sig(data.get("evaluator")),
                instruction_type=str(opt.get("instruction_type", "")),
                answer_shape=answer_shape(opt.get("answer_position")),
                ops=ops,
                objs=objs,
            )
            task.goal_key = "{}.{}.{}".format(task.app, ops[0], objs[0])
            tail = task.instruction_type or task.answer_shape or "-"
            task.task_cluster_key = "|".join([task.source, task.domain, task.eval_func, ops[0], objs[0], tail])
            task.metadata_strength = metadata_strength(task)
            tasks[task_id] = task
    return tasks


# ---------------------------------------------------------------------------
# API schema and call -> subgoal mapping


def load_api_descriptions(schema_roots: Sequence[str]) -> Dict[str, str]:
    desc: Dict[str, str] = {}
    for root in schema_roots:
        for path in glob.glob(os.path.join(root, "*.json")):
            data = safe_json_load(path)
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                fn = item.get("function") or {}
                name = str(fn.get("name") or "")
                if name:
                    desc[name] = str(fn.get("description") or "")
    return desc


@dataclass(frozen=True)
class Subgoal:
    call: str
    l1: str
    l2: str
    source: str
    is_fallback: bool = False


def method_name(call: str) -> str:
    return (call or "").split(".")[-1].lower()


def tool_name(call: str) -> str:
    return (call or "").split(".")[0]


def is_adapter_or_terminal(call: str) -> bool:
    if not call:
        return True
    if call in TERMINAL_CALLS:
        return True
    return method_name(call) in ADAPTER_METHODS


def l1_for_call(call: str, desc: str = "") -> str:
    if not call:
        return "unknown"
    tool = tool_name(call)
    method = method_name(call)
    text = (method + " " + (desc or "")).lower()

    if is_adapter_or_terminal(call):
        return "terminal" if method in {"exit", "wait"} or call in {"DONE", "FAIL", "WAIT"} else "adapter"
    if tool == "pyautogui":
        if method in {"click", "doubleclick", "rightclick", "moveto", "dragto", "mousedown", "mouseup", "scroll"}:
            return "lowlevel_select"
        if method in {"write", "press", "hotkey", "keydown", "keyup", "type"}:
            return "lowlevel_input"
        return "lowlevel"
    if tool == "Agent":
        if method in {"click", "drag_and_drop", "scroll"}:
            return "lowlevel_select"
        if method in {"type", "hotkey"}:
            return "lowlevel_input"
        if method in {"open_app"}:
            return "setup"
        if method in {"switch_window"}:
            return "locate"
        if method in {"quote"}:
            return "inspect"

    if method.startswith("save") or method.startswith("export") or method in {"print"}:
        return "commit"
    if any(k in text for k in ("get_", " get ", "list", "read", "check", "count", "describe", "find highlighted", "current time", "duration", "info")):
        return "inspect"
    if any(method.startswith(p) for p in ("go_to", "goto", "switch", "open", "launch", "scroll", "focus", "activate", "navigate", "bring_back")):
        return "locate"
    if any(k in text for k in ("install", "uninstall", "disable", "enable", "toggle", "configure", "settings", "setting", "validation", "fullscreen", "sync")):
        return "configure"
    if any(k in text for k in ("font", "color", "align", "style", "spacing", "border", "background", "highlight", "orientation", "format", "width", "height", "zoom", "pane", "strikethrough")):
        return "format"
    if any(k in text for k in ("transpose", "pivot", "chart", "formula", "sort", "merge", "split", "convert", "extract", "clean", "round", "calculate")):
        return "transform"
    if any(method.startswith(p) for p in ("insert", "create", "duplicate", "copy", "add", "new")):
        return "create"
    if any(method.startswith(p) for p in ("set", "write", "fill", "type", "replace", "delete", "remove", "clear", "hide", "rename", "move", "reorder", "update")):
        return "edit"
    if tool == "VLCTools" and method in {"play", "pause", "next", "previous"}:
        return "media"
    return "other"


def l2_for_call(call: str, l1: str, desc: str = "") -> str:
    tool = tool_name(call)
    method = method_name(call)
    text = (method + " " + (desc or "")).lower()

    if tool == "CalcTools":
        if "workbook" in text:
            obj = "workbook"
        elif "sheet" in text:
            obj = "sheet"
        elif any(k in text for k in ("column", "row", "cell", "range", "validation", "number")):
            obj = "range"
        elif "chart" in text:
            obj = "chart"
        elif "pivot" in text:
            obj = "pivot"
        else:
            obj = "spreadsheet"
    elif tool == "ImpressTools":
        if "slide" in text:
            obj = "slide"
        elif any(k in text for k in ("text", "font", "box", "title", "alignment", "strikethrough")):
            obj = "textbox"
        elif any(k in text for k in ("image", "video", "audio", "file")):
            obj = "media"
        elif "display" in text:
            obj = "display"
        else:
            obj = "presentation"
    elif tool == "WriterTools":
        if any(k in text for k in ("font", "text", "paragraph", "highlight", "line", "case", "strikethrough")):
            obj = "text"
        elif "image" in text:
            obj = "image"
        elif "formula" in text:
            obj = "formula"
        elif "page" in text:
            obj = "page"
        else:
            obj = "document"
    elif tool == "BrowserTools":
        if "bookmark" in text:
            obj = "bookmark"
        elif "tab" in text:
            obj = "tab"
        elif "password" in text:
            obj = "password_settings"
        elif any(k in text for k in ("privacy", "profile", "appearance", "search", "settings", "data")):
            obj = "browser_settings"
        else:
            obj = "browser"
    elif tool == "CodeTools":
        if "extension" in text:
            obj = "extension"
        elif "folder" in text:
            obj = "folder"
        elif "file" in text:
            obj = "file"
        elif "merge" in text:
            obj = "merge"
        else:
            obj = "code"
    elif tool == "VLCTools":
        if "playlist" in text:
            obj = "playlist"
        elif "settings" in text:
            obj = "settings"
        elif any(k in text for k in ("play", "pause", "media", "time", "duration", "fullscreen")):
            obj = "playback"
        else:
            obj = "media"
    elif tool == "Agent":
        obj = "agent"
    elif tool == "pyautogui":
        obj = "gui"
    else:
        obj = tool.lower() or "unknown"

    op = l1
    if l1 in {"inspect", "locate", "setup", "commit", "configure", "format", "transform", "create", "edit", "media"}:
        op = l1
    return "{}.{}".format(obj, op)


def classify_call(call: str, source: str, api_desc: Dict[str, str], is_fallback: bool = False) -> Optional[Subgoal]:
    call = (call or "").strip()
    if not call or is_adapter_or_terminal(call):
        return None
    desc = api_desc.get(call, "")
    l1 = l1_for_call(call, desc)
    if l1 in {"adapter", "terminal", "unknown"}:
        return None
    return Subgoal(call=call, l1=l1, l2=l2_for_call(call, l1, desc), source=source, is_fallback=is_fallback)


def extract_subgoals_from_event(
    response: Any,
    raw_action: Any,
    action_text: Any,
    v3_action_sig: str,
    api_desc: Dict[str, str],
) -> List[Subgoal]:
    found: List[Tuple[str, str, bool]] = []
    for call in extract_calls_from_text(response, prefer_answer=True):
        found.append((call, "response", False))
    if not found:
        for call in extract_calls_from_text(action_text, prefer_answer=False):
            found.append((call, "action_text", False))
    if not found:
        for call in extract_calls_from_text(raw_action, prefer_answer=False):
            found.append((call, "traj_action", False))
    if not found and v3_action_sig:
        sig = str(v3_action_sig).strip()
        if sig and sig != "unknown":
            found.append((sig, "v3_fallback", True))

    out: List[Subgoal] = []
    seen = set()
    for call, source, fallback in found:
        sg = classify_call(call, source, api_desc, fallback)
        if sg is None:
            continue
        key = (sg.call, sg.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(sg)
    return out


# ---------------------------------------------------------------------------
# Artifact loading


@dataclass
class StepEvent:
    step_idx: int
    response: Any = ""
    raw_action: Any = ""
    action_text: Any = ""
    exe_result: str = ""
    is_error: bool = False
    done: bool = False
    source: str = ""
    v3_action_sig: str = ""
    subgoals: List[Subgoal] = field(default_factory=list)


@dataclass
class Observation:
    run_id: str
    task_id: str
    artifact_paths: List[str] = field(default_factory=list)
    steps: List[StepEvent] = field(default_factory=list)
    audit_success: Optional[bool] = None
    score: Optional[float] = None
    baseline_score: Optional[float] = None
    injected_count: int = 0
    log_signals: Counter = field(default_factory=Counter)


@dataclass
class ManifestEntry:
    kind: str
    path: str
    count: int = 0
    note: str = ""


ERROR_MARKERS = (
    "Error:",
    "error:",
    "Traceback",
    "Exception:",
    "TypeError",
    "ValueError",
    "AttributeError",
    "NameError",
    "SyntaxError",
    "RuntimeError",
    "FileNotFoundError",
    "Invalid action",
)


def is_error_text(text: Any) -> bool:
    s = str(text or "")
    return any(m in s for m in ERROR_MARKERS)


def collect_files(run_roots: Sequence[str], explicit: Sequence[str], patterns: Sequence[str]) -> List[str]:
    files = []
    for path in explicit:
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            for pat in patterns:
                files.extend(glob.glob(os.path.join(path, "**", pat), recursive=True))
    for root in run_roots:
        if not root:
            continue
        for pat in patterns:
            files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    return sorted(set(files))


def parse_scores(roots: Sequence[str]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "result.txt"), recursive=True):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    score = first_float(f.read())
            except OSError:
                score = None
            if score is None:
                continue
            out[(infer_run_label(path, roots), parent_task_id(path))] = score
    return out


def load_audit_observations(
    audit_files: Sequence[str],
    run_roots: Sequence[str],
    api_desc: Dict[str, str],
    manifest: List[ManifestEntry],
) -> Dict[Tuple[str, str], Observation]:
    obs: Dict[Tuple[str, str], Observation] = {}
    for path in audit_files:
        run_id = infer_run_label(path, run_roots)
        n = 0
        for _lineno, rec in iter_jsonl(path):
            task_id = normalize_task_id(rec.get("task_id"))
            if not task_id:
                continue
            key = (run_id, task_id)
            cur = obs.setdefault(key, Observation(run_id=run_id, task_id=task_id))
            cur.artifact_paths.append(path)
            cur.audit_success = bool(rec.get("success"))
            cur.injected_count += len(rec.get("injected") or [])
            for raw in rec.get("steps") or []:
                if not isinstance(raw, dict):
                    continue
                idx = int(raw.get("step_idx") or len(cur.steps) + 1)
                ev = StepEvent(
                    step_idx=idx,
                    response=raw.get("response") or "",
                    raw_action=raw.get("action") or "",
                    action_text=raw.get("action_text") or "",
                    exe_result=str(raw.get("exe_result") or ""),
                    is_error=bool(raw.get("is_error")) or is_error_text(raw.get("exe_result")),
                    done=bool(raw.get("done")),
                    source="audit",
                    v3_action_sig=str(raw.get("action_sig") or raw.get("api_call") or ""),
                )
                ev.subgoals = extract_subgoals_from_event(
                    ev.response, ev.raw_action, ev.action_text, ev.v3_action_sig, api_desc
                )
                cur.steps.append(ev)
            n += 1
        manifest.append(ManifestEntry(kind="audit", path=path, count=n))
    return obs


def merge_traj_observations(
    observations: Dict[Tuple[str, str], Observation],
    traj_files: Sequence[str],
    run_roots: Sequence[str],
    api_desc: Dict[str, str],
    manifest: List[ManifestEntry],
) -> None:
    for path in traj_files:
        run_id = infer_run_label(path, run_roots)
        task_id = parent_task_id(path)
        key = (run_id, task_id)
        cur = observations.setdefault(key, Observation(run_id=run_id, task_id=task_id))
        cur.artifact_paths.append(path)
        n = 0
        existing = {ev.step_idx: ev for ev in cur.steps}
        for _lineno, raw in iter_jsonl(path):
            idx = int(raw.get("step_num") or raw.get("step_idx") or n + 1)
            if idx in existing and existing[idx].subgoals:
                # Keep audit ordering but add trajectory path to the manifest.
                n += 1
                continue
            ev = StepEvent(
                step_idx=idx,
                response=raw.get("response") or "",
                raw_action=raw.get("action") or "",
                action_text=raw.get("action") or "",
                exe_result=str(raw.get("exe_result") or ""),
                is_error=is_error_text(raw.get("exe_result")),
                done=bool(raw.get("done")),
                source="traj",
                v3_action_sig="",
            )
            ev.subgoals = extract_subgoals_from_event(
                ev.response, ev.raw_action, ev.action_text, ev.v3_action_sig, api_desc
            )
            cur.steps.append(ev)
            existing[idx] = ev
            n += 1
        manifest.append(ManifestEntry(kind="traj", path=path, count=n))


LOG_EXAMPLE_RE = re.compile(r"\[Example ID\]:\s*([A-Za-z0-9_.:@/-]+)|Example ID\]\s*:?\s*([A-Za-z0-9_.:@/-]+)")
LOG_RESPONSE_RE = re.compile(r"RESPONSE(?:\([^)]*\))?:\s*(.*)")
LOG_PSEUDO_RE = re.compile(r"The pesudo action is\s+(.*)")
LOG_GROUNDED_RE = re.compile(r"The grounded action is\s+(.*)")
LOG_STEP_RE = re.compile(r"Step\s+(\d+):\s+(.*)")
LOG_RESULT_RE = re.compile(r"Result:\s*([-+]?\d+(?:\.\d+)?)")


def merge_runtime_logs(
    observations: Dict[Tuple[str, str], Observation],
    log_files: Sequence[str],
    run_roots: Sequence[str],
    api_desc: Dict[str, str],
    manifest: List[ManifestEntry],
) -> None:
    for path in log_files:
        run_id = infer_run_label(path, run_roots)
        current_task = ""
        current_step = 0
        log_events: Dict[str, List[StepEvent]] = defaultdict(list)
        signals: Dict[str, Counter] = defaultdict(Counter)
        count = 0
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    text = strip_ansi(line.rstrip("\n"))
                    m = LOG_EXAMPLE_RE.search(text)
                    if m:
                        current_task = normalize_task_id(m.group(1) or m.group(2))
                        current_step = 0
                        signals[current_task]["seen_in_log"] += 1
                        continue
                    if not current_task:
                        continue
                    if "ActionRepair" in text:
                        signals[current_task]["action_repair"] += 1
                    if "IntraRecovery:" in text:
                        signals[current_task]["intra_recovery_log"] += 1
                        if "'interface_ok': False" in text or '"interface_ok": False' in text:
                            signals[current_task]["interface_not_ok"] += 1
                    if "Failed to parse action from response" in text:
                        signals[current_task]["parse_failed"] += 1
                    mr = LOG_RESULT_RE.search(text)
                    if mr:
                        key = (run_id, current_task)
                        cur = observations.setdefault(key, Observation(run_id=run_id, task_id=current_task))
                        cur.score = first_float(mr.group(1))
                    ms = LOG_STEP_RE.search(text)
                    if ms:
                        current_step = int(ms.group(1))
                    response = LOG_RESPONSE_RE.search(text)
                    pseudo = LOG_PSEUDO_RE.search(text)
                    grounded = LOG_GROUNDED_RE.search(text)
                    if response or pseudo or grounded:
                        ev = StepEvent(
                            step_idx=current_step or (len(log_events[current_task]) + 1),
                            response=response.group(1) if response else "",
                            raw_action=pseudo.group(1) if pseudo else "",
                            action_text=grounded.group(1) if grounded else "",
                            source="runtime_log",
                        )
                        ev.subgoals = extract_subgoals_from_event(
                            ev.response, ev.raw_action, ev.action_text, "", api_desc
                        )
                        log_events[current_task].append(ev)
                        count += 1
        except OSError:
            manifest.append(ManifestEntry(kind="log", path=path, count=0, note="failed_to_open"))
            continue

        for task_id, sigs in signals.items():
            key = (run_id, task_id)
            cur = observations.setdefault(key, Observation(run_id=run_id, task_id=task_id))
            cur.artifact_paths.append(path)
            cur.log_signals.update(sigs)
        for task_id, events in log_events.items():
            key = (run_id, task_id)
            cur = observations.setdefault(key, Observation(run_id=run_id, task_id=task_id))
            cur.artifact_paths.append(path)
            # Runtime logs are supplemental. Add them to the positive sequence only
            # when no audit/traj steps exist; otherwise keep them as evidence counts
            # to avoid double-counting the same model response.
            if not cur.steps:
                cur.steps.extend(events)
            cur.log_signals["runtime_events"] += len(events)
            cur.log_signals["runtime_subgoals"] += sum(1 for ev in events if ev.subgoals)
        manifest.append(ManifestEntry(kind="log", path=path, count=count))


def load_v3_bank(bank_files: Sequence[str], manifest: List[ManifestEntry]) -> Dict[str, dict]:
    cards: Dict[str, dict] = {}
    for path in bank_files:
        data = safe_json_load(path)
        n = 0
        if isinstance(data, dict):
            for section in ("error_notes", "success_snippets", "action_stats"):
                vals = data.get(section) or {}
                if isinstance(vals, dict):
                    for key, value in vals.items():
                        cards[str(key)] = value if isinstance(value, dict) else {"value": value}
                        n += 1
        manifest.append(ManifestEntry(kind="v3_bank", path=path, count=n, note="attribution_only"))
    return cards


# ---------------------------------------------------------------------------
# Sequence and gap analysis


def observation_success(obs: Observation) -> Optional[bool]:
    if obs.score is not None:
        return obs.score > 0.0
    return obs.audit_success


def paired_label(base_score: Optional[float], target_score: Optional[float], success: Optional[bool]) -> str:
    tp = success if success is not None else (target_score is not None and target_score > 0.0)
    if base_score is None:
        return "tgt_pass" if tp else "tgt_fail"
    bp = base_score > 0.0
    if bp and tp:
        return "both_pass"
    if (not bp) and (not tp):
        return "both_fail"
    return "target_fixed" if tp else "target_broke"


def ordered_steps(obs: Observation) -> List[StepEvent]:
    return sorted(obs.steps, key=lambda ev: ev.step_idx)


def subgoal_sequence(obs: Observation, include_lowlevel: bool = True) -> Tuple[List[str], List[str], Counter]:
    l1: List[str] = []
    l2: List[str] = []
    sources = Counter()
    for ev in ordered_steps(obs):
        for sg in ev.subgoals:
            sources[sg.source] += 1
            if sg.is_fallback:
                sources["fallback"] += 1
            if sg.l1 in {"adapter", "terminal", "unknown", "other"}:
                continue
            if not include_lowlevel and sg.l1 in LOWLEVEL_L1:
                continue
            l1.append(sg.l1)
            l2.append(sg.l2)
    return l1, l2, sources


def collapse_repeats(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def max_run(flags: Sequence[bool]) -> int:
    best = run = 0
    for flag in flags:
        if flag:
            run += 1
        else:
            run = 0
        best = max(best, run)
    return best


def stall_signals(obs: Observation) -> Counter:
    roles, _l2, _src = subgoal_sequence(obs, include_lowlevel=True)
    steps = ordered_steps(obs)
    sig = Counter()
    sig["n_steps"] = len(steps)
    sig["n_roles"] = len(roles)
    sig["n_errors"] = sum(1 for ev in steps if ev.is_error)
    sig["lowlevel_steps"] = sum(1 for r in roles if r in LOWLEVEL_L1)
    sig["core_steps"] = sum(1 for r in roles if r in CORE_L1)
    sig["commit_steps"] = sum(1 for r in roles if r == "commit")
    sig["fallback_steps"] = sum(1 for ev in steps for sg in ev.subgoals if sg.is_fallback)
    sig["raw_steps_without_subgoal"] = sum(1 for ev in steps if not ev.subgoals)
    if max_run([ev.is_error for ev in steps]) >= 3:
        sig["error_loop"] = 1
    if max_run([r in LOWLEVEL_L1 for r in roles]) >= 3:
        sig["lowlevel_loop"] = 1
    if roles and "commit" in roles:
        first_commit = roles.index("commit")
        if not any(r in CORE_L1 for r in roles[:first_commit]):
            sig["premature_commit"] = 1
    if roles and not any(r in CORE_L1 for r in roles):
        sig["no_core_action"] = 1
    sig.update(obs.log_signals)
    return sig


def task_for_obs(obs: Observation, tasks: Dict[str, TaskInfo]) -> TaskInfo:
    task = tasks.get(obs.task_id)
    if task:
        return task
    # Fallback goal from first recorded instruction.
    instruction = ""
    for ev in ordered_steps(obs):
        if isinstance(ev.response, str):
            # Audit step usually stores instruction separately outside StepEvent,
            # so this is only a last resort.
            pass
    ops = tags(instruction, OP_RULES)
    objs = tags(instruction, OBJ_RULES)
    return TaskInfo(
        task_id=obs.task_id,
        path="",
        domain="unknown",
        app="unknown",
        source="unknown",
        instruction=instruction,
        eval_func="unknown",
        ops=ops,
        objs=objs,
        goal_key="unknown.{}.{}".format(ops[0], objs[0]),
        task_cluster_key="unknown",
        metadata_strength="missing_task_json",
    )


@dataclass
class ObsRow:
    obs: Observation
    task: TaskInfo
    success: Optional[bool]
    label: str
    l1: List[str]
    l2: List[str]
    l1_collapsed: List[str]
    l2_collapsed: List[str]
    source_counts: Counter
    stall: Counter


def build_rows(observations: Dict[Tuple[str, str], Observation], tasks: Dict[str, TaskInfo]) -> List[ObsRow]:
    rows: List[ObsRow] = []
    for obs in observations.values():
        task = task_for_obs(obs, tasks)
        success = observation_success(obs)
        l1, l2, sources = subgoal_sequence(obs, include_lowlevel=True)
        rows.append(
            ObsRow(
                obs=obs,
                task=task,
                success=success,
                label=paired_label(obs.baseline_score, obs.score, success),
                l1=l1,
                l2=l2,
                l1_collapsed=collapse_repeats(l1),
                l2_collapsed=collapse_repeats(l2),
                source_counts=sources,
                stall=stall_signals(obs),
            )
        )
    return rows


def rate(rows: Sequence[ObsRow], pred) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if pred(row)) / len(rows)


def summarize_goal(rows: Sequence[ObsRow]) -> Dict[str, Any]:
    pass_rows = [r for r in rows if r.success is True]
    fail_rows = [r for r in rows if r.success is False]
    all_l1 = sorted(set(x for r in rows for x in set(r.l1_collapsed)))
    all_l2 = sorted(set(x for r in rows for x in set(r.l2_collapsed)))

    l1_gap = []
    for role in all_l1:
        pr = rate(pass_rows, lambda r, role=role: role in set(r.l1_collapsed))
        fr = rate(fail_rows, lambda r, role=role: role in set(r.l1_collapsed))
        if not math.isnan(pr) and not math.isnan(fr):
            l1_gap.append((pr - fr, role, pr, fr))
    l2_gap = []
    for role in all_l2:
        pr = rate(pass_rows, lambda r, role=role: role in set(r.l2_collapsed))
        fr = rate(fail_rows, lambda r, role=role: role in set(r.l2_collapsed))
        if not math.isnan(pr) and not math.isnan(fr):
            l2_gap.append((pr - fr, role, pr, fr))

    stall_names = sorted(set(k for r in rows for k in r.stall if k not in {"n_steps", "n_roles"}))
    stall_gap = []
    for name in stall_names:
        pr = rate(pass_rows, lambda r, name=name: r.stall.get(name, 0) > 0)
        fr = rate(fail_rows, lambda r, name=name: r.stall.get(name, 0) > 0)
        if not math.isnan(pr) and not math.isnan(fr):
            stall_gap.append((fr - pr, name, pr, fr))

    common_pass_l1 = Counter()
    common_fail_l1 = Counter()
    for row in pass_rows:
        common_pass_l1.update(set(row.l1_collapsed))
    for row in fail_rows:
        common_fail_l1.update(set(row.l1_collapsed))

    return {
        "n": len(rows),
        "pass_n": len(pass_rows),
        "fail_n": len(fail_rows),
        "sources": Counter(r.task.source for r in rows),
        "domains": Counter(r.task.domain for r in rows),
        "strengths": Counter(r.task.metadata_strength for r in rows),
        "l1_gap": sorted(l1_gap, reverse=True),
        "l2_gap": sorted(l2_gap, reverse=True),
        "stall_gap": sorted(stall_gap, reverse=True),
        "pass_l1": common_pass_l1,
        "fail_l1": common_fail_l1,
        "fallback_obs": sum(1 for r in rows if r.source_counts.get("fallback", 0) > 0),
    }


def candidate_from_summary(goal_key: str, summary: Dict[str, Any], min_pass: int, min_fail: int) -> Optional[dict]:
    if summary["pass_n"] < min_pass or summary["fail_n"] < min_fail:
        return None
    missing_l1 = [
        {"role": role, "pass_rate": pr, "fail_rate": fr, "gap": gap}
        for gap, role, pr, fr in summary["l1_gap"]
        if gap >= 0.35 and pr >= 0.5
    ][:5]
    fail_stalls = [
        {"signal": name, "pass_rate": pr, "fail_rate": fr, "gap": gap}
        for gap, name, pr, fr in summary["stall_gap"]
        if gap >= 0.25 and fr >= 0.4
    ][:5]
    if not missing_l1 and not fail_stalls:
        return None
    return {
        "goal_key": goal_key,
        "n": summary["n"],
        "pass_n": summary["pass_n"],
        "fail_n": summary["fail_n"],
        "sources": dict(summary["sources"].most_common()),
        "domains": dict(summary["domains"].most_common()),
        "missing_l1_in_fail": missing_l1,
        "fail_stall_signals": fail_stalls,
        "fallback_obs": summary["fallback_obs"],
    }


# ---------------------------------------------------------------------------
# Output


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_tsv(path: str, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def task_goal_rows(tasks: Dict[str, TaskInfo]) -> Iterable[dict]:
    for task in sorted(tasks.values(), key=lambda t: (t.domain, t.task_id)):
        yield {
            "task_id": task.task_id,
            "domain": task.domain,
            "app": task.app,
            "source": task.source,
            "eval_func": task.eval_func,
            "instruction_type": task.instruction_type,
            "answer_shape": task.answer_shape,
            "goal_key": task.goal_key,
            "task_cluster_key": task.task_cluster_key,
            "metadata_strength": task.metadata_strength,
            "ops": ",".join(task.ops),
            "objs": ",".join(task.objs),
            "instruction": task.instruction,
            "path": relpath(task.path),
        }


def subgoal_rows(rows: Sequence[ObsRow]) -> Iterable[dict]:
    for row in sorted(rows, key=lambda r: (r.obs.run_id, r.obs.task_id)):
        yield {
            "run_id": row.obs.run_id,
            "task_id": row.obs.task_id,
            "domain": row.task.domain,
            "app": row.task.app,
            "source": row.task.source,
            "goal_key": row.task.goal_key,
            "metadata_strength": row.task.metadata_strength,
            "score": "" if row.obs.score is None else row.obs.score,
            "baseline_score": "" if row.obs.baseline_score is None else row.obs.baseline_score,
            "success": "" if row.success is None else int(bool(row.success)),
            "label": row.label,
            "n_steps": row.stall.get("n_steps", 0),
            "n_errors": row.stall.get("n_errors", 0),
            "injected_count": row.obs.injected_count,
            "l1_seq": " > ".join(row.l1_collapsed),
            "l2_seq": " > ".join(row.l2_collapsed),
            "source_counts": json.dumps(dict(row.source_counts), sort_keys=True),
            "stall_signals": json.dumps(dict(row.stall), sort_keys=True),
            "artifact_paths": ";".join(sorted(set(relpath(p) for p in row.obs.artifact_paths))[:8]),
        }


def manifest_rows(manifest: Sequence[ManifestEntry]) -> Iterable[dict]:
    for entry in manifest:
        yield {
            "kind": entry.kind,
            "path": relpath(entry.path),
            "count": entry.count,
            "note": entry.note,
        }


def build_markdown_report(
    rows: Sequence[ObsRow],
    tasks: Dict[str, TaskInfo],
    summaries: Dict[str, Dict[str, Any]],
    candidates: Sequence[dict],
    top: int,
) -> str:
    lines = []
    lines.append("# Goal/Subgoal Gap Probe")
    lines.append("")
    lines.append("This report uses raw response / trajectory / runtime-log calls first. v3 action_sig is only a marked fallback.")
    lines.append("")
    lines.append("## Corpus")
    lines.append("")
    lines.append("- tasks loaded: {}".format(len(tasks)))
    lines.append("- run task observations: {}".format(len(rows)))
    lines.append("- observations with any v3 fallback subgoal: {}".format(sum(1 for r in rows if r.source_counts.get("fallback", 0) > 0)))
    lines.append("")
    lines.append("### Task Metadata Reality")
    lines.append("")
    strength = Counter(t.metadata_strength for t in tasks.values())
    for key, count in strength.most_common():
        lines.append("- {}: {}".format(key, count))
    lines.append("")

    lines.append("## Top Goal Groups")
    lines.append("")
    ordered = sorted(summaries.items(), key=lambda kv: (kv[1]["pass_n"] + kv[1]["fail_n"], kv[1]["pass_n"]), reverse=True)
    for goal_key, s in ordered[:top]:
        lines.append("### `{}`".format(goal_key))
        lines.append("")
        lines.append("- n/pass/fail: {}/{}/{}".format(s["n"], s["pass_n"], s["fail_n"]))
        lines.append("- domains: `{}`".format(dict(s["domains"].most_common(5))))
        lines.append("- sources: `{}`".format(dict(s["sources"].most_common(5))))
        lines.append("- metadata: `{}`".format(dict(s["strengths"].most_common())))
        if s["fallback_obs"]:
            lines.append("- warning: {} observations needed v3 fallback extraction".format(s["fallback_obs"]))
        if s["l1_gap"]:
            bits = []
            for gap, role, pr, fr in s["l1_gap"][:5]:
                bits.append("{} pass={} fail={} gap={:+.0f}pp".format(role, pct(pr, 1), pct(fr, 1), gap * 100))
            lines.append("- pass-minus-fail L1: " + "; ".join(bits))
        if s["stall_gap"]:
            bits = []
            for gap, name, pr, fr in s["stall_gap"][:5]:
                bits.append("{} pass={} fail={} gap={:+.0f}pp".format(name, pct(pr, 1), pct(fr, 1), gap * 100))
            lines.append("- fail-minus-pass stall: " + "; ".join(bits))
        lines.append("")

    lines.append("## Procedure Candidates")
    lines.append("")
    if not candidates:
        lines.append("No candidate passed the default gap filters. This is a useful negative result if the logs are complete.")
    for cand in candidates[:top]:
        lines.append("### `{}`".format(cand["goal_key"]))
        lines.append("")
        lines.append("- n/pass/fail: {}/{}/{}".format(cand["n"], cand["pass_n"], cand["fail_n"]))
        lines.append("- domains: `{}`".format(cand["domains"]))
        lines.append("- sources: `{}`".format(cand["sources"]))
        if cand["missing_l1_in_fail"]:
            lines.append("- missing L1 in failures: `{}`".format(cand["missing_l1_in_fail"]))
        if cand["fail_stall_signals"]:
            lines.append("- failure stall signals: `{}`".format(cand["fail_stall_signals"]))
        if cand["fallback_obs"]:
            lines.append("- caution: fallback observations: {}".format(cand["fallback_obs"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_console_summary(
    tasks: Dict[str, TaskInfo],
    rows: Sequence[ObsRow],
    summaries: Dict[str, Dict[str, Any]],
    candidates: Sequence[dict],
    top: int,
) -> None:
    print("tasks_loaded", len(tasks))
    print("run_observations", len(rows))
    print("fallback_observations", sum(1 for r in rows if r.source_counts.get("fallback", 0) > 0))
    print("metadata_strength", dict(Counter(t.metadata_strength for t in tasks.values()).most_common()))
    print("\nTOP_GOALS")
    ordered = sorted(summaries.items(), key=lambda kv: (kv[1]["n"], kv[1]["pass_n"]), reverse=True)
    for goal_key, s in ordered[:top]:
        print(
            "{}\tn={}\tpass={}\tfail={}\tdomains={}\tsources={}".format(
                goal_key,
                s["n"],
                s["pass_n"],
                s["fail_n"],
                dict(s["domains"].most_common(3)),
                dict(s["sources"].most_common(3)),
            )
        )
        if s["l1_gap"]:
            print("  l1_gap", [(role, round(gap, 2), round(pr, 2), round(fr, 2)) for gap, role, pr, fr in s["l1_gap"][:4]])
        if s["stall_gap"]:
            print("  stall_gap", [(name, round(gap, 2), round(pr, 2), round(fr, 2)) for gap, name, pr, fr in s["stall_gap"][:4]])
    print("\nCANDIDATES", len(candidates))
    for cand in candidates[:top]:
        print(json.dumps(cand, ensure_ascii=False, sort_keys=True))


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-root", action="append", default=[],
                    help="Task JSON root. Repeatable. Defaults to evaluation_examples/examples and examples_office.")
    ap.add_argument("--schema-root", action="append", default=[],
                    help="Tool API schema root. Defaults to mm_agents/autoglm_v/tools/apis.")
    ap.add_argument("--run-root", action="append", default=[],
                    help="Run/result root to recursively scan for audits, traj.jsonl, logs, and result.txt.")
    ap.add_argument("--baseline-root", action="append", default=[],
                    help="Baseline run root(s) for paired labels; optional.")
    ap.add_argument("--audit", action="append", default=[], help="Explicit *.v3.audit.jsonl file or root.")
    ap.add_argument("--traj", action="append", default=[], help="Explicit traj.jsonl file or root.")
    ap.add_argument("--log", action="append", default=[], help="Explicit runtime/debug log file or root.")
    ap.add_argument("--v3-bank", action="append", default=[], help="Explicit *.v3.json file or root; attribution only.")
    ap.add_argument("--out-dir", default="", help="If set, write report artifacts here.")
    ap.add_argument("--min-goal-size", type=int, default=3)
    ap.add_argument("--min-pass", type=int, default=2)
    ap.add_argument("--min-fail", type=int, default=2)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    task_roots = args.task_root or [
        "evaluation_examples/examples",
        "evaluation_examples/examples_office",
    ]
    schema_roots = args.schema_root or ["mm_agents/autoglm_v/tools/apis"]
    run_roots = args.run_root

    manifest: List[ManifestEntry] = []
    tasks = load_tasks(task_roots)
    for root in task_roots:
        manifest.append(ManifestEntry(kind="task_root", path=root, count=sum(1 for t in tasks.values() if under_root(t.path, root))))

    api_desc = load_api_descriptions(schema_roots)
    for root in schema_roots:
        manifest.append(ManifestEntry(kind="schema_root", path=root, count=sum(1 for k in api_desc if k)))

    audit_files = collect_files(run_roots, args.audit, ["*.v3.audit.jsonl", "*.audit.jsonl"])
    traj_files = collect_files(run_roots, args.traj, ["traj.jsonl"])
    log_files = collect_files(run_roots, args.log, ["runtime.log", "*.log", "debug*.log", "sdebug*.log", "normal*.log"])
    bank_files = collect_files(run_roots, args.v3_bank, ["*.v3.json"])

    observations = load_audit_observations(audit_files, run_roots, api_desc, manifest)
    merge_traj_observations(observations, traj_files, run_roots, api_desc, manifest)
    merge_runtime_logs(observations, log_files, run_roots, api_desc, manifest)
    load_v3_bank(bank_files, manifest)

    scores = parse_scores(run_roots)
    baseline_scores = parse_scores(args.baseline_root)
    # If there is exactly one baseline root but run labels differ, also allow task-id only lookup.
    baseline_by_tid: Dict[str, float] = {}
    for (_run, tid), score in baseline_scores.items():
        baseline_by_tid[tid] = score

    for key, obs in observations.items():
        if key in scores:
            obs.score = scores[key]
        else:
            # If the score key's run label differs from audit/log inferred label, fall back by task id.
            candidates = [score for (_run, tid), score in scores.items() if tid == obs.task_id]
            if len(candidates) == 1:
                obs.score = candidates[0]
        obs.baseline_score = baseline_scores.get(key, baseline_by_tid.get(obs.task_id))

    rows = build_rows(observations, tasks)
    grouped: Dict[str, List[ObsRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task.goal_key].append(row)
    summaries = {
        key: summarize_goal(vals)
        for key, vals in grouped.items()
        if len(vals) >= args.min_goal_size
    }
    candidates = [
        cand
        for key, summary in summaries.items()
        for cand in [candidate_from_summary(key, summary, args.min_pass, args.min_fail)]
        if cand is not None
    ]
    candidates.sort(key=lambda c: (len(c.get("missing_l1_in_fail", [])) + len(c.get("fail_stall_signals", [])), c["n"]), reverse=True)

    print_console_summary(tasks, rows, summaries, candidates, args.top)

    if args.out_dir:
        ensure_dir(args.out_dir)
        write_jsonl(os.path.join(args.out_dir, "artifact_manifest.jsonl"), manifest_rows(manifest))
        write_tsv(
            os.path.join(args.out_dir, "task_goal_table.tsv"),
            [
                "task_id", "domain", "app", "source", "eval_func", "instruction_type",
                "answer_shape", "goal_key", "task_cluster_key", "metadata_strength",
                "ops", "objs", "instruction", "path",
            ],
            task_goal_rows(tasks),
        )
        write_tsv(
            os.path.join(args.out_dir, "task_subgoal_sequences.tsv"),
            [
                "run_id", "task_id", "domain", "app", "source", "goal_key",
                "metadata_strength", "score", "baseline_score", "success", "label",
                "n_steps", "n_errors", "injected_count", "l1_seq", "l2_seq",
                "source_counts", "stall_signals", "artifact_paths",
            ],
            subgoal_rows(rows),
        )
        write_jsonl(os.path.join(args.out_dir, "procedure_candidates.jsonl"), candidates)
        report = build_markdown_report(rows, tasks, summaries, candidates, args.top)
        with open(os.path.join(args.out_dir, "goal_subgoal_gap_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
