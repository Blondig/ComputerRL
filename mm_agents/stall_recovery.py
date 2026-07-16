"""Stall-triggered retrieval-augmented recovery (--stall_recovery).

Three arms over the SAME minimal detector (exact-repeat was selected by the
offline precision audit in analyze_stall_retrieval_probe.py):

  replan (Arm A): on a detected stall, append a deterministic break-loop nudge
                  to the next user turn ("stop repeating, re-plan differently").
  hint   (Arm B): Arm A + retrieve a similar successful transition from a
                  cross-task hint bank and attach it as a REFERENCE card for the
                  replanner (never executed, never coordinates).
  forbid (Arm C): Arm A + a state-local contract that rejects the exact repeated
                  action and asks the same model for one different action.

Independence contract: this module is orthogonal to --recovery (the frozen
intra-task action-interface repair arm, which owns pre-boundary/representation
failures) and to the error ledger. The replan/hint arms only add text to the next
user turn. The forbid arm may reject and regenerate one exact repeated action;
it does not alter the interface-repair flow. With --stall_recovery off the agent
prompt stays byte-identical to baseline.

Online stall rule (deliberately minimal -- one rule):
  exact_repeat  k consecutive equivalent EXECUTED actions (args included), no
                error in the window, and every adjacent post-action screenshot
                is available and at least 0.995 similar.

Why only this rule (pass-vs-fail precision audit, 9 domains, 2026-07):
exact_repeat is the only failure-enriched signal (pass 12/122 = 9.8% vs fail
78/209 = 37.3%). lowlevel_static was non-discriminative (11.5% vs 14.4% -- it
matches ordinary GUI operation), and execution-error loops already feed back
through Previous Action Result, so both were dropped instead of being carried
as extra machinery. representation/no_action failures belong to --recovery.

Hysteresis / budget (the v3 flooding lesson): cooldown steps between fires and
a hard per-task intervention cap; the retrieved hint is attached only on the
FIRST fire of a task -- if the agent is still stuck afterwards, later fires
fall back to the vanilla nudge so a bad hint cannot keep anchoring the replan.
"""

import ast
import json
import logging
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from .error_ledger import _is_error

try:
    from PIL import Image
except ImportError:  # pragma: no cover - PIL is in requirements; stay importable without it
    Image = None

logger = logging.getLogger("desktopenv.stall_recovery")

THUMB_W, THUMB_H = 32, 18

_CODE_FENCE_RE = re.compile(r"```(?:\w+\s+)?(.*?)```", re.DOTALL)
_CALL_RE = re.compile(r"\b((?:\w*Tools|Agent)\.\w+)\s*\(")
_ADAPTER_CALL_RE = re.compile(r"\b[A-Za-z_]\w*Tools\.print_result\s*\([^)]*\)\s*;?")
_RESULT_ERROR_RE = re.compile(
    r"\b(?:error|failed|failure|exception|traceback)\b|invalid action",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")
_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "your",
    "you", "please", "could", "would", "help", "want", "need", "using",
    "current", "make", "set", "can", "all", "are", "have", "has", "its",
}

_APP_ALIASES = {
    "google_chrome": "browser", "chrome": "browser", "chromium": "browser",
    "libreoffice_calc": "calc", "libreoffice_impress": "impress",
    "libreoffice_writer": "writer", "vs_code": "code", "vscode": "code",
    "code": "code", "vlc": "vlc", "gimp": "gimp", "thunderbird": "thunderbird",
    "os": "os", "multi_apps": "multi_apps",
}


def normalize_app_family(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _APP_ALIASES.get(key, key or "unknown")


def pseudo_action(response_text: str) -> str:
    """The pre-grounding action the model submitted: first fenced code block,
    or a bare WAIT/DONE/FAIL. Empty string when nothing parseable exists."""
    text = str(response_text or "")
    stripped = text.strip()
    if stripped in ("WAIT", "DONE", "FAIL"):
        return stripped
    matches = _CODE_FENCE_RE.findall(text)
    return matches[0].strip() if matches else ""


def _strip_mangle_noise(text: str) -> str:
    """Same normalization as main._strip_code_noise: drop the serialization
    mangle (leading ``python\\n`` literal or line, trailing ``\\n``) so a mangled
    and a clean submission of the SAME action share one key. Without this the
    impress uptake audit showed the loop 'escaping' by re-emitting the identical
    action minus the mangle prefix -- a fake behavior change."""
    text = re.sub(r"^\s*python\\n", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*python\s*\n", "", text, flags=re.IGNORECASE).strip()
    if text.endswith("\\n"):
        text = text[:-2].strip()
    return text


def action_key(action_text: str) -> str:
    text = _strip_mangle_noise(str(action_text or ""))
    text = _ADAPTER_CALL_RE.sub("", text).strip(" ;")
    return re.sub(r"\s+", "", text).lower()


def _executed_action_key(action: Any) -> str:
    """Canonical key for exact-repeat detection without changing arguments."""
    if action is None:
        return ""
    if isinstance(action, (dict, list, tuple)):
        try:
            return json.dumps(
                action,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            return str(action)

    # Mangle-normalize BEFORE keying: python\nX(...) and X(...) must share one
    # key, or the detector misses mangle-alternating loops and the forbid hard
    # ban is bypassable by re-emitting the same action with the mangle prefix.
    text = _strip_mangle_noise(str(action))
    text = _ADAPTER_CALL_RE.sub("", text).strip(" ;")
    if not text or text.lower() == "parse error":
        return ""
    try:
        return ast.dump(ast.parse(text), include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        return text


def _has_error_result(exe_result: str) -> bool:
    text = str(exe_result or "")
    return _is_error(text) or bool(_RESULT_ERROR_RE.search(text))


def call_name(action_text: str) -> str:
    m = _CALL_RE.search(str(action_text or ""))
    return m.group(1) if m else ""


def screenshot_thumbnail(screenshot_bytes) -> Optional[tuple]:
    if not screenshot_bytes or Image is None:
        return None
    try:
        with Image.open(BytesIO(screenshot_bytes)) as source:
            gray = source.convert("L").resize(
                (THUMB_W, THUMB_H), resample=Image.Resampling.BILINEAR
            )
            return tuple(gray.getdata())
    except Exception:
        return None


def thumb_similarity(a: Optional[tuple], b: Optional[tuple]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    diff = sum(abs(x - y) for x, y in zip(a, b))
    return max(0.0, 1.0 - diff / float(255 * len(a)))


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
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


class StallDetector:
    """Minimal online stall detector: one precision-first exact-repeat rule.

    update() is fed once per agent turn with the PREVIOUS action's outcome and
    returns a signal dict when the intervention should fire this turn. A stall
    is k consecutive equivalent executed actions with no error in the window
    and no visible progress between their post-action screenshots. Missing
    state evidence vetoes the trigger.
    """

    def __init__(
        self,
        repeat_k: int = 2,
        state_threshold: float = 0.995,
        cooldown_steps: int = 4,
        max_interventions: int = 2,
    ):
        self.repeat_k = max(2, repeat_k)
        self.state_threshold = state_threshold
        self.cooldown_steps = cooldown_steps
        self.max_interventions = max_interventions
        self.reset()

    def reset(self):
        self.steps: List[Dict[str, Any]] = []
        self.fires = 0
        self._cooldown_until = -1

    def _match(self) -> Optional[Dict[str, Any]]:
        if len(self.steps) < self.repeat_k:
            return None
        window = self.steps[-self.repeat_k:]
        if any(s["is_error"] for s in window):
            return None
        keys = [s["key"] for s in window]
        if not keys[0] or len(set(keys)) != 1:
            return None

        similarities = [
            thumb_similarity(left["thumb"], right["thumb"])
            for left, right in zip(window, window[1:])
        ]
        if any(sim is None or sim < self.state_threshold for sim in similarities):
            return None
        return {
            "rule": "exact_repeat",
            "detail": "your last {} executed actions were identical without visible progress".format(
                self.repeat_k
            ),
            "state_similarity": min(similarities),
        }

    def update(
        self,
        executed_action: Any,
        exe_result: str,
        screenshot: Any,
    ) -> Optional[Dict[str, Any]]:
        key = _executed_action_key(executed_action)
        result = str(exe_result or "")
        self.steps.append({
            "key": key,
            "action": " ".join(str(executed_action or "").split())[:120],
            "is_error": not key or _has_error_result(result),
            "thumb": screenshot_thumbnail(screenshot),
        })
        step_index = len(self.steps)
        if self.fires >= self.max_interventions or step_index <= self._cooldown_until:
            return None
        match = self._match()
        if match is None:
            return None
        self.fires += 1
        self._cooldown_until = step_index + self.cooldown_steps
        return {
            "rule": match["rule"],
            "detail": match["detail"],
            "state_similarity": match["state_similarity"],
            "fire_index": self.fires,
            "step_index": step_index,
            "stalled_key": self.steps[-1]["key"],
            "stalled_action": self.steps[-1]["action"],
        }


class HintBank:
    """Read-only cross-task hint bank (jsonl from build_stall_hint_bank.py).

    Retrieval keys: hard app-family filter, instruction jaccard, recent-call
    suffix match; the screenshot is a WEAK ranker only (hint-grade, not
    replay-grade -- state similarity orders candidates, it does not gate them).
    Donors whose instruction equals the current one are excluded (same-task
    leakage proxy: the agent does not know its own task_id)."""

    MIN_INSTRUCTION_JACCARD = 0.1
    MIN_CONTEXT_SUFFIX = 0.5

    def __init__(self, path: str):
        self.path = path
        self.by_app: Dict[str, List[dict]] = defaultdict(list)
        count = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry["_instr_tokens"] = set(entry.get("instr_tokens") or []) or _tokens(
                    entry.get("instruction", "")
                )
                thumb = entry.get("pre_thumb")
                entry["_thumb"] = tuple(thumb) if thumb else None
                self.by_app[entry.get("app_family", "unknown")].append(entry)
                count += 1
        logger.info("HintBank loaded: %d entries, %d app families (%s)",
                    count, len(self.by_app), path)

    def count(self) -> int:
        return sum(len(v) for v in self.by_app.values())

    def retrieve(
        self,
        app: str,
        instruction: str,
        recent_responses: Sequence[str],
        screenshot=None,
    ) -> Optional[dict]:
        family = normalize_app_family(app)
        candidates = self.by_app.get(family, [])
        if not candidates:
            return None
        instr_tokens = _tokens(instruction)
        recent_calls = [c for c in (call_name(pseudo_action(r)) for r in recent_responses) if c]
        cur_thumb = screenshot_thumbnail(screenshot)

        best, best_score = None, -1.0
        seen_tasks = set()
        for entry in candidates:
            if entry.get("instruction", "").strip() == (instruction or "").strip():
                continue
            instr_sim = _jaccard(instr_tokens, entry["_instr_tokens"])
            ctx_sim = _suffix_similarity(recent_calls, entry.get("context_calls") or [])
            if instr_sim < self.MIN_INSTRUCTION_JACCARD and ctx_sim < self.MIN_CONTEXT_SUFFIX:
                continue
            state_sim = thumb_similarity(cur_thumb, entry["_thumb"])
            score = 0.4 * instr_sim + 0.3 * ctx_sim + 0.3 * (state_sim or 0.0)
            task_id = entry.get("task_id", "")
            if score > best_score or (score == best_score and task_id not in seen_tasks):
                best, best_score = entry, score
            seen_tasks.add(task_id)
        if best is None:
            return None
        result = dict(best)
        result.pop("_instr_tokens", None)
        result.pop("_thumb", None)
        result["score"] = round(best_score, 4)
        return result


def call_dumps(code_text: str) -> set:
    """AST dump of each top-level statement in a submitted code block.

    Used for EXACT per-statement containment checks in the state-local hard ban:
    case- and string-content-faithful, unlike action_key (a fuzzy retrieval key
    that lowercases and strips whitespace, so Agent.type(text='A B') and
    Agent.type(text='ab') would collide). Returns an empty set when the text does
    not parse; executable candidates are checked separately before dispatch."""
    text = _strip_mangle_noise(str(code_text or ""))
    text = _ADAPTER_CALL_RE.sub("", text).strip(" ;")
    if not text:
        return set()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    return {ast.dump(node, include_attributes=False) for node in tree.body}


def _masked_call_sequence(calls: Sequence[str], limit: int = 3) -> str:
    masked = []
    for call in list(calls)[:limit]:
        call = re.sub(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?", "<num>", str(call))
        masked.append(call[:120])
    return " -> ".join(masked) if masked else "(not recorded)"


def format_stall_intervention(
    signal: Dict[str, Any], hint: Optional[dict] = None, forbid: bool = False
) -> str:
    lines = [
        "* Stall Alert: You appear to be stuck -- {}. The current approach is not "
        "making progress. Stop repeating it. Re-examine the current screen and "
        "re-plan with a DIFFERENT method (a different menu path, tool, or API) to "
        "move the task forward.".format(signal.get("detail", "repeated behavior detected"))
    ]
    if forbid:
        lines.append(
            "* Recovery Contract (mandatory): the current screen did not change after "
            "`{stalled}`. Do NOT submit that exact action again while this screen remains "
            "unchanged. Choose ONE different executable next action for the current "
            "screen and output it in a single fenced python code block. Do not generate "
            "multiple candidate plans.".format(
                stalled=str(signal.get("stalled_action", ""))[:120] or "(your previous action)"
            )
        )
    if hint is not None:
        lines.append(
            "* Reference from a similar past task (it may NOT apply here -- trust the "
            "current screen over this reference):\n"
            "  - Similar task: \"{instruction}\"\n"
            "  - At a similar point it reasoned: \"{reasoning}\"\n"
            "  - It then executed: {continuation} ... and that task completed successfully.".format(
                instruction=str(hint.get("instruction", ""))[:160],
                reasoning=str(hint.get("reasoning", "")).strip()[:240] or "(no reasoning recorded)",
                continuation=_masked_call_sequence(hint.get("continuation_calls") or []),
            )
        )
    return "\n\n".join(lines)
