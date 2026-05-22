"""
Cross-task Error Ledger.

Lifecycle:
    ledger = ErrorLedger("error_ledger.json")

    # each step, after env.step():
    ledger.record_step(task_id, app, action, exe_result, step_idx)

    # task end:
    ledger.finalize_task(task_id, success=result > 0)

    # inside agent.prepare(), per-step injection:
    context = ledger.retrieve(cur_app)   # -> str, injected into system message
"""

import json
import os
import re
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("desktopenv.error_ledger")

_ERROR_MARKERS = ("Error:", "error:", "Traceback", "Exception:", "TypeError", "ValueError",
                  "AttributeError", "NameError", "SyntaxError", "RuntimeError", "FileNotFoundError")
 
# Cheap deterministic gate signals (no LLM).
_PRECISION_KEYWORDS = ("exact", "specific", "target", "sheet", "title",
                       "range", "cell", "column", "row", "rename")
_DONE_TOKENS = ("DONE", "done()", "finish", "complete the task")

def _is_error(exe_result: str) -> bool:
    return any(m in exe_result for m in _ERROR_MARKERS)


def _extract_api_call(action) -> str:
    if isinstance(action, dict):
        return action.get("action_type", "unknown")
    m = re.search(r'(\w+\.\w+)\s*\(', str(action))
    return m.group(1) if m else "unknown"


def normalize_app(app: Optional[str]) -> str:
    """Canonical app key. Matches the tool_name normalization in autoglm_v/main.py
    and desktop_env _get_obs: wmctrl WM_CLASS like 'Google-chrome' -> 'google_chrome'."""
    if not app:
        return ""
    return app.strip().lower().replace("-", "_")


class ErrorLedger:
    def __init__(self, ledger_path: str, max_inject: int = 5):
        self.ledger_path = ledger_path
        self.max_inject = max_inject
        self.entries: list[dict] = self._load()
        self._current: list[dict] = []   # buffer for the running task

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(self, task_id: str, app: Optional[str], action, exe_result: str, step_idx: int):
        """Call after every env.step(). exe_result is obs['exe_result']."""
        if not exe_result or not _is_error(exe_result):
            return
        app = normalize_app(app) 
        entry = {
            "task_id": task_id,
            "app": app or "unknown",
            "api_call": _extract_api_call(action),
            "error": exe_result[:500],
            "step_idx": step_idx,
            "resolved": False,
        }
        self._current.append(entry)
        logger.debug(f"ErrorLedger: recorded error at step {step_idx}: {entry['api_call']}")

    def finalize_task(self, task_id: str, success: bool):
        """Call once at task end with the final score."""
        if not self._current:
            return

        if success:
            # All errors in a successful task are considered resolved.
            # Coarse but avoids needing step-level credit assignment.
            for e in self._current:
                e["resolved"] = True

        self.entries.extend(self._current)
        self._current = []
        self._save()
        logger.info(f"ErrorLedger: finalized task {task_id}, success={success}, "
                    f"total entries={len(self.entries)}")

    # ------------------------------------------------------------------
    # Retrieval (called inside agent.prepare())
    # ------------------------------------------------------------------

    def retrieve(self, app: Optional[str]) -> str:
        """Return a short prompt snippet of unresolved errors for this app."""
        if not app:
            return ""
        app = normalize_app(app)
        # relevant = [e for e in self.entries if e.get("app") == app and not e.get("resolved")]
        relevant = [e for e in self.entries if e.get("app") == app]
        if not relevant:
            return ""

        # Deduplicate by (api_call, error_prefix) — keep most recent occurrences
        seen: set[tuple] = set()
        unique: list[dict] = []
        for e in reversed(relevant):
            key = (e.get("api_call", ""), e.get("error", "")[:80])
            if key not in seen:
                seen.add(key)
                unique.append(e)
            if len(unique) >= self.max_inject:
                break

        lines = [f"## Known errors for {app} (from past tasks — avoid repeating these mistakes):"]
        for e in reversed(unique):
            lines.append(f"- {e['api_call']}: {e['error'][:200]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"ErrorLedger: failed to load {self.ledger_path}: {e}")
        return []

    def _save(self):
        try:
            with open(self.ledger_path, "w") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"ErrorLedger: failed to save {self.ledger_path}: {e}")

# ======================================================================
# v2 -- deterministic gated structured error memory (Condition-2).
# ======================================================================

class ErrorLedgerV2:
    inject_target = "user"   # state-dependent guidance -> user/state turn, not system

    def __init__(self, ledger_path: str, max_inject: int = 1,
                 max_consults_per_task: int = 1, ttl_steps: int = 4):
        self.ledger_path = ledger_path
        base, _ext = os.path.splitext(ledger_path)
        self.cards_path = base + ".cards.json"
        self.audit_path = base + ".audit.jsonl"
        self.max_inject = max_inject
        self.max_consults_per_task = max_consults_per_task
        self.ttl_steps = ttl_steps

        self.cards: List[dict] = self._load_cards()

        # per-task runtime bookkeeping
        self._steps: List[dict] = []                 # EVERY step of the running task
        self._task_consults: Dict[str, int] = {}     # memory_id -> consults this task
        self._active_memo: Optional[dict] = None
        self._active_memo_step: Optional[int] = None

    def count(self) -> int:
        return len(self.cards)

    # ----- recording: every step --------------------------------------
    def record_step(self, task_id: str, app: Optional[str], action, exe_result: str, step_idx: int):
        app = normalize_app(app)
        exe_result = exe_result or ""
        self._steps.append({
            "task_id": task_id,
            "app": app or "unknown",
            "step_idx": step_idx,
            "api_call": _extract_api_call(action),
            "exe_result": exe_result[:1000],
            "is_error": _is_error(exe_result),
        })

    # ----- finalize: build/refresh templated cards + audit dump -------
    def finalize_task(self, task_id: str, success: bool):
        if self._steps:
            # audit: persist the full step trace (raw material for a future v3 distill)
            self._append_audit(task_id, success, self._steps)
            for s in self._steps:
                if s["is_error"]:
                    self._upsert_card(s, task_id, success)
            self._save_cards()
            logger.info(f"ErrorLedgerV2: finalized {task_id} success={success} "
                        f"steps={len(self._steps)} cards={len(self.cards)}")
        # reset per-task state UNCONDITIONALLY (consult cap / active memo are per-task)
        self._steps = []
        self._task_consults = {}
        self._active_memo = None
        self._active_memo_step = None

    def _memory_id(self, app: str, api_call: str) -> str:
        return f"ERR_{app}_{api_call}".replace(".", "_")

    def _upsert_card(self, step: dict, task_id: str, success: bool):
        app, api_call = step["app"], step["api_call"]
        mid = self._memory_id(app, api_call)
        for c in self.cards:
            if c["memory_id"] == mid:
                c["support_count"] += 1
                c["last_seen_task"] = task_id
                c["last_seen_step"] = step["step_idx"]
                c["last_task_success"] = bool(success)   # metadata only; NOT used to filter
                return
        # deterministic, generic, honest template (no LLM, no fabricated specifics)
        self.cards.append({
            "memory_id": mid,
            "app": app,
            "api_call": api_call,
            "trigger": f"About to call {api_call} in {app}, OR the previous action "
                       f"errored / the screen did not visibly change.",
            "failure_mode": f"A {api_call} call in {app} previously failed or did not take effect.",
            "recovery_plan": "Re-check the result before proceeding; retry the operation "
                             "or use an alternate route to achieve the same effect.",
            "do_not_do": "Do not assume the operation succeeded and move to the next "
                         "subgoal (or call DONE) without checking.",
            "verification_cue": "Confirm the expected UI / value / state actually changed.",
            "when_not_to_use": "Ignore if the current screen already shows the expected result.",
            "support_count": 1,
            "last_seen_task": task_id,
            "last_seen_step": step["step_idx"],
            "last_task_success": bool(success),
        })

    # ----- gate: when to even consider injecting ----------------------
    def _should_inject(self, instruction: str, last_result: str, recent_actions: List[str]) -> bool:
        if _is_error(last_result or ""):
            return True
        ra = [a for a in (recent_actions or []) if a]
        # repeated action (agent likely stuck on the same call)
        if len(ra) >= 2:
            last_calls = [_extract_api_call(a) for a in ra[-2:]]
            if last_calls[0] == last_calls[1] and last_calls[0] != "unknown":
                return True
        # about to finish -> good moment to surface a "verify before DONE" memo
        if ra and any(tok in ra[-1] for tok in _DONE_TOKENS):
            return True
        il = (instruction or "").lower()
        if any(k in il for k in _PRECISION_KEYWORDS):
            return True
        return False

    # ----- scoring (deterministic heuristic, no embedding) ------------
    def _score(self, card: dict, last_result: str, recent_actions: List[str]) -> int:
        score = 0
        ra_text = " ".join(recent_actions or [])
        if card["api_call"] in ra_text:
            score += 3
        if card["api_call"].split(".")[0] in (last_result or ""):
            score += 1
        score += min(card.get("support_count", 1), 3)              # mild frequency prior
        if self._task_consults.get(card["memory_id"], 0) > 0:      # already used this task
            score -= 5
        return score

    # ----- retrieval --------------------------------------------------
    def retrieve(self, app: Optional[str], instruction: str = "", last_result: str = "",
                 recent_actions: Optional[list] = None, step_idx: int = 0, **kwargs) -> str:
        app = normalize_app(app)
        if not app or not self.cards:
            return ""
        recent_actions = recent_actions or []

        # TTL: keep an already-selected memo active for a few steps without re-selecting
        if self._active_memo is not None and self._active_memo_step is not None:
            if step_idx - self._active_memo_step <= self.ttl_steps:
                return self._format([self._active_memo])
            self._active_memo = None
            self._active_memo_step = None

        if not self._should_inject(instruction, last_result, recent_actions):
            return ""

        candidates = [c for c in self.cards if c["app"] == app
                      and self._task_consults.get(c["memory_id"], 0) < self.max_consults_per_task]
        if not candidates:
            return ""
        candidates.sort(key=lambda c: self._score(c, last_result, recent_actions), reverse=True)
        chosen = candidates[: self.max_inject]
        if not chosen or self._score(chosen[0], last_result, recent_actions) <= 0:
            return ""

        for c in chosen:
            self._task_consults[c["memory_id"]] = self._task_consults.get(c["memory_id"], 0) + 1
        self._active_memo = chosen[0]
        self._active_memo_step = step_idx
        return self._format(chosen)

    def _format(self, cards: List[dict]) -> str:
        lines = [
            "* Cross-task Error Memory:",
            "Past failures may come from a different state. Use a note below ONLY if its "
            "Trigger matches the current screen and instruction; otherwise ignore it and "
            "trust the current screen.",
        ]
        for c in cards:
            lines += [
                "",
                f"[{c['memory_id']}]",
                f"Trigger: {c['trigger']}",
                f"Past failure: {c['failure_mode']}",
                f"Recovery: {c['recovery_plan']}",
                f"Do not: {c['do_not_do']}",
                f"Verify before continuing or DONE: {c['verification_cue']}",
                f"Ignore if: {c['when_not_to_use']}",
            ]
        return "\n".join(lines)

    # ----- persistence -------------------------------------------------
    def _load_cards(self) -> List[dict]:
        if os.path.exists(self.cards_path):
            try:
                with open(self.cards_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"ErrorLedgerV2: failed to load {self.cards_path}: {e}")
        return []

    def _save_cards(self):
        try:
            with open(self.cards_path, "w") as f:
                json.dump(self.cards, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"ErrorLedgerV2: failed to save {self.cards_path}: {e}")

    def _append_audit(self, task_id: str, success: bool, steps: List[dict]):
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps({"task_id": task_id, "success": bool(success),
                                    "steps": steps}, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"ErrorLedgerV2: failed to append audit {self.audit_path}: {e}")


# ======================================================================
# factory
# ======================================================================

def make_error_ledger(path: str, version: str = "v1", max_inject: int = 1,
                      max_consults_per_task: int = 1, ttl_steps: int = 4):
    """v1 keeps its original defaults (max_inject=5, system injection).
    v2 tunables only apply to v2."""
    if version == "v2":
        return ErrorLedgerV2(path, max_inject=max_inject,
                             max_consults_per_task=max_consults_per_task,
                             ttl_steps=ttl_steps)
    return ErrorLedger(path)