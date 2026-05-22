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
from typing import Optional

logger = logging.getLogger("desktopenv.error_ledger")

_ERROR_MARKERS = ("Error:", "error:", "Traceback", "Exception:", "TypeError", "ValueError",
                  "AttributeError", "NameError", "SyntaxError", "RuntimeError", "FileNotFoundError")


def _is_error(exe_result: str) -> bool:
    return any(m in exe_result for m in _ERROR_MARKERS)


def _extract_api_call(action) -> str:
    if isinstance(action, dict):
        return action.get("action_type", "unknown")
    m = re.search(r'(\w+\.\w+)\s*\(', str(action))
    return m.group(1) if m else "unknown"


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
        relevant = [e for e in self.entries if e.get("app") == app and not e.get("resolved")]
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
