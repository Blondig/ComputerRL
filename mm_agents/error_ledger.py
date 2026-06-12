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
import hashlib
from collections import Counter
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


# --- GUI-agent-generic ledger admission classifier --------------------------
# Grounding boundary:  model output -> parser -> EXECUTABLE COMMAND -> env/tool.
# Cross-task memory may learn ONLY from failures AFTER a valid command crosses
# this boundary (the world actually ran it and pushed back). Failures BEFORE the
# boundary (the command never became valid code / never ran) carry no
# transferable world-knowledge -- they are this agent's own output-formatting /
# serialization noise, and writing them in produces false causality
# ("Tool.X failed" when really the agent emitted a malformed fenced block).
#
# Detection is STRUCTURAL -- "did the command compile and run" -- never by app
# or tool name. That is what keeps it from overfitting to OSWorld/LibreOffice:
# a Calc tool call, a browser action, or a JSON action schema all share this one
# boundary. A compile-time error means the dispatched code was not valid code,
# so nothing was ever tested against the world (this is the generic form of the
# python\n / content-block-repr / nXxxTools mangling -- they all land here).
_COMPILE_ERROR_MARKERS = ("SyntaxError", "IndentationError", "TabError")


def classify_error_step(exe_result: str, action_sig: str = "") -> str:
    """Bucket an errored step by WHERE in the action pipeline it failed.

    Returns:
        "representation" -- command never compiled/ran (pre-boundary)  -> DROP
        "no_action"      -- parser produced no executable action       -> DROP
        "execution"      -- command ran, env/tool returned a failure   -> ADMIT
    """
    text = exe_result or ""
    if any(m in text for m in _COMPILE_ERROR_MARKERS):
        return "representation"
    if not action_sig or action_sig == "unknown":
        return "no_action"
    return "execution"


def is_admissible_error(exe_result: str, action_sig: str = "") -> bool:
    """True iff the error reflects real post-boundary world feedback, i.e. the
    only kind worth carrying across tasks. Used by ErrorLedgerV3.finalize_task to
    gate which errored steps become persistent cross-task error_notes.

    SCOPE CAVEAT: this treats a compile-time error (SyntaxError/...) as a
    pre-boundary mangling failure, which holds when the dispatched command is the
    AGENT's own code wrapping a GUI action (the office domains). If a tool itself
    *runs user code* (e.g. CodeTools.run_python), its SyntaxError IS post-boundary
    feedback and this rule would wrongly drop it. Before extending to such a
    domain, re-check the audit class distribution rather than assuming this rule."""
    return classify_error_step(exe_result, action_sig) == "execution"


# Adapter / instrumentation actions: calls the grounding layer appends to EVERY
# command (grounding_agent.tool_commands suffixes `{Tool}.print_result()`), so the
# agent never *chooses* them and they sit in recent_text on every task. Keying
# cross-task memory on such an action yields a semantically-empty note that fires
# on ~all tasks (it was 88% of v33's injections). Exclude it from both learning
# and retrieval. This is the framework's universal observe hook -- a per-agent
# fact, NOT a per-app/per-tool allowlist, so it does not reintroduce overfitting.
_ADAPTER_METHODS = ("print_result",)


def _is_adapter_action(sig: str) -> bool:
    sig = sig or ""
    return any(sig == m or sig.endswith("." + m) for m in _ADAPTER_METHODS)


def _extract_api_call(action) -> str:
    if isinstance(action, dict):
        return action.get("action_type", "unknown")
    # str(action) may carry literal "\n"/"\t" escape sequences (a stray backslash-n,
    # not a real newline); the \w+ then swallows the leading char -> "nCalcTools.save".
    # Unescape to real whitespace first so the extracted call name is clean.
    text = str(action).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    m = re.search(r'(\w+\.\w+)\s*\(', text)
    return m.group(1) if m else "unknown"


def normalize_app(app: Optional[str]) -> str:
    """Canonical app key. Matches the tool_name normalization in autoglm_v/main.py
    and desktop_env _get_obs: wmctrl WM_CLASS like 'Google-chrome' -> 'google_chrome'."""
    if not app:
        return ""
    return app.strip().lower().replace("-", "_")


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:#-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "your", "you",
    "are", "was", "were", "will", "have", "has", "had", "none", "true", "false",
}


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()[:length]


def _tokenize(text: str, max_tokens: int = 80) -> List[str]:
    counts = Counter(
        tok.lower()
        for tok in _TOKEN_RE.findall(text or "")
        if tok.lower() not in _STOPWORDS and len(tok) <= 48
    )
    return [tok for tok, _count in counts.most_common(max_tokens)]


def _active_title(obs: Optional[dict]) -> str:
    if not obs:
        return ""
    apps = obs.get("apps") or {}
    cur_id = obs.get("cur_window_id")
    if cur_id in apps:
        return str(apps[cur_id].get("title", ""))[:160]
    return ""


def summarize_observation(obs: Optional[dict]) -> dict:
    """Lightweight, non-visual state abstraction for v3 trajectory memory.

    It intentionally avoids screenshot embeddings and raw screenshot storage.
    The signature is conservative: good enough for high-precision retrieval,
    not intended to be a full GUI state identity system.
    """
    obs = obs or {}
    app = normalize_app(obs.get("cur_app"))
    title = _active_title(obs)
    app_info = str(obs.get("app_info") or "")
    tree = str(obs.get("accessibility_tree") or "")

    token_text = " ".join([
        app,
        title,
        app_info[:5000],
        tree[:5000],
    ])
    tokens = _tokenize(token_text, max_tokens=80)

    sig_src = "\n".join([
        app,
        title.lower(),
        " ".join(tokens[:60]),
        _short_hash(app_info[:5000]),
    ])
    return {
        "app": app or "unknown",
        "title": title,
        "tokens": tokens,
        "state_sig": _short_hash(sig_src),
        "app_info_hash": _short_hash(app_info[:5000]),
        "a11y_hash": _short_hash(tree[:5000]),
    }


# High-level tool calls the agent emits in its response, e.g. CalcTools.set_value(,
# ImpressTools.add_text(, BrowserTools.open_tab(, Agent.click(. These carry real
# semantics; the grounded action that actually runs is a coarse pyautogui verb.
_HIGH_LEVEL_CALL_RE = re.compile(r"((?:\w*Tools|Agent)\.\w+)\s*\(")

# Actions that are not part of a procedure's "shape". For this agent,
# procedural_memory documents `Agent.exit(success=True)` as the canonical "done"
# and `Agent.wait()` as wait.
#   _WAIT_SIGS  : non-semantic pauses -> dropped, but they do NOT break a run
#                 (open_url -> wait -> click is the same procedure as open_url ->
#                 click; the agent waits reactively from the live screen anyway).
#   _BREAK_SIGS : real discontinuities -> they break a run, so a snippet never
#                 stitches across them (this also keeps Agent.exit out of every
#                 "successful pattern", which would otherwise train premature exit).
_WAIT_SIGS = {"WAIT", "Agent.wait"}
_BREAK_SIGS = {"unknown", "DONE", "FAIL", "Agent.exit"}


def _executed_code(response: str) -> str:
    """Return only the code the agent actually executes.

    Responses look like ``<think>{plan}</think><answer>```python\\n{ONE-LINE}\\n```</answer>``.
    The <think> section is free-form planning that may *mention* other tool calls, so
    extracting from the whole response can record a planned call instead of the
    executed one. Restrict to the final <answer> code block (fall back gracefully).
    """
    if not response:
        return ""
    tail = response.rsplit("<answer>", 1)[-1]   # after the last <answer>, else the whole text
    # `\s+` (not `\s*`): a markdown language tag is a word followed by whitespace.
    # With `\s*`, an untagged single-line block ```CalcTools.save()``` would have
    # "CalcTools" eaten as the "tag", leaving ".save()" and losing the call name.
    blocks = re.findall(r"```(?:\w+\s+)?(.*?)```", tail, re.DOTALL)
    if blocks:
        return blocks[-1]
    return tail


def _action_signature(action, response: str = "") -> str:
    """Canonical action label for trajectory memory.

    Prefer the high-level tool call the agent actually executes in the <answer>
    code block (e.g. ``CalcTools.set_value``). The grounded ``action`` that gets
    executed is usually a coarse ``pyautogui.{click,write,hotkey}`` that collapses
    every distinct operation into ~3 buckets, which starves all three memory banks.
    """
    if response:
        m = _HIGH_LEVEL_CALL_RE.search(_executed_code(response))
        if m:
            return m.group(1)
    api_call = _extract_api_call(action)
    if api_call != "unknown":
        return api_call
    text = str(action or "")
    if text in {"WAIT", "DONE", "FAIL"}:
        return text
    if "pyautogui.hotkey" in text:
        return "pyautogui.hotkey"
    if "pyautogui.click" in text:
        return "pyautogui.click"
    if "pyautogui.write" in text:
        return "pyautogui.write"
    return "unknown"


class ErrorLedger:
    def __init__(self, ledger_path: str, max_inject: int = 5):
        self.ledger_path = ledger_path
        self.max_inject = max_inject
        self.entries: list[dict] = self._load()
        self._current: list[dict] = []   # buffer for the running task

    def count(self) -> int:
        return len(self.entries)

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

    def retrieve(self, app: Optional[str], **kwargs) -> str:
        """Return a short prompt snippet of unresolved errors for this app.

        v1 is the dumb baseline: it ignores the extra context kwargs
        (instruction/last_result/recent_actions/step_idx) that v2 gates on.
        Accepting **kwargs keeps the agent's retrieve() call site uniform across versions."""
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
        self._injected: List[dict] = []          # {step_idx, memory_id} surfaced this task

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

    def reset_task(self):
        """Drop per-task buffers. Called at task start so a task that crashed before
        finalize_task() cannot leak its steps/injections into the next task's audit."""
        self._steps = []
        self._task_consults = {}
        self._active_memo = None
        self._active_memo_step = None
        self._injected = []

    # ----- finalize: build/refresh templated cards + audit dump -------
    def finalize_task(self, task_id: str, success: bool):
        if self._steps:
            # audit: persist the full step trace + what the ledger injected this task
            self._append_audit(task_id, success, self._steps, self._injected)
            for s in self._steps:
                if s["is_error"]:
                    self._upsert_card(s, task_id, success)
            self._save_cards()
            logger.info(f"ErrorLedgerV2: finalized {task_id} success={success} "
                        f"steps={len(self._steps)} cards={len(self.cards)} "
                        f"injected={len(self._injected)}")
        # reset per-task state UNCONDITIONALLY (consult cap / active memo are per-task)
        self.reset_task()

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

        # TTL: keep an already-selected memo active for a few steps without re-selecting,
        # but only while we are still in the same app it was selected for (an in-task
        # app switch must drop it, otherwise a stale memo pollutes the new app's context).
        if self._active_memo is not None and self._active_memo_step is not None:
            if (self._active_memo.get("app") == app
                    and 0 <= step_idx - self._active_memo_step <= self.ttl_steps):
                self._injected.append({"step_idx": step_idx,
                                       "memory_id": self._active_memo["memory_id"],
                                       "ttl_repeat": True})
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
            self._injected.append({"step_idx": step_idx, "memory_id": c["memory_id"]})
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

    def _append_audit(self, task_id: str, success: bool, steps: List[dict],
                      injected: Optional[List[dict]] = None):
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps({"task_id": task_id, "success": bool(success),
                                    "steps": steps, "injected": injected or []},
                                   ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"ErrorLedgerV2: failed to append audit {self.audit_path}: {e}")


# ======================================================================
# v3 -- state-conditioned passive trajectory memory.
# ======================================================================

class ErrorLedgerV3:
    """Lightweight cross-task trajectory memory from passive rollouts.

    v3 deliberately avoids active exploration, embeddings, model training, and
    MCTS. It records pre_state -> action_signature -> post_state transitions, and
    retrieves by EXACT action-signature trigger (no state-token similarity, no
    score thresholds): memory surfaces only when the agent just invoked an action
    that is in the bank, so there is nothing to hand-tune and no cross-task flooding.

    Two banks:
      * error_notes: "this exact call has failed here before" (carries the real error)
      * success_snippets: "after this call, a past success did X" (next-step proposal)
    (An earlier ambiguity/action_stats bank was removed: same-state dispersion never
    recurs under an exact state_sig, so it was dead weight.)
    """

    inject_target = "user"

    def __init__(self, ledger_path: str, max_inject: int = 2,
                 max_consults_per_task: int = 1, ttl_steps: int = 4):
        self.ledger_path = ledger_path
        base, _ext = os.path.splitext(ledger_path)
        self.memory_path = base + ".v3.json"
        self.audit_path = base + ".v3.audit.jsonl"
        self.max_inject = max_inject
        self.max_consults_per_task = max_consults_per_task
        self.ttl_steps = ttl_steps

        self.memory: Dict[str, Any] = self._load_memory()
        self._steps: List[dict] = []
        self._task_consults: Dict[str, int] = {}
        self._injected: List[dict] = []          # {step_idx, memory_id, kind} surfaced this task

    def count(self) -> int:
        return (
            len(self.memory.get("error_notes", {}))
            + len(self.memory.get("success_snippets", {}))
            + len(self.memory.get("action_stats", {}))
        )

    def record_step(self, task_id: str, app: Optional[str], action, exe_result: str, step_idx: int):
        """Compatibility fallback. The v3 runner should call record_transition."""
        if not exe_result or not _is_error(exe_result):
            return
        app = normalize_app(app) or "unknown"
        state = {"app": app, "tokens": [], "state_sig": "unknown", "title": ""}
        self._steps.append({
            "task_id": task_id,
            "app": app,
            "step_idx": step_idx,
            "pre": state,
            "post": state,
            "action_sig": _action_signature(action),
            "action_text": str(action)[:1000],
            "response": "",
            "exe_result": (exe_result or "")[:1000],
            "is_error": True,
            "done": False,
        })

    def record_transition(self, task_id: str, instruction: str, response: str, action,
                          pre_obs: dict, post_obs: dict, reward: float = 0.0,
                          done: bool = False, info: Optional[dict] = None,
                          step_idx: int = 0):
        pre = summarize_observation(pre_obs)
        post = summarize_observation(post_obs)
        exe_result = (post_obs or {}).get("exe_result", "") or ""
        action_sig = _action_signature(action, response)
        self._steps.append({
            "task_id": task_id,
            "instruction": instruction or "",
            "task_tokens": _tokenize(instruction or "", max_tokens=40),
            "app": pre.get("app") or post.get("app") or "unknown",
            "step_idx": step_idx,
            "pre": pre,
            "post": post,
            "action_sig": action_sig,
            "action_text": str(action)[:1000],
            "response": (response or "")[:1200],
            "exe_result": exe_result[:1000],
            "is_error": _is_error(exe_result),
            "done": bool(done),
            "info": info or {},
            "reward": reward,
        })

    def reset_task(self):
        """Drop per-task buffers (see ErrorLedgerV2.reset_task)."""
        self._steps = []
        self._task_consults = {}
        self._injected = []

    def finalize_task(self, task_id: str, success: bool):
        if not self._steps:
            self.reset_task()
            return

        self._append_audit(task_id, success, self._steps, self._injected)
        # Admission control: an errored step is written to cross-task memory ONLY
        # if it is a post-grounding-boundary failure (the command actually ran and
        # the tool/env pushed back). Pre-boundary failures -- the command never
        # compiled into a valid action (parser/serialization mangling) -- stay in
        # the audit above for visibility but never become a cross-task "tool risk".
        admitted = adapter = dropped = 0
        for step in self._steps:
            if not step.get("is_error"):
                continue
            if not is_admissible_error(step.get("exe_result", ""), step.get("action_sig", "")):
                dropped += 1                       # pre-grounding-boundary -> audit only
            elif self._upsert_error_note(step, task_id, success):
                admitted += 1                      # post-boundary, real action -> learned
            else:
                adapter += 1                       # post-boundary but adapter -> not learned
        if success:
            self._upsert_success_snippets(task_id)

        self._save_memory()
        logger.info(
            "ErrorLedgerV3: finalized %s success=%s steps=%d err_admit=%d err_adapter=%d err_drop=%d memory=%d injected=%d",
            task_id, success, len(self._steps), admitted, adapter, dropped, self.count(), len(self._injected)
        )
        self.reset_task()

    def retrieve(self, app: Optional[str], instruction: str = "", last_result: str = "",
                 recent_actions: Optional[list] = None, step_idx: int = 0,
                 obs: Optional[dict] = None, **kwargs) -> str:
        app = normalize_app(app)
        if not app:
            return ""
        # Exact action-trigger retrieval: memory surfaces ONLY when the agent just
        # invoked an action that is in the bank. No state-token similarity, no score
        # thresholds -- nothing to hand-tune, and the chrome-dominated cross-task
        # matching that caused the flooding is gone.
        recent_text = " ".join(a for a in (recent_actions or []) if a)
        if not recent_text:
            return ""

        notes = []
        notes.extend(self._retrieve_error_notes(app, recent_text))
        notes.extend(self._retrieve_success_snippets(app, recent_text))
        if not notes:
            return ""

        notes.sort(key=lambda item: item["score"], reverse=True)
        selected = []
        for item in notes:
            mid = item["id"]
            if self._task_consults.get(mid, 0) >= self.max_consults_per_task:
                continue
            self._task_consults[mid] = self._task_consults.get(mid, 0) + 1
            selected.append(item)
            self._injected.append({"step_idx": step_idx, "memory_id": mid,
                                   "kind": item.get("kind")})
            if len(selected) >= self.max_inject:
                break
        if not selected:
            return ""
        return self._format(selected)

    # ----- memory builders --------------------------------------------

    def _upsert_error_note(self, step: dict, task_id: str, success: bool) -> bool:
        """Write/refresh a cross-task error note. Returns True iff a note was
        actually persisted (False = skipped as an adapter/observe action), so the
        caller's err_admit count reflects real notes, not adapter-filtered ones."""
        pre = step.get("pre", {})
        action_sig = step.get("action_sig", "unknown")
        if _is_adapter_action(action_sig):
            return False  # never key memory on the auto-appended observe hook
        key = "ERR|{app}|{state}|{action}".format(
            app=step.get("app", "unknown"),
            state=pre.get("state_sig", "unknown"),
            action=action_sig,
        )
        notes = self.memory.setdefault("error_notes", {})
        note = notes.setdefault(key, {
            "id": key,
            "app": step.get("app", "unknown"),
            "state_sig": pre.get("state_sig", "unknown"),
            "state_tokens": pre.get("tokens", [])[:80],
            "title": pre.get("title", ""),
            "action_sig": action_sig,
            "failure_type": "execution_error",
            "example_result": "",
            "support_count": 0,
            "last_task_success": False,
            "last_seen_task": "",
        })
        note["support_count"] += 1
        note["example_result"] = step.get("exe_result", "")[:240]
        note["last_task_success"] = bool(success)
        note["last_seen_task"] = task_id
        return True

    def _upsert_success_snippets(self, task_id: str):
        # Split the trajectory into runs of consecutive real actions, then window
        # WITHIN each run. Waits are non-semantic pauses: dropped, but they do NOT
        # break a run (open_url -> wait -> click stays open_url -> click; the agent
        # waits reactively from the live screen anyway). Errors / terminals /
        # unrecognized actions DO break a run -- stitching across them would
        # fabricate an adjacency that was never taken (e.g. promoting a context-
        # dependent recovery step into the "always-do" procedure).
        runs: List[List[dict]] = []
        run: List[dict] = []
        for s in self._steps:
            sig = s.get("action_sig", "unknown")
            if s.get("is_error") or sig in _BREAK_SIGS:
                if run:
                    runs.append(run)
                    run = []
            elif sig in _WAIT_SIGS or _is_adapter_action(sig):
                continue            # drop non-semantic pause / auto-appended observe hook, keep the run going
            else:
                run.append(s)
        if run:
            runs.append(run)

        snippets = self.memory.setdefault("success_snippets", {})
        for seg in runs:
            for start in range(len(seg)):
                for length in (2, 3):
                    window = seg[start:start + length]
                    if len(window) < length:
                        continue
                    app = window[0].get("app", "unknown")
                    if any(s.get("app") != app for s in window):
                        continue
                    actions = [s.get("action_sig", "unknown") for s in window]
                    key = "SUC|{app}|{actions}".format(app=app, actions=">".join(actions))
                    pre = window[0].get("pre", {})
                    task_tokens = window[0].get("task_tokens", [])
                    snippet = snippets.setdefault(key, {
                        "id": key,
                        "app": app,
                        "action_sigs": actions,
                        "state_tokens": pre.get("tokens", [])[:80],
                        "task_tokens": task_tokens[:40],
                        "support_count": 0,
                        "last_seen_task": "",
                    })
                    snippet["support_count"] += 1
                    snippet["last_seen_task"] = task_id
                    # Keep a small union so retrieval can generalize without drifting too far.
                    merged_tokens = list(dict.fromkeys(
                        snippet.get("state_tokens", []) + pre.get("tokens", [])[:40]
                    ))
                    snippet["state_tokens"] = merged_tokens[:80]

    # ----- retrieval ---------------------------------------------------

    def _retrieve_error_notes(self, app: str, recent_text: str) -> List[dict]:
        # Aggregate failures by action_sig (the bank keys per state_sig; here we only
        # care that the agent JUST invoked an action that has failed before in this app).
        # Exact substring match on the high-level call -> no similarity, no threshold.
        by_action: Dict[str, dict] = {}
        for note in self.memory.get("error_notes", {}).values():
            if note.get("app") != app:
                continue
            sig = note.get("action_sig") or ""
            if not sig or sig == "unknown" or _is_adapter_action(sig) or sig not in recent_text:
                continue
            cur = by_action.get(sig)
            if cur is None or note.get("support_count", 0) > cur.get("support_count", 0):
                by_action[sig] = note
        results = []
        for sig, note in by_action.items():
            results.append({
                "id": "RISK|{}|{}".format(app, sig),
                "kind": "risk",
                # score is for ORDERING only (never compared to a cutoff): prefer risks,
                # then higher support.
                "score": 1.0 + min(note.get("support_count", 1), 3) * 0.1,
                "text": ("Risk: {action} has failed before here (e.g. \"{result}\"). "
                         "Verify it actually took effect before continuing.").format(
                    action=sig, result=(note.get("example_result") or "unknown")[:160]),
            })
        return results

    def _retrieve_success_snippets(self, app: str, recent_text: str) -> List[dict]:
        # Prefix trigger: if the agent JUST invoked the first action of a known-successful
        # short pattern, propose what came next. Exact match on the first action.
        results = []
        seen = set()
        for snip in self.memory.get("success_snippets", {}).values():
            if snip.get("app") != app:
                continue
            actions = snip.get("action_sigs") or []
            if len(actions) < 2:
                continue
            first = actions[0]
            if not first or first == "unknown" or _is_adapter_action(first) or first not in recent_text:
                continue
            key = ">".join(actions)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "id": "NEXT|{}|{}".format(app, key),
                "kind": "success",
                "score": 0.9 + min(snip.get("support_count", 1), 3) * 0.1,
                "text": ("After {first}, a past successful run did: {rest}. "
                         "Consider it as a next step (adapt arguments to the current task).").format(
                    first=first, rest=" -> ".join(actions[1:])),
            })
        return results

    def _format(self, items: List[dict]) -> str:
        lines = [
            "* Cross-task Trajectory Memory (v3):",
            "Use these notes only if they match the current screen and task. "
            "They are passive memories from prior rollouts, not guaranteed commands.",
        ]
        for item in items:
            lines.append(f"- {item['text']}")
        return "\n".join(lines)

    # ----- persistence -------------------------------------------------

    def _empty_memory(self) -> Dict[str, Any]:
        return {
            "error_notes": {},
            "success_snippets": {},
            "action_stats": {},
        }

    def _load_memory(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, encoding="utf-8") as f:
                    data = json.load(f)
                base = self._empty_memory()
                base.update(data if isinstance(data, dict) else {})
                return base
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"ErrorLedgerV3: failed to load {self.memory_path}: {e}")
        return self._empty_memory()

    def _save_memory(self):
        try:
            tmp_path = self.memory_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.memory_path)
        except IOError as e:
            logger.error(f"ErrorLedgerV3: failed to save {self.memory_path}: {e}")

    def _append_audit(self, task_id: str, success: bool, steps: List[dict],
                      injected: Optional[List[dict]] = None):
        try:
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "task_id": task_id,
                    "success": bool(success),
                    "steps": steps,
                    "injected": injected or [],
                }, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"ErrorLedgerV3: failed to append audit {self.audit_path}: {e}")


# ======================================================================
# factory
# ======================================================================

def make_error_ledger(path: str, version: str = "v1", max_inject: int = 1,
                      max_consults_per_task: int = 1, ttl_steps: int = 4):
    """v1 keeps its original defaults (max_inject=5, system injection)."""
    if version == "v2":
        return ErrorLedgerV2(path, max_inject=max_inject,
                             max_consults_per_task=max_consults_per_task,
                             ttl_steps=ttl_steps)
    if version == "v3":
        return ErrorLedgerV3(path, max_inject=max_inject,
                             max_consults_per_task=max_consults_per_task,
                             ttl_steps=ttl_steps)
    return ErrorLedger(path)
