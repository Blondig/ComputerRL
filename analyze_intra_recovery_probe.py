"""
Read-only probe for conservative intra-task recovery triggers.

The signal taxonomy follows common GUI-agent reflection practice rather than
application-specific failure fingerprints:

  * action_interface_failure: the generated action never crossed the action
    interface (parse/validation failure, or a grounded Python wrapper that did
    not compile). This is distinct from a tool that ran and returned an error.
  * repeated_action: the same full action text repeats consecutively.
  * repeated_state: several consecutive actions leave both stored app-info and
    accessibility hashes unchanged (a conservative proxy for repeated screens).
  * accumulated_failure: at least N errored actions in the latest W steps.
  * repeated_failure: the same normalized failure occurs at least N times in W.
  * confirmed_failure_loop: the same normalized failure reaches N occurrences
    inside W while the trajectory remains in one uninterrupted error streak.
    It fires once per streak, after evidence of failed self-recovery.

Low-level action runs are reported only as diagnostics. A click/type sequence is
normal in many GUI domains and is therefore never promoted to a recovery trigger
by this probe.

IMPORTANT COVERAGE LIMIT:
The current runner calls record_transition() only after env.step(). If the parser
returns no action, env.step() is skipped, so that attempt is absent from existing
v3 audits. The probe can identify recorded invalid/no-action rows and grounded
wrapper compile failures, but it cannot recover unrecorded parser failures from
audit alone. The output reports this limitation explicitly.

Usage:
  python analyze_intra_recovery_probe.py /path/to/v3_results [more roots ...]
  python analyze_intra_recovery_probe.py --audit /path/to/ledger.v3.audit.jsonl
  python analyze_intra_recovery_probe.py ROOT --domain libreoffice_calc
"""

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict


_COMPILE_ERROR_MARKERS = ("SyntaxError", "IndentationError", "TabError")
_LOWLEVEL_METHODS = {
    "click", "double_click", "right_click", "type", "write", "press",
    "hotkey", "key", "scroll", "drag", "drag_and_drop", "move_to", "moveto",
}
_TERMINAL_SIGS = {"WAIT", "DONE", "FAIL", "Agent.wait", "Agent.exit", ""}

_PRIMARY_SIGNALS = (
    "action_interface_failure",
    "confirmed_failure_loop",
    "repeated_action",
    "repeated_state",
    "accumulated_failure",
    "repeated_failure",
)
_DIAGNOSTIC_SIGNALS = ("lowlevel_run", "repeated_lowlevel_action")


def _find_audits(roots):
    files = []
    for root in roots:
        if os.path.isfile(root):
            files.append(root)
            continue
        for pattern in ("*.v3.audit.jsonl", "*.audit.jsonl"):
            files.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(set(files))


def _majority_app(steps):
    apps = [s.get("app") for s in steps if s.get("app")]
    return Counter(apps).most_common(1)[0][0] if apps else "unknown"


def _action_sig(step):
    return (step.get("action_sig") or step.get("api_call") or "").strip()


def _normalized_action(step):
    """Preserve arguments/targets while removing irrelevant whitespace.

    action_sig alone is too coarse: applying the same method to different GUI
    objects is often legitimate. action_text is the grounded action and keeps
    arguments, so exact repetition is a higher-precision stall signal.
    """
    text = str(step.get("action_text") or "").strip()
    if not text:
        text = str(step.get("response") or "").strip()
    if not text:
        text = _action_sig(step)
    return re.sub(r"\s+", " ", text)[:1000]


def _is_lowlevel(step):
    sig = _action_sig(step)
    tool, _, method = sig.partition(".")
    method = (method or tool).strip().lower()
    return tool in {"Agent", "pyautogui"} and method in _LOWLEVEL_METHODS


def _is_error(step):
    if "is_error" in step:
        return bool(step.get("is_error"))
    text = str(step.get("exe_result") or "").lower()
    return any(marker in text for marker in ("error", "traceback", "exception", "failed"))


def _error_template(text):
    text = str(text or "").strip()
    if not text:
        return "(empty)"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    picked = next(
        (line for line in reversed(lines) if re.search(r"(Error|Exception)\b", line)),
        lines[-1],
    )
    picked = re.sub(r"'[^']*'", "'X'", picked)
    picked = re.sub(r'"[^"]*"', '"X"', picked)
    picked = re.sub(r"0x[0-9a-fA-F]+", "0xN", picked)
    picked = re.sub(r"\b\d+\b", "N", picked)
    return re.sub(r"\s+", " ", picked)[:180]


def _state_key(state):
    state = state or {}
    app_hash = state.get("app_info_hash")
    a11y_hash = state.get("a11y_hash")
    if app_hash is None or a11y_hash is None:
        return None
    return app_hash, a11y_hash


def _compile_origin(step):
    """Classify a recorded compile-looking error without executing the action.

    Returns one of: parse_validation, dispatch_wrapper, execution_syntax,
    unknown_truncated, unknown_syntax, or None.
    """
    result = str(step.get("exe_result") or "")
    sig = _action_sig(step)
    action = str(step.get("action_text") or "").strip()

    if sig in {"", "unknown", "Parse error", "PARSE_ERROR"}:
        return "parse_validation"
    if "Invalid action" in result:
        return "parse_validation"
    if not any(marker in result for marker in _COMPILE_ERROR_MARKERS):
        return None
    if not action:
        return "unknown_syntax"
    # record_transition truncates action_text at 1000 characters. Compiling a
    # truncated command could manufacture a false wrapper failure.
    if len(action) >= 1000:
        return "unknown_truncated"
    if action in _TERMINAL_SIGS or action.startswith("{"):
        return "unknown_syntax"
    try:
        compile(action, "<recorded-agent-action>", "exec")
    except (SyntaxError, IndentationError, TabError):
        return "dispatch_wrapper"
    except Exception:
        return "unknown_syntax"
    return "execution_syntax"


def _is_interface_origin(origin):
    return origin in {"parse_validation", "dispatch_wrapper"}


def _crossed_interface(origin):
    """Whether the recorded action is known to have reached execution."""
    return origin is None or origin == "execution_syntax"


def _is_terminal_step(step):
    sig = _action_sig(step)
    action = str(step.get("action_text") or "").strip()
    return bool(sig and sig in _TERMINAL_SIGS) or bool(
        action and action in _TERMINAL_SIGS
    )


def _max_true_run(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _repeated_value_run(values, threshold):
    best = run = 0
    previous = object()
    for value in values:
        if value and value == previous:
            run += 1
        elif value:
            run = 1
        else:
            run = 0
        previous = value
        best = max(best, run)
    return best >= threshold


def _rolling_count(flags, window, threshold):
    if not flags:
        return False
    for end in range(len(flags)):
        start = max(0, end - window + 1)
        if sum(flags[start:end + 1]) >= threshold:
            return True
    return False


def _rolling_repeated(values, window, threshold):
    for end in range(len(values)):
        start = max(0, end - window + 1)
        counts = Counter(v for v in values[start:end + 1] if v)
        if counts and counts.most_common(1)[0][1] >= threshold:
            return True
    return False


def _interface_failure_events(steps, errors, origins):
    """Describe local self-recovery after each pre-execution failure."""
    events = []
    for index, origin in enumerate(origins):
        if not _is_interface_origin(origin):
            continue

        next_step = index + 1
        has_next = next_step < len(steps)
        next_dispatch = (
            has_next
            and not _is_terminal_step(steps[next_step])
            and _crossed_interface(origins[next_step])
        )
        clean_within_two = any(
            not _is_terminal_step(steps[candidate])
            and _crossed_interface(origins[candidate])
            and not errors[candidate]
            for candidate in range(index + 1, min(len(steps), index + 3))
        )
        events.append({
            "step_index": index,
            "origin": origin,
            "has_next": has_next,
            "next_dispatch": next_dispatch,
            "clean_within_two": clean_within_two,
        })
    return events


def _confirmed_failure_loop_events(errors, templates, origins, window, threshold):
    """Fire once when a repeated error is confirmed in a continuous streak.

    A non-error step resets the streak and permits a future trigger. This is
    deliberately stricter than repeated_failure: successful execution between
    errors is treated as evidence that the agent may still be making progress.
    """
    events = []
    streak_start = 0
    fired_in_streak = False

    for end, is_error in enumerate(errors):
        if not is_error:
            streak_start = end + 1
            fired_in_streak = False
            continue
        if fired_in_streak:
            continue

        start = max(streak_start, end - window + 1)
        counts = Counter(template for template in templates[start:end + 1] if template)
        repeated = [
            (template, count)
            for template, count in counts.items()
            if count >= threshold
        ]
        if not repeated:
            continue

        template, count = max(repeated, key=lambda item: (item[1], item[0]))
        matching = [
            index
            for index in range(start, end + 1)
            if templates[index] == template
        ]
        interface_count = sum(
            _is_interface_origin(origins[index]) for index in matching
        )
        if interface_count == len(matching):
            failure_stage = "interface"
        elif interface_count == 0:
            failure_stage = "execution"
        else:
            failure_stage = "mixed"

        events.append({
            "step_index": end,
            "template": template,
            "occurrences": count,
            "failure_stage": failure_stage,
            "failed_steps_before_trigger": end - streak_start + 1,
        })
        fired_in_streak = True

    return events


def analyze_task(record, repeat_threshold, window, failure_threshold):
    steps = sorted(record.get("steps", []) or [], key=lambda s: s.get("step_idx", 0))
    actions = [_normalized_action(step) for step in steps]
    errors = [_is_error(step) for step in steps]
    templates = [
        _error_template(step.get("exe_result")) if is_error else ""
        for step, is_error in zip(steps, errors)
    ]
    lowlevel = [_is_lowlevel(step) for step in steps]

    stable_transitions = []
    state_hash_steps = 0
    for step in steps:
        pre = _state_key(step.get("pre"))
        post = _state_key(step.get("post"))
        have = pre is not None and post is not None
        state_hash_steps += int(have)
        stable_transitions.append(have and pre == post)

    step_origins = [_compile_origin(step) for step in steps]
    origins = Counter(origin for origin in step_origins if origin)
    interface_events = _interface_failure_events(steps, errors, step_origins)
    loop_events = _confirmed_failure_loop_events(
        errors, templates, step_origins, window, failure_threshold
    )

    for event in loop_events:
        index = event["step_index"]
        event["clean_within_two"] = any(
            not _is_terminal_step(steps[candidate])
            and _crossed_interface(step_origins[candidate])
            and not errors[candidate]
            for candidate in range(index + 1, min(len(steps), index + 3))
        )

    interface_failure = bool(interface_events)
    confirmed_failure_loop = bool(loop_events)
    repeated_action = _repeated_value_run(actions, repeat_threshold)
    repeated_state = _max_true_run(stable_transitions) >= repeat_threshold
    accumulated_failure = _rolling_count(errors, window, failure_threshold)
    repeated_failure = _rolling_repeated(templates, window, failure_threshold)
    lowlevel_run = _max_true_run(lowlevel) >= repeat_threshold
    repeated_lowlevel = any(
        lowlevel[i] and actions[i] and
        _repeated_value_run(actions[max(0, i - repeat_threshold + 1):i + 1], repeat_threshold)
        for i in range(len(steps))
    )

    signals = {
        "action_interface_failure": interface_failure,
        "confirmed_failure_loop": confirmed_failure_loop,
        "repeated_action": repeated_action,
        "repeated_state": repeated_state,
        "accumulated_failure": accumulated_failure,
        "repeated_failure": repeated_failure,
        "lowlevel_run": lowlevel_run,
        "repeated_lowlevel_action": repeated_lowlevel,
    }
    return {
        "steps": len(steps),
        "state_hash_steps": state_hash_steps,
        "signals": signals,
        "origins": origins,
        "interface_events": interface_events,
        "loop_events": loop_events,
        "action_preview": " > ".join(_action_sig(s) or "unknown" for s in steps)[:500],
        "error_preview": " | ".join(t for t in templates if t)[:500],
    }


def _rate(successes, count):
    return successes / count * 100 if count else math.nan


def _fmt_rate(value):
    return "   -" if math.isnan(value) else f"{value:5.1f}%"


def _print_recovery_events(tasks):
    interface_rows = [
        (task, event)
        for task in tasks
        for event in task["analysis"]["interface_events"]
    ]
    interface_tasks = [
        task for task in tasks if task["analysis"]["interface_events"]
    ]
    next_observed = sum(event["has_next"] for _, event in interface_rows)
    next_dispatch = sum(event["next_dispatch"] for _, event in interface_rows)
    interface_clean = sum(event["clean_within_two"] for _, event in interface_rows)
    interface_task_sr = _rate(
        sum(task["success"] for task in interface_tasks), len(interface_tasks)
    )

    loop_rows = [
        (task, event)
        for task in tasks
        for event in task["analysis"]["loop_events"]
    ]
    loop_tasks = [task for task in tasks if task["analysis"]["loop_events"]]
    loop_clean = sum(event["clean_within_two"] for _, event in loop_rows)
    failed_before = sum(
        event["failed_steps_before_trigger"] for _, event in loop_rows
    )
    loop_task_sr = _rate(sum(task["success"] for task in loop_tasks), len(loop_tasks))
    stages = Counter(event["failure_stage"] for _, event in loop_rows)

    print("\n[RECOVERY EVENT ANALYSIS -- local opportunities, not causal effects]")
    if interface_rows:
        print(
            "  interface failure: "
            f"events={len(interface_rows)} tasks={len(interface_tasks)} "
            f"unexecuted_attempts={len(interface_rows)} task_success={_fmt_rate(interface_task_sr)}"
        )
        print(
            "    next recorded action crossed interface: "
            f"{next_dispatch}/{next_observed} "
            f"({_fmt_rate(_rate(next_dispatch, next_observed))})"
        )
        print(
            "    non-error, non-terminal execution within 2 recorded steps: "
            f"{interface_clean}/{len(interface_rows)} "
            f"({_fmt_rate(_rate(interface_clean, len(interface_rows)))})"
        )
    else:
        print("  interface failure: events=0")

    if loop_rows:
        print(
            "  confirmed failure loop: "
            f"events={len(loop_rows)} tasks={len(loop_tasks)} "
            f"stages={dict(stages)} task_success={_fmt_rate(loop_task_sr)}"
        )
        print(
            "    failed steps consumed before trigger: "
            f"total={failed_before} avg={failed_before / len(loop_rows):.1f}"
        )
        print(
            "    non-error, non-terminal execution within 2 recorded steps: "
            f"{loop_clean}/{len(loop_rows)} "
            f"({_fmt_rate(_rate(loop_clean, len(loop_rows)))})"
        )
    else:
        print("  confirmed failure loop: events=0")


def print_group(name, tasks, top):
    count = len(tasks)
    successes = sum(t["success"] for t in tasks)
    base_sr = _rate(successes, count)
    step_count = sum(t["analysis"]["steps"] for t in tasks)
    hash_steps = sum(t["analysis"]["state_hash_steps"] for t in tasks)
    origin_counts = Counter()
    for task in tasks:
        origin_counts.update(task["analysis"]["origins"])

    print("\n" + "=" * 100)
    print(f"{name}: tasks={count}  success={_fmt_rate(base_sr)}  executed_steps={step_count}")
    coverage = _rate(hash_steps, step_count)
    print(f"state-hash coverage: {hash_steps}/{step_count} ({_fmt_rate(coverage)})")
    print("compile/error origin observations:", dict(origin_counts) or "(none)")
    print("NOTE: parser attempts that produced actions=[] are absent from current audits.")

    _print_recovery_events(tasks)

    print("\n[TASK-LEVEL SIGNAL ASSOCIATIONS -- descriptive, not causal]")
    print(f"  {'signal':30s} {'tasks':>7s} {'task%':>7s} {'succ':>7s} {'succ(no)':>9s} {'fail-gap':>9s}")
    for signal in _PRIMARY_SIGNALS:
        with_signal = [t for t in tasks if t["analysis"]["signals"][signal]]
        without = [t for t in tasks if not t["analysis"]["signals"][signal]]
        with_sr = _rate(sum(t["success"] for t in with_signal), len(with_signal))
        without_sr = _rate(sum(t["success"] for t in without), len(without))
        gap = ((100 - with_sr) - (100 - without_sr)
               if not math.isnan(with_sr) and not math.isnan(without_sr) else math.nan)
        gap_text = "-" if math.isnan(gap) else f"{gap:+.1f}pp"
        print(f"  {signal:30s} {len(with_signal):7d} "
              f"{_rate(len(with_signal), count):6.1f}% {_fmt_rate(with_sr):>7s} "
              f"{_fmt_rate(without_sr):>9s} {gap_text:>9s}")

    print("\n[DIAGNOSTIC ONLY -- never a positive recovery trigger by itself]")
    for signal in _DIAGNOSTIC_SIGNALS:
        selected = [t for t in tasks if t["analysis"]["signals"][signal]]
        sr = _rate(sum(t["success"] for t in selected), len(selected))
        print(f"  {signal:30s} tasks={len(selected):3d}  success={_fmt_rate(sr)}")

    print("\n[PRIMARY-SIGNAL COMBINATIONS]")
    combinations = defaultdict(list)
    for task in tasks:
        active = tuple(s for s in _PRIMARY_SIGNALS if task["analysis"]["signals"][s])
        combinations[active or ("none",)].append(task)
    for active, members in sorted(combinations.items(), key=lambda item: len(item[1]), reverse=True):
        sr = _rate(sum(t["success"] for t in members), len(members))
        print(f"  {len(members):4d} tasks  success={_fmt_rate(sr)}  {' + '.join(active)}")

    print("\n[OVERLAP MATRIX: task counts]")
    print(" " * 32 + " ".join(f"{s[:8]:>8s}" for s in _PRIMARY_SIGNALS))
    for left in _PRIMARY_SIGNALS:
        cells = []
        for right in _PRIMARY_SIGNALS:
            cells.append(sum(
                t["analysis"]["signals"][left] and t["analysis"]["signals"][right]
                for t in tasks
            ))
        print(f"  {left:28s} " + " ".join(f"{v:8d}" for v in cells))

    print("\n[SAMPLES]")
    for signal in _PRIMARY_SIGNALS + _DIAGNOSTIC_SIGNALS:
        samples = [t for t in tasks if t["analysis"]["signals"][signal]][:top]
        if not samples:
            continue
        print(f"  {signal}:")
        for task in samples:
            print(f"    {task['instance_id']} success={int(task['success'])} "
                  f"actions={task['analysis']['action_preview']}")
            if task["analysis"]["error_preview"]:
                print(f"      errors={task['analysis']['error_preview']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roots", nargs="*", help="result roots or audit files")
    parser.add_argument("--audit", action="append", default=[], help="explicit audit file; repeatable")
    parser.add_argument("--domain", default=None, help="restrict to majority app/domain")
    parser.add_argument("--window", type=int, default=5, help="recent-step window for accumulated signals")
    parser.add_argument("--threshold", type=int, default=3, help="failures required inside --window")
    parser.add_argument("--repeat", type=int, default=3, help="consecutive repeats required")
    parser.add_argument("--top", type=int, default=3, help="sample task instances per signal")
    args = parser.parse_args()

    audits = _find_audits(list(args.roots) + list(args.audit))
    if not audits:
        print("No *.v3.audit.jsonl / *.audit.jsonl found.")
        return
    if args.window < 1 or args.threshold < 1 or args.repeat < 2:
        parser.error("--window/--threshold must be >=1 and --repeat must be >=2")
    if args.threshold > args.window:
        parser.error("--threshold cannot exceed --window")

    tasks = []
    malformed_lines = 0
    for audit_path in audits:
        run_name = os.path.basename(os.path.dirname(audit_path)) or os.path.basename(audit_path)
        with open(audit_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                steps = record.get("steps", []) or []
                domain = _majority_app(steps)
                if args.domain and domain != args.domain:
                    continue
                task_id = record.get("task_id") or f"line-{line_no}"
                tasks.append({
                    "instance_id": f"{run_name}:{task_id}",
                    "domain": domain,
                    "success": bool(record.get("success")),
                    "analysis": analyze_task(record, args.repeat, args.window, args.threshold),
                })

    if not tasks:
        print("No task records matched the requested domain.")
        return

    print("=" * 100)
    print("Intra-task recovery trigger probe (READ-ONLY)")
    print(f"audits={len(audits)} task_instances={len(tasks)} malformed_lines={malformed_lines}")
    print(f"window={args.window} failure_threshold={args.threshold} repeat_threshold={args.repeat}")
    print("Candidate recovery events are action-interface failures and confirmed failure loops;")
    print("broader repetition/low-level signals remain descriptive diagnostics only.")
    print("No parser, Omni, ledger, or runner behavior is changed.")

    by_domain = defaultdict(list)
    for task in tasks:
        by_domain[task["domain"]].append(task)
    for domain in sorted(by_domain):
        print_group(domain, by_domain[domain], args.top)
    if len(by_domain) > 1:
        print_group("ALL DOMAINS", tasks, args.top)


if __name__ == "__main__":
    main()
