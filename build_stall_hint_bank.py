"""Build the cross-task hint bank for --stall_recovery hint (Arm B).

Read-only over finished run roots: from every SUCCESSFUL episode it extracts
state-changing, error-free transitions (the same donor rule the retrieval
feasibility probe validated) and stores, per transition, everything the online
HintBank needs precomputed -- so the agent-side retrieval stays pure matching
with zero trajectory parsing at run time.

Entry fields (jsonl, one per transition):
  domain, app_family, task_id, instruction, instr_tokens,
  context_calls        call names leading into the transition (<=4, repeats collapsed)
  action_call          the donor's pseudo action (pre-grounding, truncated)
  reasoning            tail of the donor's non-code response text (the intent)
  continuation_calls   this + next pseudo actions (<=3, truncated; numbers are
                       masked again at format time -- coordinates never reach a hint)
  pre_thumb            32x18 grayscale thumbnail of the pre-transition screen
  step_num

Usage:
  python build_stall_hint_bank.py \
      --run-root ~/computerrl_rec2_gimp --run-root ~/computerrl_rec2_chrome \
      --out hint_bank.jsonl
"""

import argparse
import glob
import json
import os

from mm_agents.error_ledger import _is_error
from mm_agents.stall_recovery import (
    _tokens,
    action_key,
    call_name,
    normalize_app_family,
    pseudo_action,
    screenshot_thumbnail,
    thumb_similarity,
)

TERMINAL_ACTIONS = {"wait", "done", "fail"}


def load_instructions(task_roots):
    index = {}
    for root in task_roots:
        for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("instruction"):
                index[os.path.splitext(os.path.basename(path))[0]] = str(data["instruction"])
    return index


def load_steps(traj_path, task_dir):
    steps = []
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
            response = str(raw.get("response") or "")
            action = pseudo_action(response) or str(raw.get("action") or "")
            shot = str(raw.get("screenshot_file") or "")
            steps.append({
                "step_num": int(raw.get("step_num") or raw.get("step_idx") or len(steps) + 1),
                "response": response,
                "action": action,
                "key": action_key(action),
                "call": call_name(action),
                "is_error": _is_error(str(raw.get("exe_result") or "")),
                "screenshot": os.path.join(task_dir, shot) if shot else "",
            })
    return steps


def read_thumb(path, cache):
    if path in cache:
        return cache[path]
    thumb = None
    if path and os.path.isfile(path):
        try:
            with open(path, "rb") as handle:
                thumb = screenshot_thumbnail(handle.read())
        except OSError:
            thumb = None
    cache[path] = thumb
    return thumb


def collapse_repeats(seq):
    out = []
    for item in seq:
        if item and (not out or out[-1] != item):
            out.append(item)
    return out


def reasoning_excerpt(response, limit=240):
    import re
    text = re.sub(r"```(?:\w+\s+)?.*?```", " ", response, flags=re.DOTALL)
    text = re.sub(r"</?(?:think|answer)>", " ", text)
    text = " ".join(text.split())
    return text[-limit:] if text else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", action="append", required=True, help="Finished result root; repeatable.")
    parser.add_argument("--task-root", action="append", default=[],
                        help="Task-json roots for instructions (default: evaluation_examples/examples[+_office]).")
    parser.add_argument("--out", required=True, help="Output hint-bank jsonl.")
    parser.add_argument("--success-threshold", type=float, default=0.0)
    parser.add_argument("--state-change-threshold", type=float, default=0.995,
                        help="Adjacent-screenshot similarity BELOW this marks a state-changing donor step.")
    parser.add_argument("--context-length", type=int, default=4)
    parser.add_argument("--continuation-steps", type=int, default=3)
    parser.add_argument("--max-per-task", type=int, default=8)
    args = parser.parse_args()

    instructions = load_instructions(args.task_root or [
        "evaluation_examples/examples",
        "evaluation_examples/examples_office",
    ])

    entries, skipped, thumb_cache = [], {"fail_or_unknown": 0, "no_instruction": 0}, {}
    for raw_root in args.run_root:
        root = os.path.abspath(os.path.expanduser(raw_root))
        for result_path in sorted(glob.glob(os.path.join(root, "**", "result.txt"), recursive=True)):
            task_dir = os.path.dirname(result_path)
            task_id = os.path.basename(task_dir)
            domain = os.path.basename(os.path.dirname(task_dir))
            try:
                with open(result_path, encoding="utf-8", errors="replace") as handle:
                    score = float(handle.read().strip().split()[0])
            except (OSError, ValueError, IndexError):
                score = None
            if score is None or score <= args.success_threshold:
                skipped["fail_or_unknown"] += 1
                continue
            instruction = instructions.get(task_id, "")
            if not instruction:
                skipped["no_instruction"] += 1
                continue

            steps = load_steps(os.path.join(task_dir, "traj.jsonl"), task_dir)
            per_task = 0
            for pos in range(1, len(steps)):
                if per_task >= args.max_per_task:
                    break
                step = steps[pos]
                if (not step["key"] or step["is_error"]
                        or step["key"] in TERMINAL_ACTIONS
                        or step["call"].endswith(".exit")):
                    continue
                pre = read_thumb(steps[pos - 1]["screenshot"], thumb_cache)
                post = read_thumb(step["screenshot"], thumb_cache)
                sim = thumb_similarity(pre, post)
                if sim is None or sim >= args.state_change_threshold:
                    continue
                context = collapse_repeats(
                    [s["call"] for s in steps[max(0, pos - args.context_length):pos + 1]]
                )[-args.context_length:]
                continuation = [
                    s["action"][:120] for s in steps[pos:pos + args.continuation_steps] if s["action"]
                ]
                entries.append({
                    "domain": domain,
                    "app_family": normalize_app_family(domain),
                    "task_id": task_id,
                    "instruction": instruction,
                    "instr_tokens": sorted(_tokens(instruction)),
                    "context_calls": context,
                    "action_call": step["action"][:160],
                    "reasoning": reasoning_excerpt(step["response"]),
                    "continuation_calls": continuation,
                    "pre_thumb": list(pre),
                    "step_num": step["step_num"],
                })
                per_task += 1

    with open(args.out, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    by_family = {}
    for entry in entries:
        by_family.setdefault(entry["app_family"], set()).add(entry["task_id"])
    print("hint bank: {} entries -> {}".format(len(entries), args.out))
    for family in sorted(by_family):
        count = sum(1 for e in entries if e["app_family"] == family)
        print("  {:<14} {:>4} entries from {:>3} tasks".format(family, count, len(by_family[family])))
    print("skipped: {}".format(skipped))


if __name__ == "__main__":
    main()
