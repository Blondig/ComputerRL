"""Uptake decomposition for stall-recovery fires (read-only).

For every StallRecovery fire found in runtime.log, answer three questions from
the sibling traj.jsonl:

  1. immediate uptake -- did the NEXT submitted action differ from the stalled
     action key (the model actually changed behavior on the intervened turn)?
  2. relapse -- did the stalled key come back within the following window?
  3. cross-fire -- when a task fired twice, is the second stall the SAME loop
     (same stalled key) returning, or a new one?

This splits "replan rescued 0/N" into two different diseases:
  - mostly unchanged      -> the model ignores advice; a hard output CONTRACT
                             (two-plan / forbidden-key regen) is the right lever.
  - mostly changed+failed -> the model complies but lacks information/ability;
                             skip two-plan, go to donor hints / stronger planner.

Usage (on the machine that has the result root):
  python analyze_stall_uptake.py \
      --run-root ~/computerrl_omni_rec2_replan [--domain libreoffice_impress]

Step-index mapping: the detector consumes the PREVIOUS turn's action, so signal
step_index k = traj step_num k closed the stalled window (its action is the
stalled key) and the intervention text was injected into step_num k+1's prompt.
"""

import argparse
import ast
import glob
import json
import os

from mm_agents.stall_recovery import action_key, pseudo_action

RELAPSE_WINDOW = 6


def parse_fires(runtime_path):
    fires = []
    try:
        handle = open(runtime_path, encoding="utf-8", errors="replace")
    except OSError:
        return fires
    with handle:
        for line in handle:
            marker = line.find("StallRecovery: {")
            if marker < 0:
                continue
            try:
                record = ast.literal_eval(line[marker + len("StallRecovery: "):].strip())
            except (ValueError, SyntaxError):
                continue
            if isinstance(record, dict) and "step_index" in record:
                fires.append(record)
    return fires


def load_actions(traj_path):
    """step_num -> submitted (pseudo) action key + display text."""
    actions = {}
    try:
        handle = open(traj_path, encoding="utf-8", errors="replace")
    except OSError:
        return actions
    with handle:
        for line in handle:
            try:
                raw = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            step = raw.get("step_num") or raw.get("step_idx")
            if not step:
                continue
            act = pseudo_action(str(raw.get("response") or "")) or str(raw.get("action") or "")
            actions[int(step)] = {
                "key": action_key(act),
                "text": " ".join(str(act).split())[:70],
            }
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--relapse-window", type=int, default=RELAPSE_WINDOW)
    args = parser.parse_args()

    rows = []
    for raw_root in args.run_root:
        root = os.path.abspath(os.path.expanduser(raw_root))
        for runtime_path in sorted(glob.glob(os.path.join(root, "**", "runtime.log"), recursive=True)):
            task_dir = os.path.dirname(runtime_path)
            domain = os.path.basename(os.path.dirname(task_dir))
            if args.domain and domain not in args.domain:
                continue
            fires = parse_fires(runtime_path)
            if not fires:
                continue
            actions = load_actions(os.path.join(task_dir, "traj.jsonl"))
            task_id = os.path.basename(task_dir)
            prev_stalled_key = None
            for fire in fires:
                k = int(fire["step_index"])
                stalled = actions.get(k, {"key": "", "text": "(missing)"})
                nxt = actions.get(k + 1)
                changed = None if nxt is None else (nxt["key"] != stalled["key"] and bool(nxt["key"]))
                relapse_steps = [
                    j for j in range(k + 2, k + 2 + args.relapse_window)
                    if actions.get(j, {}).get("key") == stalled["key"] and stalled["key"]
                ]
                rows.append({
                    "task": task_id,
                    "domain": domain,
                    "fire": fire.get("fire_index"),
                    "step": k,
                    "stalled": stalled["text"],
                    "next": "(episode ended)" if nxt is None else nxt["text"],
                    "changed": changed,
                    "relapse_at": relapse_steps[:3],
                    "same_loop_as_prev_fire": (
                        None if fire.get("fire_index", 1) == 1
                        else stalled["key"] == prev_stalled_key
                    ),
                })
                if fire.get("fire_index", 1) == 1:
                    prev_stalled_key = stalled["key"]

    if not rows:
        print("no StallRecovery fires found under the given roots")
        return

    print("=" * 100)
    print("Stall uptake decomposition  ({} fires, {} tasks)".format(
        len(rows), len({r["task"] for r in rows})))
    print("=" * 100)
    for r in rows:
        print("\n[{}] {} fire#{} @step {}".format(r["domain"], r["task"][:20], r["fire"], r["step"]))
        print("  stalled : {}".format(r["stalled"]))
        print("  next    : {}".format(r["next"]))
        print("  changed immediately: {}   relapse within {} steps: {}{}".format(
            {True: "YES", False: "NO", None: "n/a"}[r["changed"]],
            args.relapse_window,
            "YES at " + str(r["relapse_at"]) if r["relapse_at"] else "no",
            "" if r["same_loop_as_prev_fire"] is None else
            "   [same loop as fire#1: {}]".format("YES" if r["same_loop_as_prev_fire"] else "no"),
        ))

    judged = [r for r in rows if r["changed"] is not None]
    changed = [r for r in judged if r["changed"]]
    relapsed = [r for r in changed if r["relapse_at"]]
    second = [r for r in rows if r["same_loop_as_prev_fire"] is not None]
    same_loop = [r for r in second if r["same_loop_as_prev_fire"]]
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("  fires with a next step:            {}/{}".format(len(judged), len(rows)))
    print("  changed immediately:               {}/{}".format(len(changed), len(judged)))
    print("  ...of which relapsed to same key:  {}/{}".format(len(relapsed), len(changed)))
    print("  second fires that are the SAME loop returning: {}/{}".format(len(same_loop), len(second)))
    print()
    print("  reading: LOW changed  -> advice is ignored; a hard contract (forbid the")
    print("           stalled key + regenerate) is the right next lever (two-plan).")
    print("           HIGH changed but relapsed/failed -> compliance is not the problem;")
    print("           the model lacks information/ability -> donor hints / stronger planner.")


if __name__ == "__main__":
    main()
