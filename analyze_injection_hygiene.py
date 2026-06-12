"""
Injection-side memory hygiene: what does the v3 ledger actually SURFACE, and is
it semantically load-bearing? Read-only over existing *.v3.audit.jsonl -- no
re-run, no code-behavior change. Answers the two questions raised by the
70bca0cc regression post-mortem:

  1. RISK notes on ADAPTER/INSTRUMENTATION anchors (e.g. `print_result`, which
     grounding_agent.tool_commands auto-appends to EVERY command -- the agent
     never authors it). These can't carry transferable task knowledge.
  2. NEXT (success-snippet) notes anchored on LOW-SELECTIVITY actions (e.g.
     `go_to_slide`) that appear in almost every task -> the anchor over-triggers
     and injects task-irrelevant "what a past run did next" noise.

For every injected memory we parse its id:
    RISK|app|ANCHOR                       (error_note)
    NEXT|app|FIRST>rest                   (success_snippet; FIRST = anchor)
and report, per anchor: how many tasks it was injected into, the success rate of
those tasks, and a selectivity figure (distinct tasks injected / total tasks) so
a near-ubiquitous anchor stands out. Instrumentation anchors are flagged by the
ONE generic signal we trust -- the method name `print_result` that tool_commands
appends -- NOT by an app/tool allowlist.

Usage:
  python analyze_injection_hygiene.py /path/to/v33_result_root [more ...]
  python analyze_injection_hygiene.py --audit /path/to/ledger.v3.audit.jsonl
  # optional --domain libreoffice_impress (matches the majority app of a task)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

# the only "this is an adapter action" signal we hardcode is the wrapper method
# grounding_agent.tool_commands appends to every command. Generic across tools.
_ADAPTER_METHODS = ("print_result",)


def _find_audits(roots):
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            files.extend(glob.glob(os.path.join(r, "**", "*.v3.audit.jsonl"), recursive=True))
    return sorted(set(files))


def _anchor(mid):
    """(kind, anchor_action) from a memory id, or (None, None)."""
    parts = (mid or "").split("|")
    if len(parts) < 3:
        return None, None
    kind, chain = parts[0], parts[2]
    if kind == "RISK":
        return "RISK", chain
    if kind == "NEXT":
        return "NEXT", chain.split(">")[0]   # first action = the trigger anchor
    return kind, chain


def _is_adapter(anchor):
    return any(anchor.endswith("." + m) or anchor == m for m in _ADAPTER_METHODS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--audit", action="append", default=[])
    ap.add_argument("--domain", default=None, help="restrict to tasks whose majority app == this")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    audits = _find_audits(list(args.roots) + list(args.audit))
    if not audits:
        print("No *.v3.audit.jsonl found (these live on the run host, e.g. xpu80).")
        return

    # per (kind, anchor): set of tasks injected, and successes among them
    inj = defaultdict(lambda: {"tasks": set(), "succ": set()})
    n_tasks = n_inj_tasks = 0

    for path in audits:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps = d.get("steps", []) or []
            if args.domain:
                apps = [s.get("app") for s in steps if s.get("app")]
                if not apps or max(set(apps), key=apps.count) != args.domain:
                    continue
            tid = d.get("task_id", "")
            success = bool(d.get("success"))
            n_tasks += 1
            injected = d.get("injected", []) or []
            if injected:
                n_inj_tasks += 1
            seen = set()
            for it in injected:
                kind, anchor = _anchor(it.get("memory_id"))
                if not anchor or (kind, anchor) in seen:
                    continue
                seen.add((kind, anchor))
                rec = inj[(kind, anchor)]
                rec["tasks"].add(tid)
                if success:
                    rec["succ"].add(tid)

    if not n_tasks:
        print("No tasks matched (check --domain).")
        return

    print(f"tasks audited: {n_tasks}   tasks with >=1 injection: {n_inj_tasks}\n")

    rows = []
    for (kind, anchor), rec in inj.items():
        t = len(rec["tasks"])
        s = len(rec["succ"])
        rows.append((kind, anchor, t, s, t / n_tasks, _is_adapter(anchor)))
    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"{'kind':5s} {'anchor action':38s} {'#tasks':>6s} {'succ':>5s} {'sel%':>6s}  flag")
    print("-" * 78)
    for kind, anchor, t, s, sel, adapter in rows[:args.top]:
        flag = "ADAPTER(drop?)" if adapter else ("low-sel(noisy?)" if kind == "NEXT" and sel >= 0.5 else "")
        sr = f"{s}/{t}"
        print(f"{kind:5s} {anchor:38s} {t:6d} {sr:>5s} {sel*100:5.0f}%  {flag}")

    adapter_risk = [r for r in rows if r[0] == "RISK" and r[5]]
    if adapter_risk:
        n = sum(r[2] for r in adapter_risk)
        print(f"\n>> ADAPTER RISK notes (e.g. print_result): injected into {n} task-instances "
              f"-> semantically empty, candidates to exclude (key memory on agent-authored actions only).")
    nav = [r for r in rows if r[0] == "NEXT" and not r[5] and r[4] >= 0.5]
    if nav:
        print(f">> Low-selectivity NEXT anchors (appear in >=50% of tasks): "
              f"{[r[1] for r in nav]} -> over-trigger; gate by anchor selectivity, not by a navigation allowlist.")


if __name__ == "__main__":
    main()
