"""
Analyze the per-task effect of the cross-task error ledger (v1/v2/v3) vs a baseline.

It answers two questions the aggregate "+10%" cannot:
  1. Is the gap real or noise?  -> paired flip matrix + exact McNemar test, and an
     honest accounting of tasks that are MISSING in one run (env crashes/timeouts),
     which is a separate source of the delta.
  2. Is the gap *because of the memory*?  -> for the tasks v2 newly solved, cross-
     reference the ledger audit to see whether a relevant card was even available
     when that task ran. (Availability/relevance proxy -- NOT confirmed injection;
     the gate/instruction signals are not in the audit. See note at the end.)

Usage:
  python analyze_ledger_effect.py \
      --baseline /path/to/baseline_result_dir \
      --v2       /path/to/v2_result_dir \
      --domain   libreoffice_calc \
      --audit    /path/to/ledger.audit.jsonl        # optional (v2/v3 run)

You can point --baseline/--v2 at any level above the task dirs; result.txt is found
recursively. task_id = the task dir name; domain = its parent dir name.
"""
import argparse
import glob
import json
import os
from math import comb


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------

def _parse_score(path):
    try:
        raw = open(path).read().strip()
    except OSError:
        return None
    try:
        return float(raw)
    except ValueError:
        try:
            return float(eval(raw))   # tolerate "1.0\n" style leftovers like show_result.py
        except Exception:
            return None


def collect(root, domain_filter=None):
    """task_id -> {score, domain} for every result.txt under `root`."""
    out = {}
    for p in glob.glob(os.path.join(root, "**", "result.txt"), recursive=True):
        task_dir = os.path.dirname(p)
        task_id = os.path.basename(task_dir)
        domain = os.path.basename(os.path.dirname(task_dir))
        if domain_filter and domain != domain_filter:
            continue
        score = _parse_score(p)
        if score is None:
            continue
        out[task_id] = {"score": score, "domain": domain}
    return out


def load_instruction(eval_base, domain, task_id):
    cfg = os.path.join(eval_base, domain, f"{task_id}.json")
    try:
        return json.load(open(cfg)).get("instruction", "")
    except Exception:
        return ""


def load_audit(path):
    """Return (per_task, order) where per_task[task_id] holds availability proxies.

    Audit lines are appended in run order, so a card for (app, api_call) only exists
    for tasks that come AFTER the task that first errored on it.
    """
    per_task, order = {}, []
    carded = set()                      # (app, api_call) that have errored so far
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return {}, []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = d.get("task_id", "")
        steps = d.get("steps", []) or []
        apps = [s.get("app") for s in steps]
        app = max(set(apps), key=apps.count) if apps else "unknown"
        api_calls = {s.get("api_call") for s in steps if s.get("api_call")}
        err_calls = sorted({s.get("api_call") for s in steps if s.get("is_error")})

        # availability is judged BEFORE this task's own errors are added to the bank
        cards_for_app = sum(1 for (a, _c) in carded if a == app)
        relevant_hit = any((app, c) in carded for c in api_calls)

        per_task[tid] = {
            "n_steps": len(steps),
            "n_errors": sum(1 for s in steps if s.get("is_error")),
            "err_calls": err_calls,
            "cards_for_app_before": cards_for_app,
            "relevant_card_available": relevant_hit,
        }
        order.append(tid)
        for s in steps:
            if s.get("is_error") and s.get("api_call"):
                carded.add((app, s.get("api_call")))
    return per_task, order


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------

def mcnemar_exact_p(b, c):
    """Two-sided exact (binomial) McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="baseline result dir (no ledger)")
    ap.add_argument("--v2", required=True, help="v2/v3 result dir (with ledger)")
    ap.add_argument("--domain", default=None, help="restrict to one domain, e.g. libreoffice_calc")
    ap.add_argument("--audit", default=None, help="path to v2 ledger .audit.jsonl (optional)")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="success = score > threshold (default 0.0, matches finalize's result>0)")
    ap.add_argument("--eval-base", default="evaluation_examples/examples",
                    help="for loading task instructions (best effort)")
    args = ap.parse_args()

    base = collect(args.baseline, args.domain)
    v2 = collect(args.v2, args.domain)

    def ok(score):
        return score > args.threshold

    both = sorted(set(base) & set(v2))
    only_base_ran = sorted(set(base) - set(v2))
    only_v2_ran = sorted(set(v2) - set(base))

    print("=" * 72)
    print(f"Ledger effect: baseline vs v2   (domain={args.domain or 'ALL'}, success = score > {args.threshold})")
    print("=" * 72)
    print(f"tasks scored:   baseline={len(base)}   v2={len(v2)}   in BOTH={len(both)}")
    if only_base_ran or only_v2_ran:
        print(f"  !! MISSING/crashed (excluded from the paired test, but they move the raw average):")
        print(f"     only baseline has a result.txt: {len(only_base_ran)}  {only_base_ran or ''}")
        print(f"     only v2 has a result.txt:       {len(only_v2_ran)}  {only_v2_ran or ''}")
    if not both:
        print("\nNo overlapping tasks -- check the two dirs / --domain.")
        return

    # ---- aggregate on the paired set ----
    base_mean = sum(base[t]["score"] for t in both) / len(both)
    v2_mean = sum(v2[t]["score"] for t in both) / len(both)
    base_sr = sum(ok(base[t]["score"]) for t in both) / len(both)
    v2_sr = sum(ok(v2[t]["score"]) for t in both) / len(both)
    print(f"\nPaired set (n={len(both)}):")
    print(f"  mean score:   baseline={base_mean:.3f}   v2={v2_mean:.3f}   delta={v2_mean - base_mean:+.3f}")
    print(f"  success rate: baseline={base_sr*100:.1f}%   v2={v2_sr*100:.1f}%   "
          f"delta={(v2_sr - base_sr)*100:+.1f} pp ({round((v2_sr-base_sr)*len(both)):+d} tasks)")

    # ---- flip matrix ----
    both_pass = [t for t in both if ok(base[t]["score"]) and ok(v2[t]["score"])]
    both_fail = [t for t in both if not ok(base[t]["score"]) and not ok(v2[t]["score"])]
    base_only = [t for t in both if ok(base[t]["score"]) and not ok(v2[t]["score"])]   # regressions
    v2_only = [t for t in both if not ok(base[t]["score"]) and ok(v2[t]["score"])]     # improvements
    b, c = len(base_only), len(v2_only)
    print(f"\nFlip matrix:")
    print(f"  both pass: {len(both_pass):3d}   both fail: {len(both_fail):3d}")
    print(f"  v2 FIXED  (baseline fail -> v2 pass): {c:3d}   <- the improvements")
    print(f"  v2 BROKE  (baseline pass -> v2 fail): {b:3d}   <- the regressions")
    print(f"  net = {c - b:+d} tasks")
    p = mcnemar_exact_p(b, c)
    verdict = ("SIGNIFICANT" if p < 0.05 else
               "borderline" if p < 0.10 else
               "NOT distinguishable from noise")
    print(f"  McNemar exact two-sided p = {p:.3f}   ->  {verdict}  (discordant n={b+c})")

    # ---- per-task detail + memory-availability proxy ----
    audit, _order = load_audit(args.audit) if args.audit else ({}, [])

    def show(tasks, header):
        if not tasks:
            return
        print(f"\n{header}")
        for t in tasks:
            instr = load_instruction(args.eval_base, v2.get(t, base.get(t))["domain"], t)
            instr = (instr[:90] + "...") if len(instr) > 90 else instr
            line = f"  {t}  [base={base[t]['score']:.2f} v2={v2[t]['score']:.2f}]  {instr}"
            if t in audit:
                a = audit[t]
                rel = "RELEVANT-card-available" if a["relevant_card_available"] else "no-relevant-card"
                line += (f"\n        audit: steps={a['n_steps']} errors={a['n_errors']} "
                         f"cards_for_app_before={a['cards_for_app_before']} -> {rel}"
                         + (f" | errored on: {a['err_calls']}" if a["err_calls"] else ""))
            print(line)

    show(v2_only, ">>> v2 FIXED these (the source of the gain):")
    show(base_only, ">>> v2 BROKE these (regressions -- watch these):")

    if audit and v2_only:
        rel = sum(1 for t in v2_only if audit.get(t, {}).get("relevant_card_available"))
        print(f"\nMemory-availability proxy on the {c} fixed tasks: "
              f"{rel}/{c} had a directly relevant card available when they ran.")
        print("  NOTE: 'available' != 'injected' (the gate depends on the instruction/last-result,")
        print("        which the audit does not store) and 'injected' != 'caused'. Treat this as an")
        print("        UPPER BOUND on how many wins the memory could plausibly explain.")
        if rel == 0:
            print("  -> 0 relevant cards on the fixed tasks => the +gain is almost certainly NOT the memory.")
    elif not args.audit:
        print("\n(Pass --audit ledger.audit.jsonl to cross-reference the fixed tasks against the memory.)")


if __name__ == "__main__":
    main()
