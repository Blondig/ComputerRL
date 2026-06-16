"""
No-op / silent-failure PROBE over existing v3 audits -- read-only, no re-run, no
ledger/pipeline change. Tests the ONE thing a delta memory could add that the
current error-only ledger is blind to: a MUTATING action that returns Success but
leaves the observable state unchanged ("silent failure"), which the audit can
approximate because record_transition stores per-step pre/post `app_info_hash` and
`a11y_hash`.

HARD LIMITS (do not over-read the output):
  * The audit stores only HASHES of app_info[:5000] / a11y[:5000], not raw state.
    So we can only tell "did the hash change", never WHAT changed. A deep cell edit
    beyond 5000 chars can hash-collide to "no change" (false silent-failure); a UI
    focus/selection move can change a11y without changing the document. Hence the
    label is `hash_no_delta`, NOT `no-op`.
  * Only MUTATING actions count. Reads (get/print), saves/terminals (save/exit/wait),
    and navigation (go_to/select/open) legitimately leave the document hash unchanged,
    so counting them would manufacture fake silent-failures. Action role is inferred
    from the method verb (conservative: unmatched -> "other", excluded).

The verdict is PREDICTIVE, not volumetric: hash_no_delta only matters if it is
concentrated on specific mutating actions AND tasks that have it fail more often.
If it's diffuse or doesn't predict failure -> the error-only ledger already has
what little signal exists; don't build delta memory.

Usage:
  python analyze_noop_probe.py /path/to/v34_result_root [more ...]
  python analyze_noop_probe.py --audit /path/to/ledger.v3.audit.jsonl
  # optional --domain libreoffice_calc
"""
import argparse
import glob
import json
import os
from collections import defaultdict

# Action role from the method verb (action_sig = "Tool.method" or a bare token).
# Priority order matters; first bucket whose keyword the method starts with wins.
_ROLE_KEYWORDS = [
    ("terminal",   ("save", "wait", "exit", "done", "fail", "finish", "close")),
    ("read",       ("get", "read", "count", "list", "env", "print", "find", "check", "is_", "describe", "show")),
    ("navigation", ("go_to", "goto", "switch", "open", "select", "navigate", "scroll", "focus", "activate", "click")),
    ("mutating",   ("set", "write", "add", "insert", "delete", "remove", "merge", "sort", "rename",
                    "highlight", "replace", "create", "duplicate", "move", "clear", "fill", "type", "paste", "cut")),
]


def action_role(sig: str) -> str:
    method = (sig or "").split(".")[-1].strip().lower()
    if not method or method in ("unknown",):
        return "other"
    for role, kws in _ROLE_KEYWORDS:
        if any(method.startswith(k) for k in kws):
            return role
    return "other"


def _find_audits(roots):
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            files.extend(glob.glob(os.path.join(r, "**", "*.v3.audit.jsonl"), recursive=True))
    return sorted(set(files))


def _delta(pre, post):
    """(have_hashes, app_info_changed, a11y_changed). have_hashes=False if the
    audit predates hash storage (then we can't judge)."""
    pa, aa = pre.get("app_info_hash"), pre.get("a11y_hash")
    pb, ab = post.get("app_info_hash"), post.get("a11y_hash")
    if pa is None or aa is None or pb is None or ab is None:
        return False, None, None
    return True, (pa != pb), (aa != ab)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--audit", action="append", default=[])
    ap.add_argument("--domain", default=None, help="restrict to tasks whose majority app == this")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    audits = _find_audits(list(args.roots) + list(args.audit))
    if not audits:
        print("No *.v3.audit.jsonl found (these live on the run host, e.g. xpu80).")
        return

    role_steps = defaultdict(int)               # role -> #non-error steps
    mut_total = mut_no_delta = mut_a11y_only = 0
    mut_missing_hash = 0
    per_action = defaultdict(lambda: {"n": 0, "no_delta": 0})
    # task-level: among tasks with >=1 mutating Success step, did it have a no-delta one?
    task_has_mut = {}                           # tid -> bool (has >=1 mutating Success step)
    task_has_no_delta = defaultdict(bool)       # tid -> bool
    task_success = {}

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
            task_success[tid] = bool(d.get("success"))
            task_has_mut.setdefault(tid, False)

            for s in steps:
                if s.get("is_error"):
                    continue                    # errors are a different category
                role = action_role(s.get("action_sig", ""))
                role_steps[role] += 1
                if role != "mutating":
                    continue
                have, app_chg, a11y_chg = _delta(s.get("pre", {}) or {}, s.get("post", {}) or {})
                if not have:
                    mut_missing_hash += 1
                    continue
                mut_total += 1
                task_has_mut[tid] = True
                sig = s.get("action_sig", "unknown")
                per_action[sig]["n"] += 1
                no_delta = (not app_chg) and (not a11y_chg)
                if no_delta:
                    mut_no_delta += 1
                    per_action[sig]["no_delta"] += 1
                    task_has_no_delta[tid] = True
                elif a11y_chg and not app_chg:
                    mut_a11y_only += 1

    if mut_total == 0:
        print("No mutating Success steps with hashes found.")
        if mut_missing_hash:
            print(f"  ({mut_missing_hash} mutating steps lacked pre/post hashes -- audit predates hash storage.)")
        print("\nRole distribution of non-error steps:", dict(role_steps))
        return

    print("=" * 74)
    print(f"No-op / silent-failure probe  (domain={args.domain or 'ALL'})")
    print("LOW-CONFIDENCE: 'hash_no_delta' approximates a silent failure, it is NOT a confirmed no-op.")
    print("=" * 74)
    print("Non-error step roles:", {k: role_steps[k] for k in sorted(role_steps)})
    if mut_missing_hash:
        print(f"(skipped {mut_missing_hash} mutating steps with no stored hashes)")

    # ---- Block 1: overall rate among mutating Success steps ----
    print(f"\n[1] mutating Success steps: {mut_total}")
    print(f"    hash_no_delta (app_info AND a11y unchanged): {mut_no_delta}  ({mut_no_delta/mut_total*100:.1f}%)")
    print(f"    a11y-changed-only (focus/selection moved, doc maybe not): {mut_a11y_only}  "
          f"({mut_a11y_only/mut_total*100:.1f}%)  <- ambiguous, reported for context")

    # ---- Block 2: per-action no-delta rate ----
    print(f"\n[2] per mutating action (sorted by no_delta count):")
    rows = sorted(per_action.items(), key=lambda kv: kv[1]["no_delta"], reverse=True)
    print(f"    {'action':40s} {'n':>4s} {'no_delta':>8s} {'rate':>6s}")
    for sig, c in rows[:args.top]:
        if c["n"] == 0:
            continue
        print(f"    {sig:40s} {c['n']:4d} {c['no_delta']:8d} {c['no_delta']/c['n']*100:5.0f}%")

    # ---- Block 3: does it PREDICT task failure? (the actual verdict) ----
    tids = [t for t in task_has_mut if task_has_mut[t]]
    with_nd = [t for t in tids if task_has_no_delta.get(t)]
    without_nd = [t for t in tids if not task_has_no_delta.get(t)]

    def sr(ts):
        return (sum(task_success[t] for t in ts) / len(ts) * 100) if ts else float("nan")

    print(f"\n[3] task-level: among {len(tids)} tasks with >=1 mutating Success step")
    print(f"    with >=1 hash_no_delta mut step: {len(with_nd):3d}  success={sr(with_nd):.1f}%  fail={100-sr(with_nd):.1f}%")
    print(f"    with  0 hash_no_delta mut step:  {len(without_nd):3d}  success={sr(without_nd):.1f}%  fail={100-sr(without_nd):.1f}%")
    if with_nd and without_nd:
        gap = (100 - sr(with_nd)) - (100 - sr(without_nd))
        print(f"    failure-rate gap (no_delta minus clean): {gap:+.1f} pp")
        print("\nVERDICT GUIDE: build delta memory only if hash_no_delta is CONCENTRATED on a few")
        print("mutating actions (block 2) AND the failure-rate gap is clearly positive (block 3).")
        print("If diffuse or gap ~0/negative -> error-only ledger already captures the signal; skip.")


if __name__ == "__main__":
    main()
