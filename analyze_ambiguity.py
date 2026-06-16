"""
Matched-action outcome-dispersion (ambiguity) INFORMATION PROBE over existing v3
audits -- read-only, no re-run, no ledger/pipeline change, NO tuned parameters.
Ports ScreenSearch's ambiguity idea (arXiv:2605.16024) onto the
(pre, action_sig, post) transitions record_transition already logs passively.
NO new exploration, NO training, NO gate change.

DESIGN STANCE -- a pure probe, not a gate. We deliberately introduce none of the
knobs a working gate would need: no domain wordlist (cf. v2's Office-flavored
_PRECISION_KEYWORDS = sheet/cell/range...), no Jaccard/fuzzy-dedup threshold, no
kappa shrinkage synthesizing a single score, no injection cutoff. We report the
two RAW dimensions separately and let the reader judge:
  * normalized_entropy -- how unpredictable the outcome is (the ambiguity axis)
  * support_count      -- how much evidence backs that number (the confidence axis)
Collapsing these into one tuned `u` would smuggle in a "entropy vs sample-size"
tradeoff that belongs to gate DESIGN, not to "is there signal". So we don't.

GROUPING (source side). group = (app, action_sig). This is the granularity v3
retrieval actually conditions on: the write-key carries pre.state_sig but retrieval
aggregates by app+action (error_ledger.py:832 / :860), so v3's "state-conditioned
memory" is nominally present but inert. --state-cond ALSO reports
(app, pre.state_sig, action_sig) ONLY to show how sparse exact-hash conditioning is
(error_ledger.py:151); it is not proposed as a working key, and exact-hash dedup
must NOT be wired into an online gate -- that reintroduces thresholds/tuning.

OUTCOME (s' side). Default bucket = (post.app, is_error). We do NOT use
post.state_sig: it is an exact sha1, so nearly every successor is unique and the
entropy saturates -- it would measure UI-hash jitter, not aliasing.
  * is_error is intentionally in the bucket: "this action sometimes raises,
    sometimes succeeds here" is itself a reason to verify before committing.
  * For a within-app action post.app is ~constant, so the default bucket largely
    reflects error-rate variability. --use-title prints a SUPPLEMENTARY view that
    adds post.title[:60] (success-state dispersion); that view does NOT change the
    default-conclusion numbers in [1]-[4].

HARD LIMIT -- CORRELATIONAL, NOT CAUSAL. This can only tell us whether dispersion
carries information (do dispersed actions fail more / get injected more / recover
less after injection). It CANNOT show that gating injection on it beats the current
injection trigger -- that needs a fixed-policy A/B (entropy-gate vs the current
action-trigger policy). NB the v3 audit's trigger is exact action-match, NOT v2's
keyword gate. This answers "is there signal", not "does the gate help".

Usage:
  python analyze_ambiguity.py /path/to/v3x_result_root [more ...]
  python analyze_ambiguity.py --audit /path/to/ledger.v3.audit.jsonl
  # optional: --domain libreoffice_calc  --use-title  --state-cond  --top 25
"""
import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict


def _find_audits(roots):
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            files.extend(glob.glob(os.path.join(r, "**", "*.v3.audit.jsonl"), recursive=True))
    return sorted(set(files))


def _entropy_bits(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _spearman(xs, ys):
    """Stdlib Spearman rho (rank-Pearson) so we avoid a scipy/numpy dep."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0          # average rank for ties (1-based)
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def _outcome_key(step, use_title):
    post = step.get("post", {}) or {}
    parts = [str(post.get("app") or "unknown"), "err" if step.get("is_error") else "ok"]
    if use_title:
        parts.append((post.get("title") or "")[:60])
    return "|".join(parts)


def _group_key(step, state_cond):
    app = step.get("app") or (step.get("pre", {}) or {}).get("app") or "unknown"
    sig = step.get("action_sig", "unknown") or "unknown"
    if state_cond:
        return (app, (step.get("pre", {}) or {}).get("state_sig", "unknown"), sig)
    return (app, sig)


def _group_stats(out_counts):
    """group_key -> dict(n, distinct, p_max, h_norm). NO shrinkage, NO synthesis:
    h_norm (Pielou evenness = H / log2 distinct) and n are reported side by side."""
    stats = {}
    for g, ctr in out_counts.items():
        n = sum(ctr.values())
        distinct = len(ctr)
        h = _entropy_bits(ctr)
        stats[g] = {
            "n": n,
            "distinct": distinct,
            "p_max": (max(ctr.values()) / n) if n else float("nan"),
            "h_norm": (h / math.log2(distinct)) if distinct >= 2 else 0.0,
        }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--audit", action="append", default=[])
    ap.add_argument("--domain", default=None, help="restrict to tasks whose majority app == this")
    ap.add_argument("--use-title", action="store_true",
                    help="print a SUPPLEMENTARY (post.app,is_error,title) view; does not change [1]-[4]")
    ap.add_argument("--state-cond", action="store_true",
                    help="report (app,pre.state_sig,action) sparsity to show exact-hash conditioning is inert")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    audits = _find_audits(list(args.roots) + list(args.audit))
    if not audits:
        print("No *.v3.audit.jsonl found (these live on the run host, e.g. xpu80).")
        return

    out_counts = defaultdict(Counter)          # (app,action) -> Counter(outcome bucket)
    title_counts = defaultdict(Counter)        # same, title-augmented bucket (supplementary)
    state_counts = defaultdict(Counter)        # (app,state_sig,action) -> Counter (sparsity probe)
    occ = []                                   # per-step occurrence records
    n_tasks = 0

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
            n_tasks += 1
            success = bool(d.get("success"))
            steps = sorted(steps, key=lambda s: s.get("step_idx", 0))
            err_by_idx = {s.get("step_idx"): bool(s.get("is_error")) for s in steps}
            inj_idx = set(it.get("step_idx") for it in (d.get("injected", []) or []))

            for s in steps:
                g = _group_key(s, state_cond=False)
                out_counts[g][_outcome_key(s, use_title=False)] += 1
                if args.use_title:
                    title_counts[g][_outcome_key(s, use_title=True)] += 1
                if args.state_cond:
                    state_counts[_group_key(s, state_cond=True)][_outcome_key(s, use_title=False)] += 1
                idx = s.get("step_idx")
                occ.append({
                    "g": g,
                    "is_error": bool(s.get("is_error")),
                    "task_fail": (not success),
                    "injected": idx in inj_idx,
                    "next_err": err_by_idx.get((idx + 1) if isinstance(idx, int) else None),
                })

    if not occ:
        print("No transition steps found (check --domain).")
        return

    stats = _group_stats(out_counts)
    # per-group failure rates (note: dispersion != failure; a reliably-FAILING action
    # is deterministic -> low entropy, so we show err% beside entropy, never merged)
    agg = defaultdict(lambda: [0, 0, 0])       # g -> [n, step_fail, task_fail]
    for o in occ:
        a = agg[o["g"]]
        a[0] += 1
        a[1] += int(o["is_error"])
        a[2] += int(o["task_fail"])
    stepfail = {g: v[1] / v[0] for g, v in agg.items()}
    taskfail = {g: v[2] / v[0] for g, v in agg.items()}

    def rate(rs, key):
        return (sum(r[key] for r in rs) / len(rs) * 100) if rs else float("nan")

    # ---- [1] corpus summary ----
    print("=" * 80)
    print(f"Ambiguity information probe  domain={args.domain or 'ALL'}  "
          f"outcome=(post.app,is_error)")
    print("Two RAW axes, NOT synthesized: normalized_entropy (unpredictability) + support (evidence).")
    print("CORRELATIONAL ONLY: answers 'is there signal', NOT 'does an entropy-gate raise SR'.")
    print("=" * 80)
    print(f"[1] tasks={n_tasks}  step-occurrences={len(occ)}  groups (app,action)={len(stats)}  "
          f"step error-rate={sum(o['is_error'] for o in occ)/len(occ)*100:.1f}%")

    # ---- [2] cases ranked by dispersion (support shown, never merged in) ----
    rows = sorted(stats.items(), key=lambda kv: (kv[1]["h_norm"], kv[1]["n"]), reverse=True)
    print(f"\n[2] top-{args.top} (app, action) by normalized_entropy   [support shown separately]")
    print(f"    {'app':14s} {'action':30s} {'support':>7s} {'dist':>4s} {'pmax':>5s} {'normH':>5s} {'errR':>5s}")
    for g, d in rows[:args.top]:
        app, sig = g
        print(f"    {app[:14]:14s} {sig[:30]:30s} {d['n']:7d} {d['distinct']:4d} "
              f"{d['p_max']:5.2f} {d['h_norm']:5.2f} {stepfail[g]*100:4.0f}%")

    # ---- [3] does dispersion predict failure? (deterministic vs dispersed; no quantile knob) ----
    det = [o for o in occ if stats[o["g"]]["h_norm"] == 0.0]      # outcome was always the same here
    dis = [o for o in occ if stats[o["g"]]["h_norm"] > 0.0]       # same action, multiple outcomes
    print(f"\n[3] does dispersion predict failure?  (split: entropy==0 vs entropy>0, no threshold)")
    print(f"    dispersed (H>0)  occ={len(dis):5d}  step-err={rate(dis,'is_error'):4.0f}%  task-fail={rate(dis,'task_fail'):4.0f}%")
    print(f"    determ.   (H==0) occ={len(det):5d}  step-err={rate(det,'is_error'):4.0f}%  task-fail={rate(det,'task_fail'):4.0f}%")
    det_singletons = sum(1 for o in det if stats[o["g"]]["n"] == 1)
    print(f"    (determ. mixes truly-stable actions with {det_singletons} occ from single-observation "
          f"groups -- stability unknown; the Spearman below uses support>=2)")
    gs = [g for g in stats if stats[g]["n"] >= 2]                 # entropy is trivially 0 for singletons
    if len(gs) >= 3:
        hn = [stats[g]["h_norm"] for g in gs]
        print(f"    Spearman(normalized_entropy, group step-fail) = {_spearman(hn, [stepfail[g] for g in gs]):+.2f}")
        print(f"    Spearman(normalized_entropy, group task-fail) = {_spearman(hn, [taskfail[g] for g in gs]):+.2f}")
        print(f"    (over {len(gs)} groups with support>=2; positive => more-dispersed actions fail more)")

    # ---- [4] injection correlation (does the current injection trigger already track dispersion?) ----
    # injected.step_idx is the step whose action was generated WITH the note in context, so the
    # FIRST-order effect is that SAME step's outcome (is_error / task-fail); idx+1 is a secondary lens.
    inj = [o for o in occ if o["injected"]]
    print(f"\n[4] injection correlation   injected occurrences={len(inj)} ({len(inj)/len(occ)*100:.1f}%)")
    if inj:
        print(f"    injection rate:  dispersed={rate(dis,'injected'):.1f}%   deterministic={rate(det,'injected'):.1f}%")
        inj_dis = [o for o in inj if stats[o["g"]]["h_norm"] > 0.0]
        inj_det = [o for o in inj if stats[o["g"]]["h_norm"] == 0.0]

        def follow_ok(rs):                                       # secondary: FOLLOWING step (idx+1) avoided an error
            r = [o for o in rs if o["next_err"] is not None]     # drop last steps (no next)
            return (sum(not o["next_err"] for o in r) / len(r) * 100) if r else float("nan")
        print(f"    injected-step error (action built WITH the note):  "
              f"dispersed={rate(inj_dis,'is_error'):.0f}% (n={len(inj_dis)})   deterministic={rate(inj_det,'is_error'):.0f}% (n={len(inj_det)})")
        print(f"    injected-step task-fail:                           "
              f"dispersed={rate(inj_dis,'task_fail'):.0f}%   deterministic={rate(inj_det,'task_fail'):.0f}%")
        print(f"    following-step non-error (secondary, idx+1):       "
              f"dispersed={follow_ok(inj_dis):.0f}%   deterministic={follow_ok(inj_det):.0f}%")

    # ---- supplementary: title-augmented outcome view (NOT a default conclusion) ----
    if args.use_title and title_counts:
        tstats = _group_stats(title_counts)
        trows = sorted(tstats.items(), key=lambda kv: (kv[1]["h_norm"], kv[1]["n"]), reverse=True)
        print(f"\n[title-view] SUPPLEMENTARY: top-{args.top} by normalized_entropy with post.title in the bucket")
        print("    (adds success-state dispersion; shown for context, does NOT feed [1]-[4])")
        print(f"    {'app':14s} {'action':30s} {'support':>7s} {'dist':>4s} {'normH':>5s}")
        for g, d in trows[:args.top]:
            app, sig = g
            print(f"    {app[:14]:14s} {sig[:30]:30s} {d['n']:7d} {d['distinct']:4d} {d['h_norm']:5.2f}")

    # ---- optional: exact-state-conditioned sparsity (why pre.state_sig is inert) ----
    if args.state_cond and state_counts:
        ns = sorted(sum(c.values()) for c in state_counts.values())
        singles = sum(1 for n in ns if n == 1)
        print(f"\n[state-cond] (app, pre.state_sig, action) groups={len(ns)}  "
              f"singletons={singles} ({singles/len(ns)*100:.0f}%)  median support/group={ns[len(ns)//2]}")
        print("    ^ high singleton share => exact-hash state conditioning fragments support;")
        print("      this is why v3's state-keyed memory is nominally present but inert.")

    print("\nVERDICT GUIDE: dispersion is worth wiring into a gate later ONLY if [3] shows dispersed")
    print("actions clearly fail more (positive Spearman / split gap) AND [4] shows the current")
    print("injection trigger is NOT already firing where dispersion is high. If [3] is flat,")
    print("ScreenSearch's signal carries no usable information at this granularity -> leave it alone.")
    print("This is association only; confirm any change with a fixed-policy entropy-gate vs")
    print("current-trigger A/B. The probe itself adds no domain terms, no thresholds, no tuning.")


if __name__ == "__main__":
    main()
