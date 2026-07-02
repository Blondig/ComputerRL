"""
Divergence-point probe -- the decisive test for the "shared-prefix opportunity".

Motivation
----------
The goal/subgoal gap probe flagged pass/fail pairs that share an early subgoal
prefix and then diverge. The open question (raised as a possible OPPORTUNITY):
could the successful trajectory's continuation, injected at the fork, guide the
failing one? That only helps if the failure at the fork is a genuine WRONG CHOICE
the agent could act on -- NOT a serialization collapse (base-bug) it already
tried, and NOT a stall where there is no next action to give.

So this script answers two things, both read-only:

  Part A -- base-bug vs execution error.
    Apply the existing STRUCTURAL classifier `error_ledger.classify_error_step`
    to every errored step in the run(s):
      representation -> base-agent serialization collapse (SyntaxError/... ; the
                        python\\n / content-block mangle) -- pre-boundary, DROP.
      no_action      -> parser produced no executable action -- pre-boundary.
      execution      -> command ran, env/tool pushed back -- REAL feedback.
    This sizes how much of the failure mass is base-bug noise vs real error, and
    is the prerequisite observation for reading Part B honestly.

  Part B -- divergence attribution.
    For each FAIL, find the same-goal PASS with the longest shared prefix (at
    L2 / typed-operation granularity by default -- L1 role prefix is too coarse
    to mean "same path"), locate the divergence step, and bucket what happened
    there in the failing run:
      A_representation      -> a base-bug mangle collapses the tail -> pass-demo
                               is REDUNDANT (agent tried it; C+B recovery owns it).
      B_stall               -> error/lowlevel loop or no core action after the
                               fork -> there is no "next action" to demonstrate.
      C_wrong_choice        -> a valid, non-error core action that differs from
                               the successful path -> the ONLY bucket a fork-time
                               pass-demo could plausibly fix.
        .silent             -> no env error after the wrong choice.
        .feedback_resistant -> got a real execution error afterward and still did
                               not recover (memory unlikely to teach what a clear
                               error did not).

    Headline = size of the C bucket among fails that shared a REAL prefix. If it
    is ~0, the shared-prefix "opportunity" is empty and the line closes cleanly.

Anti-overfit discipline: this reuses the same swappable adapters as the gap probe
(`l1_for_call`/`l2_for_call`) and the structural `classify_error_step`; it never
keys on a tool/app name. Matching is on role/typed-op SEQUENCES only.

Usage:
  python analyze_divergence_probe.py \
    --run-root /path/to/results_run \
    --baseline-root /path/to/results_baseline \
    --match l2 --min-prefix 2 \
    --out-dir logs/divergence_probe
"""
import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from analyze_goal_subgoal_gap import (
    ManifestEntry, Observation, ObsRow, StepEvent, TaskInfo,
    CORE_L1, LOWLEVEL_L1,
    build_rows, collect_files, discover_ledger_sidecars, ensure_dir,
    extract_calls_from_text,
    load_api_descriptions, load_audit_observations, load_tasks, max_run,
    merge_runtime_logs, merge_traj_observations, ordered_steps, parse_scores,
    write_jsonl,
)
from mm_agents.error_ledger import classify_error_step


# ---------------------------------------------------------------------------
# Loading (mirrors analyze_goal_subgoal_gap.main so scores/labels are identical)


def load_rows(args) -> Tuple[List[ObsRow], Dict[str, TaskInfo], List[ManifestEntry]]:
    task_roots = args.task_root or [
        "evaluation_examples/examples",
        "evaluation_examples/examples_office",
    ]
    schema_roots = args.schema_root or ["mm_agents/autoglm_v/tools/apis"]
    run_roots = args.run_root

    manifest: List[ManifestEntry] = []
    tasks = load_tasks(task_roots)
    api_desc = load_api_descriptions(schema_roots)

    discovered_audits, _banks = discover_ledger_sidecars(run_roots, manifest)
    audit_files = collect_files(run_roots, args.audit + discovered_audits, ["*.v3.audit.jsonl", "*.audit.jsonl"])
    traj_files = collect_files(run_roots, args.traj, ["traj.jsonl"])
    log_files = collect_files(run_roots, args.log, ["runtime.log", "*.log", "debug*.log", "sdebug*.log", "normal*.log"])

    observations = load_audit_observations(audit_files, run_roots, api_desc, manifest)
    merge_traj_observations(observations, traj_files, run_roots, api_desc, manifest)
    merge_runtime_logs(observations, log_files, run_roots, api_desc, manifest)

    scores = parse_scores(run_roots)
    baseline_scores = parse_scores(args.baseline_root)
    baseline_by_tid = {tid: score for (_run, tid), score in baseline_scores.items()}
    for key, obs in observations.items():
        if key in scores:
            obs.score = scores[key]
        else:
            cands = [s for (_r, tid), s in scores.items() if tid == obs.task_id]
            if len(cands) == 1:
                obs.score = cands[0]
        obs.baseline_score = baseline_scores.get(key, baseline_by_tid.get(obs.task_id))

    return build_rows(observations, tasks), tasks, manifest


# ---------------------------------------------------------------------------
# Per-step error classification (base-bug vs execution)


def step_exec_sig(ev: StepEvent) -> str:
    """The action that ACTUALLY crossed the execution interface -- used for error
    classification only. Prefer the executed-side signal (audit action_sig, then
    the grounded action string) over the response-parsed subgoal call: a parse
    failure whose *response* merely mentions a tool must NOT be scored as a real
    execution error. Response-parsed subgoals stay for the sequence (Part B)."""
    sig = str(ev.v3_action_sig or "").strip()
    if sig and sig != "unknown":
        return sig
    for text in (ev.raw_action, ev.action_text):
        calls = extract_calls_from_text(text, prefer_answer=False)
        if calls:
            return calls[0]
    return "unknown"


def step_error_class(ev: StepEvent) -> Optional[str]:
    if not ev.is_error:
        return None
    return classify_error_step(ev.exe_result or "", step_exec_sig(ev))


def error_class_distribution(rows: Sequence[ObsRow]) -> Dict[str, Counter]:
    """domain -> Counter{representation,no_action,execution, err_steps, fail_tasks,
    fail_tasks_with_representation}."""
    by_domain: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        dom = row.task.domain or "unknown"
        c = by_domain[dom]
        is_fail = row.success is False
        c["tasks"] += 1
        c["fail_tasks"] += int(is_fail)
        task_classes = set()
        for ev in ordered_steps(row.obs):
            kls = step_error_class(ev)
            if kls is None:
                continue
            c[kls] += 1
            c["err_steps"] += 1
            task_classes.add(kls)
        if is_fail and "representation" in task_classes:
            c["fail_tasks_with_representation"] += 1
        if is_fail and task_classes and task_classes <= {"execution"}:
            c["fail_tasks_execution_only"] += 1
    return by_domain


# ---------------------------------------------------------------------------
# Aligned subgoal units (same filtering as gap probe, but keep the step ref)


class Unit:
    __slots__ = ("l1", "l2", "call", "ev")

    def __init__(self, l1: str, l2: str, call: str, ev: StepEvent):
        self.l1, self.l2, self.call, self.ev = l1, l2, call, ev


_DROP_L1 = {"adapter", "terminal", "unknown", "other"}


def aligned_units(obs: Observation, include_lowlevel: bool = True) -> List[Unit]:
    units: List[Unit] = []
    for ev in ordered_steps(obs):
        for sg in ev.subgoals:
            if sg.l1 in _DROP_L1:
                continue
            if not include_lowlevel and sg.l1 in LOWLEVEL_L1:
                continue
            units.append(Unit(sg.l1, sg.l2, sg.call, ev))
    return units


def unit_key(u: Unit, match: str) -> str:
    if match == "l1":
        return u.l1
    if match == "call":
        return u.call
    return u.l2


def shared_prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ---------------------------------------------------------------------------
# Divergence attribution


def classify_tail(fail_obs: Observation, div_step_idx: int) -> Tuple[str, str]:
    """Return (bucket, sub) for the failing run from the divergence step onward."""
    tail = [ev for ev in ordered_steps(fail_obs) if ev.step_idx >= div_step_idx]
    err_classes = [step_error_class(ev) for ev in tail if ev.is_error]
    rep = "representation" in err_classes
    exec_err = "execution" in err_classes

    tail_units = [u for u in aligned_units(fail_obs) if u.ev.step_idx >= div_step_idx]
    error_run = max_run([ev.is_error for ev in tail]) >= 3
    lowlevel_run = max_run([u.l1 in LOWLEVEL_L1 for u in tail_units]) >= 3
    no_core = not any(u.l1 in CORE_L1 for u in tail_units)

    if rep:
        return "A_representation", ""
    if not tail or error_run or lowlevel_run or no_core:
        return "B_stall", ""
    return "C_wrong_choice", "feedback_resistant" if exec_err else "silent"


def prefix_contamination(fail_obs: Observation, div_step_idx: int) -> Tuple[bool, bool]:
    """Whether the SHARED prefix (fail steps before divergence) already contains a
    base-bug representation / no_action error -- a contaminated match to discount."""
    reps = noact = False
    for ev in ordered_steps(fail_obs):
        if ev.step_idx >= div_step_idx:
            break
        kls = step_error_class(ev)
        if kls == "representation":
            reps = True
        elif kls == "no_action":
            noact = True
    return reps, noact


def divergence_records(rows: Sequence[ObsRow], match: str) -> List[dict]:
    by_goal: Dict[str, List[ObsRow]] = defaultdict(list)
    for row in rows:
        by_goal[row.task.goal_key].append(row)

    records: List[dict] = []
    for goal_key, grp in by_goal.items():
        passes = [r for r in grp if r.success is True]
        fails = [r for r in grp if r.success is False]
        pass_units = [(r, aligned_units(r.obs)) for r in passes]
        for fr in fails:
            fu = aligned_units(fr.obs)
            fk = [unit_key(u, match) for u in fu]

            if not passes:
                records.append(_rec(goal_key, fr, match, matched=False,
                                    prefix_len=0, bucket="no_pass_in_goal", sub=""))
                continue

            best_len, best_pass, best_pu = -1, None, []
            for pr, pu in pass_units:
                pk = [unit_key(u, match) for u in pu]
                n = shared_prefix_len(fk, pk)
                if n > best_len:
                    best_len, best_pass, best_pu = n, pr, pu
            best_len = max(best_len, 0)

            # divergence step index in the failing run
            if best_len < len(fu):
                div_step_idx = fu[best_len].ev.step_idx
            elif best_len > 0:
                div_step_idx = fu[best_len - 1].ev.step_idx + 1  # fail is a prefix -> truncated
            else:
                first = ordered_steps(fr.obs)
                div_step_idx = first[0].step_idx if first else 0

            bucket, sub = classify_tail(fr.obs, div_step_idx)
            reps, noact = prefix_contamination(fr.obs, div_step_idx)
            fail_next = fu[best_len] if best_len < len(fu) else None
            pass_next = best_pu[best_len] if best_len < len(best_pu) else None
            records.append(_rec(
                goal_key, fr, match, matched=True, prefix_len=best_len,
                bucket=bucket, sub=sub,
                matched_pass=best_pass.obs.task_id if best_pass else "",
                fail_len=len(fu),
                prefix_key=" > ".join(fk[:best_len]),
                div_step_idx=div_step_idx,
                fail_next_l2=fail_next.l2 if fail_next else "",
                fail_next_call=fail_next.call if fail_next else "",
                pass_next_l2=pass_next.l2 if pass_next else "",
                pass_next_call=pass_next.call if pass_next else "",
                pass_continuation_l2=[u.l2 for u in best_pu[best_len:]],
                pass_continuation_call=[u.call for u in best_pu[best_len:]],
                prefix_has_representation=reps,
                prefix_has_no_action=noact,
            ))
    return records


def _rec(goal_key, fr, match, matched, prefix_len, bucket, sub,
         matched_pass="", fail_len=0, **extra) -> dict:
    rec = {
        "goal_key": goal_key,
        "domain": fr.task.domain,
        "fail_task_id": fr.obs.task_id,
        "run_id": fr.obs.run_id,
        "match": match,
        "matched": matched,
        "prefix_len": prefix_len,
        "fail_units": fail_len,
        "matched_pass": matched_pass,
        "bucket": bucket,
        "sub": sub,
    }
    rec.update(extra)
    return rec


def aggregate_candidates(records: Sequence[dict], min_prefix: int) -> List[dict]:
    """Aggregate real-prefix fails by (domain, goal_key, prefix) so a prefix with a
    reusable pass continuation and C-bucket support can become a procedure-memory
    candidate. Sorted by C support then fail support."""
    groups: Dict[Tuple[str, str, str], dict] = {}
    for r in records:
        if not r.get("matched") or r["prefix_len"] < min_prefix:
            continue
        k = (r["domain"], r["goal_key"], r.get("prefix_key", ""))
        g = groups.setdefault(k, {
            "domain": r["domain"], "goal_key": r["goal_key"],
            "prefix_key": r.get("prefix_key", ""), "fails": 0,
            "buckets": Counter(), "pass_continuations": Counter(), "contaminated": 0,
        })
        g["fails"] += 1
        g["buckets"][r["bucket"] + (("." + r["sub"]) if r["sub"] else "")] += 1
        cont = " > ".join(r.get("pass_continuation_l2") or [])
        if cont:
            g["pass_continuations"][cont] += 1
        if r.get("prefix_has_representation") or r.get("prefix_has_no_action"):
            g["contaminated"] += 1

    def c_support(g: dict) -> int:
        return (g["buckets"].get("C_wrong_choice.silent", 0)
                + g["buckets"].get("C_wrong_choice.feedback_resistant", 0))

    out = list(groups.values())
    out.sort(key=lambda g: (-c_support(g), -g["fails"]))
    return out


# ---------------------------------------------------------------------------
# Reporting


def pct(n: int, d: int) -> str:
    return "{:.0f}%".format(100.0 * n / d) if d else "-"


def build_report(rows, dist, records, args) -> str:
    L: List[str] = []
    L.append("# Divergence-Point Probe")
    L.append("")
    n_obs = len(rows)
    doms = sorted({r.task.domain for r in rows})
    n_pass = sum(1 for r in rows if r.success is True)
    n_fail = sum(1 for r in rows if r.success is False)
    L.append("## Corpus")
    L.append("")
    L.append("- observations: {} (pass {}, fail {})".format(n_obs, n_pass, n_fail))
    L.append("- domains: {}".format(", ".join(doms)))
    L.append("- match granularity: `{}`  min-prefix (real-path): {}".format(args.match, args.min_prefix))
    L.append("")

    # ---- Part A ----
    L.append("## Part A -- base-bug (representation) vs execution error")
    L.append("")
    L.append("| domain | err_steps | representation | no_action | execution | fail_tasks w/ base-bug | fail_tasks exec-only |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    tot = Counter()
    for dom in sorted(dist):
        c = dist[dom]
        tot.update(c)
        es = c["err_steps"]
        L.append("| {} | {} | {} ({}) | {} ({}) | {} ({}) | {}/{} | {}/{} |".format(
            dom, es,
            c["representation"], pct(c["representation"], es),
            c["no_action"], pct(c["no_action"], es),
            c["execution"], pct(c["execution"], es),
            c["fail_tasks_with_representation"], c["fail_tasks"],
            c["fail_tasks_execution_only"], c["fail_tasks"],
        ))
    es = tot["err_steps"]
    L.append("| **ALL** | {} | {} ({}) | {} ({}) | {} ({}) | {}/{} | {}/{} |".format(
        es,
        tot["representation"], pct(tot["representation"], es),
        tot["no_action"], pct(tot["no_action"], es),
        tot["execution"], pct(tot["execution"], es),
        tot["fail_tasks_with_representation"], tot["fail_tasks"],
        tot["fail_tasks_execution_only"], tot["fail_tasks"],
    ))
    L.append("")
    L.append("> representation+no_action = pre-boundary base-agent noise (the C+B interface-repair's turf / not real errors). execution = real env feedback that already loops back to the model.")
    L.append("")

    # ---- Part B ----
    L.append("## Part B -- divergence attribution")
    L.append("")
    matched = [r for r in records if r["matched"]]
    real = [r for r in matched if r["prefix_len"] >= args.min_prefix]
    unmatched = [r for r in records if not r["matched"]]

    def bucket_counts(recs):
        c = Counter()
        for r in recs:
            key = r["bucket"] + (("." + r["sub"]) if r["sub"] else "")
            c[key] += 1
        return c

    L.append("Fails with a same-goal pass to match: {} (of which shared a REAL prefix >= {}: {}). "
             "Fails in a goal with no pass at all: {}.".format(
                 len(matched), args.min_prefix, len(real), len(unmatched)))
    L.append("")
    L.append("### Bucket mix (fails that shared a REAL prefix >= {})".format(args.min_prefix))
    L.append("")
    bc = bucket_counts(real)
    order = ["A_representation", "B_stall",
             "C_wrong_choice.silent", "C_wrong_choice.feedback_resistant"]
    keys = order + sorted(set(bc) - set(order))
    L.append("| bucket | n | share |")
    L.append("| --- | ---: | ---: |")
    for k in keys:
        if bc.get(k):
            L.append("| {} | {} | {} |".format(k, bc[k], pct(bc[k], len(real))))
    c_total = bc.get("C_wrong_choice.silent", 0) + bc.get("C_wrong_choice.feedback_resistant", 0)
    L.append("")
    L.append("**Addressable-by-fork-demo bucket (C_wrong_choice) = {} / {} real-prefix fails ({}).**".format(
        c_total, len(real), pct(c_total, len(real))))
    L.append("")
    verdict = ("EMPTY -> shared-prefix opportunity does not exist; failures are base-bug/stall, "
               "pass-demo is redundant. Close the line."
               if c_total == 0 else
               "NON-ZERO -> a real wrong-choice bucket exists; inspect these records before building "
               "an L2 fork-time demo (feedback_resistant ones are the hard, memory-unlikely-to-fix subset).")
    L.append("Verdict: " + verdict)
    L.append("")
    contaminated = sum(1 for r in real if r.get("prefix_has_representation") or r.get("prefix_has_no_action"))
    L.append("Real-prefix fails whose SHARED prefix already contains a base-bug (representation/no_action) = {} ({}) -- discount these matches.".format(
        contaminated, pct(contaminated, len(real))))
    L.append("")

    # ---- candidate prefixes ----
    L.append("### Procedure-memory candidate prefixes (by domain/goal/prefix; C-bucket only)")
    L.append("")
    cands = [g for g in aggregate_candidates(records, args.min_prefix)
             if (g["buckets"].get("C_wrong_choice.silent", 0)
                 + g["buckets"].get("C_wrong_choice.feedback_resistant", 0)) > 0]
    if not cands:
        L.append("None: no (domain, goal, prefix) has a C_wrong_choice fail -> nothing to turn into procedure memory.")
    else:
        L.append("| domain | goal | shared_prefix | fails | buckets | top pass continuation | contaminated |")
        L.append("| --- | --- | --- | ---: | --- | --- | ---: |")
        for g in cands[:args.top]:
            bmix = ", ".join("{}={}".format(k, v) for k, v in g["buckets"].most_common())
            top = g["pass_continuations"].most_common(1)
            cont = "{} (x{})".format(top[0][0], top[0][1]) if top else "-"
            L.append("| {} | {} | {} | {} | {} | {} | {}/{} |".format(
                g["domain"], g["goal_key"], g["prefix_key"] or "(empty)", g["fails"],
                bmix, cont, g["contaminated"], g["fails"]))
    L.append("")

    # ---- per-goal breakdown ----
    L.append("### Per-goal (real-prefix fails only)")
    L.append("")
    by_goal: Dict[str, List[dict]] = defaultdict(list)
    for r in real:
        by_goal[r["goal_key"]].append(r)
    for goal_key in sorted(by_goal, key=lambda g: -len(by_goal[g])):
        recs = by_goal[goal_key]
        bc = bucket_counts(recs)
        parts = ["{}={}".format(k, v) for k, v in bc.most_common()]
        L.append("- `{}` (n={}): {}".format(goal_key, len(recs), ", ".join(parts)))
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-root", action="append", default=[])
    ap.add_argument("--schema-root", action="append", default=[])
    ap.add_argument("--run-root", action="append", default=[])
    ap.add_argument("--baseline-root", action="append", default=[])
    ap.add_argument("--audit", action="append", default=[])
    ap.add_argument("--traj", action="append", default=[])
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--match", choices=["l1", "l2", "call"], default="l2",
                    help="prefix-match granularity; l2 (typed op) is the honest default.")
    ap.add_argument("--min-prefix", type=int, default=2,
                    help="min shared prefix length to count a fail as 'on the same path'.")
    ap.add_argument("--top", type=int, default=25, help="max candidate-prefix rows to print.")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    rows, _tasks, _manifest = load_rows(args)
    dist = error_class_distribution(rows)
    records = divergence_records(rows, args.match)
    report = build_report(rows, dist, records, args)
    print(report)

    if args.out_dir:
        ensure_dir(args.out_dir)
        write_jsonl(os.path.join(args.out_dir, "divergence_records.jsonl"), records)
        write_jsonl(os.path.join(args.out_dir, "candidate_prefixes.jsonl"), [
            {"domain": g["domain"], "goal_key": g["goal_key"], "prefix_key": g["prefix_key"],
             "fails": g["fails"], "buckets": dict(g["buckets"]),
             "pass_continuations": dict(g["pass_continuations"]), "contaminated": g["contaminated"]}
            for g in aggregate_candidates(records, args.min_prefix)
        ])
        with open(os.path.join(args.out_dir, "divergence_report.md"), "w", encoding="utf-8") as f:
            f.write(report)
        print("\nwrote", args.out_dir)


if __name__ == "__main__":
    main()
