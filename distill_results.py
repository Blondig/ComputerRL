"""
Distill / fuse cross-task experiment outputs into a PAIRED, card-attributed view.
Read-only over result.txt (scores) + *.audit.jsonl (trajectories + injected cards).

v0.2 -- adds what v0.1 was missing (see the review):
  * Joins result.txt SCORES across two runs -> paired label per task
    (target_fixed / target_broke / both_pass / both_fail), so we can ask which
    card / which trajectory explains the VERSION DIFFERENCE, not just per-run rates.
  * Real INJECTED-CARD attribution by memory_id: injected_count, fixed/broke
    counts, success rate, top next-action, and whether it fired AFTER a
    representation / execution / success / start step (read-side boundary).
  * Trajectory n-grams split into success-only (clean executed flow) vs error
    steps, so procedural "scaffolding" patterns aren't mixed with failed actions.
  * [A] correctly named ERROR-STEP taxonomy (errored steps in trajectories), kept
    distinct from [D] the actually-injected memory cards.

Usage (analyze why v31 beats v35 -> baseline=v35, target=v31):
  python distill_results.py --baseline ~/...v35_impress --target ~/...v31_impress \
      --domain libreoffice_impress --csv fusion_v31_vs_v35_impress.csv
  # single run (no paired label): just --target
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict, Counter

try:
    from mm_agents.error_ledger import classify_error_step, _is_adapter_action
except Exception:
    def classify_error_step(exe, sig=""):
        return "execution"
    def _is_adapter_action(sig):
        return (sig or "").split(".")[-1] == "print_result"

_NON_PROC = {"WAIT", "Agent.wait", "DONE", "FAIL", "Agent.exit", "unknown", ""}
_TERMINAL_VERBS = ("save", "exit", "wait", "done", "fail", "finish", "close", "print")


# ----------------------------------------------------------------------
def _sig(step):
    return step.get("action_sig") or step.get("api_call") or ""


def collect_scores(root, thr=0.0):
    """{task_id: score} from every result.txt under root."""
    out = {}
    if not root:
        return out
    for p in glob.glob(os.path.join(root, "**", "result.txt"), recursive=True):
        try:
            raw = open(p).read().strip()
            score = float(raw)
        except Exception:
            try:
                score = float(eval(raw))
            except Exception:
                continue
        out[os.path.basename(os.path.dirname(p))] = score
    return out


def find_audit(root):
    for pat in ("*.v3.audit.jsonl", "*.audit.jsonl"):
        hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def load_audit(path):
    """{task_id: record} where record has steps, injected, success, app, step_by_idx."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        steps = d.get("steps", []) or []
        apps = [s.get("app") for s in steps if s.get("app")]
        d["app"] = max(set(apps), key=apps.count) if apps else "unknown"
        d["step_by_idx"] = {s.get("step_idx"): s for s in steps}
        out[d.get("task_id", "")] = d
    return out


def error_template(exe_result: str) -> str:
    text = (exe_result or "").strip()
    if not text:
        return "(empty)"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pick = next((ln for ln in reversed(lines) if re.search(r"(Error|Exception)\b", ln)), lines[-1])
    pick = re.sub(r"'[^']*'", "'X'", pick)
    pick = re.sub(r'"[^"]*"', '"X"', pick)
    pick = re.sub(r"\b\d+\b", "N", pick)
    return re.sub(r"\s+", " ", pick)[:120]


def canonical_actions(steps, clean_only=False):
    """High-level procedural action sequence (adapter/wait/terminal/unknown dropped).
    clean_only=True also drops errored steps -> the successfully executed flow."""
    seq = []
    for s in steps:
        if clean_only and s.get("is_error"):
            continue
        sig = _sig(s)
        if not sig or sig in _NON_PROC or _is_adapter_action(sig):
            continue
        method = sig.split(".")[-1].lower()
        if any(method.startswith(v) for v in _TERMINAL_VERBS):
            continue
        seq.append(sig)
    return seq


def ngrams(seq, n):
    return [" > ".join(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def prev_step_class(rec, idx):
    s = rec["step_by_idx"].get(idx - 1)
    if s is None:
        return "start"
    if not s.get("is_error"):
        return "success"
    return classify_error_step(s.get("exe_result", ""), _sig(s))


def paired_label(base, tgt, thr=0.0):
    tp = tgt is not None and tgt > thr
    if base is None:
        return "tgt_pass" if tp else "tgt_fail"
    bp = base > thr
    if bp and tp:
        return "both_pass"
    if not bp and not tp:
        return "both_fail"
    return "target_fixed" if tp else "target_broke"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="run dir to analyze (e.g. v31)")
    ap.add_argument("--baseline", default=None, help="run dir for paired labels (e.g. v35)")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    base_scores = collect_scores(args.baseline)
    tgt_scores = collect_scores(args.target)
    tgt_audit = load_audit(find_audit(args.target))
    if not tgt_scores and not tgt_audit:
        print("Nothing found under --target (no result.txt / audit).")
        return

    rows = []
    err_steps = defaultdict(lambda: {"tasks": set(), "succ": set(), "cls": Counter(),
                                     "adapter": False})
    # success-only n-grams: pattern -> {tasks, by_label}
    pat = defaultdict(lambda: {"tasks": set(), "succ": set(), "fixed": 0, "broke": 0})
    # injected card attribution: memory_id -> stats
    cards = defaultdict(lambda: {"n": 0, "kind": "", "fixed": 0, "broke": 0, "succ": 0,
                                 "next": Counter(), "prev": Counter(), "tasks": set()})

    all_tids = set(tgt_scores) | set(tgt_audit)
    for tid in all_tids:
        rec = tgt_audit.get(tid)
        app = rec["app"] if rec else "unknown"
        if args.domain and app != args.domain:
            continue
        tscore = tgt_scores.get(tid)
        bscore = base_scores.get(tid)
        label = paired_label(bscore, tscore)
        succ = tscore is not None and tscore > 0.0
        steps = rec["steps"] if rec else []

        cls_counter = Counter()
        for s in steps:
            if not s.get("is_error"):
                continue
            tmpl = error_template(s.get("exe_result", ""))
            cls = classify_error_step(s.get("exe_result", ""), _sig(s))
            cls_counter[cls] += 1
            e = err_steps[tmpl]
            e["tasks"].add(tid)
            if succ:
                e["succ"].add(tid)
            e["cls"][cls] += 1
            e["adapter"] = e["adapter"] or _is_adapter_action(_sig(s))

        seq_clean = canonical_actions(steps, clean_only=True)
        seq_err = [_sig(s) for s in steps if s.get("is_error") and _sig(s)]
        for n in (2, 3):
            for g in set(ngrams(seq_clean, n)):
                p = pat[g]
                p["tasks"].add(tid)
                if succ:
                    p["succ"].add(tid)
                if label == "target_fixed":
                    p["fixed"] += 1
                elif label == "target_broke":
                    p["broke"] += 1

        inj = (rec.get("injected") if rec else None) or []
        for it in inj:
            mid = it.get("memory_id", "")
            if not mid:
                continue
            c = cards[mid]
            c["n"] += 1
            c["kind"] = it.get("kind", c["kind"])
            c["tasks"].add(tid)
            if succ:
                c["succ"] += 1
            if label == "target_fixed":
                c["fixed"] += 1
            elif label == "target_broke":
                c["broke"] += 1
            idx = it.get("step_idx")
            c["prev"][prev_step_class(rec, idx)] += 1
            nxt = rec["step_by_idx"].get(idx)
            c["next"][_sig(nxt) if nxt else "(none)"] += 1

        rows.append({
            "task_id": tid, "app": app,
            "baseline_score": "" if bscore is None else bscore,
            "target_score": "" if tscore is None else tscore,
            "label": label,
            "n_steps": len(steps),
            "n_err": sum(1 for s in steps if s.get("is_error")),
            "err_representation": cls_counter.get("representation", 0),
            "err_execution": cls_counter.get("execution", 0),
            "err_no_action": cls_counter.get("no_action", 0),
            "injected_ids": ";".join(sorted({i.get("memory_id", "") for i in inj if i.get("memory_id")})),
            "injected_kinds": ";".join(sorted({i.get("kind", "") for i in inj if i.get("kind")})),
            "injected_steps": ";".join(str(i.get("step_idx")) for i in inj),
            "prev_classes": ";".join(prev_step_class(rec, i.get("step_idx")) for i in inj) if rec else "",
            "next_actions": ";".join((_sig(rec["step_by_idx"].get(i.get("step_idx"))) if rec else "") for i in inj),
            "seq_success_only": " > ".join(seq_clean),
            "seq_error_only": " > ".join(seq_err),
        })

    if not rows:
        print("No tasks matched (check --domain / paths).")
        return
    lab = Counter(r["label"] for r in rows)
    base_sr = sum(1 for r in rows if r["target_score"] != "" and r["target_score"] > 0) / len(rows)
    print(f"target={os.path.basename(args.target)}  baseline={os.path.basename(args.baseline) if args.baseline else '(none)'}"
          f"  tasks={len(rows)}  base_sr(target)={base_sr*100:.1f}%")
    print("paired labels:", dict(lab), "\n")

    # [A] error-STEP taxonomy
    print("=" * 92)
    print("[A] ERROR-STEP TAXONOMY  (errored steps in trajectories -- NOT injected cards)")
    print("=" * 92)
    for tmpl, e in sorted(err_steps.items(), key=lambda kv: len(kv[1]["tasks"]), reverse=True)[:args.top]:
        nt = len(e["tasks"])
        sr = len(e["succ"]) / nt * 100 if nt else 0
        cls = e["cls"].most_common(1)[0][0]
        tag = "ADMIT" if (cls == "execution" and not e["adapter"]) else "drop "
        print(f"  [{tag}] {cls:14s} x{nt:<3d} succ={sr:4.0f}%  {tmpl}")

    # [B] success-only trajectory patterns
    print("\n" + "=" * 92)
    print(f"[B] TRAJECTORY PATTERNS (success-only flow | #tasks | succ% | lift vs {base_sr*100:.0f}% | fixed/broke)")
    print("=" * 92)
    plist = sorted(((g, p) for g, p in pat.items() if len(p["tasks"]) >= 3),
                   key=lambda x: len(x[1]["tasks"]), reverse=True)
    for g, p in plist[:args.top]:
        nt = len(p["tasks"])
        sr = len(p["succ"]) / nt * 100
        lift = sr - base_sr * 100
        flag = "++" if lift >= 15 else ("--" if lift <= -15 else "  ")
        print(f"  {flag} x{nt:<3d} succ={sr:4.0f}% lift={lift:+5.0f}pp  fx/bk={p['fixed']}/{p['broke']}  {g}")

    # [D] injected-card attribution
    print("\n" + "=" * 92)
    print("[D] INJECTED-CARD ATTRIBUTION (by memory_id | inj | fixed/broke | succ% | prev-step | top next)")
    print("=" * 92)
    if not cards:
        print("  (no injected cards in this run's audit)")
    for mid, c in sorted(cards.items(), key=lambda kv: kv[1]["n"], reverse=True)[:args.top]:
        sr = c["succ"] / c["n"] * 100 if c["n"] else 0
        prev = ",".join(f"{k}:{v}" for k, v in c["prev"].most_common())
        nxt = c["next"].most_common(1)[0][0] if c["next"] else "-"
        print(f"  inj={c['n']:<3d} fx/bk={c['fixed']}/{c['broke']} succ={sr:4.0f}%  prev[{prev}]  next={nxt}")
        print(f"        {mid}")

    if args.csv:
        cols = ["task_id", "app", "baseline_score", "target_score", "label", "n_steps", "n_err",
                "err_representation", "err_execution", "err_no_action",
                "injected_ids", "injected_kinds", "injected_steps", "prev_classes", "next_actions",
                "seq_success_only", "seq_error_only"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[C] per-task fusion table -> {args.csv}  ({len(rows)} rows)")
    else:
        print("\n[C] pass --csv to dump the per-task fusion table (paired label + injection attribution + seqs).")


if __name__ == "__main__":
    main()
