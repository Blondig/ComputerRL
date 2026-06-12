"""
Validate the cross-task ledger admission rule on EXISTING v3 audits -- no re-run,
no agent/env change. It re-classifies every errored step already recorded in the
audit (which stores `exe_result` and `action_sig`) with the same structural
classifier the gate would use (`error_ledger.classify_error_step`), and reports:

  * class distribution of error STEPS  (representation / no_action / execution)
  * class distribution of distinct error_NOTES keys that would be admitted vs
    dropped  (key = ERR|app|state_sig|action_sig, the thing v3 actually stores)
  * example action_sigs + exe_result heads per class, so you can eyeball whether
    the DROP bucket is exactly the SyntaxError / nXxxTools mangling junk and the
    ADMIT bucket is real tool/state feedback.
  * a generic "suspicious sig" flag inside the ADMITTED bucket: action_sig whose
    first identifier is `[a-z][A-Z]...` (one stray lowercase char glued before a
    CamelCase name, e.g. `nCalcTools`). This is NOT a drop rule -- it is a probe
    to SEE whether a second, non-SyntaxError mangling path slips past the rule
    before we decide to add anything app-specific.

The point: decide from data whether the rule drops junk only (not real errors)
BEFORE wiring it into finalize_task. Overfitting check = read the two buckets.

Usage:
  python analyze_error_admission.py /path/to/results_root [more_roots ...]
  python analyze_error_admission.py --audit /path/to/ledger.v3.audit.jsonl
  # add --domain libreoffice_calc to restrict by the step's app field
"""
import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

from mm_agents.error_ledger import classify_error_step

# one stray lowercase char prepended to a CamelCase identifier -> a generic
# serialization-leak fingerprint (e.g. `\n`+`CalcTools` -> `nCalcTools`), with
# NO assumption about the word "Tools" or any app name.
_SUSPICIOUS_SIG = re.compile(r"^[a-z][A-Z]")


def _find_audits(roots):
    files = []
    for r in roots:
        if os.path.isfile(r):
            files.append(r)
        else:
            files.extend(glob.glob(os.path.join(r, "**", "*.v3.audit.jsonl"), recursive=True))
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", help="result dirs to scan for *.v3.audit.jsonl")
    ap.add_argument("--audit", action="append", default=[], help="explicit audit file(s)")
    ap.add_argument("--domain", default=None, help="restrict to steps whose app == this")
    ap.add_argument("--examples", type=int, default=6, help="example sigs to print per class")
    args = ap.parse_args()

    audits = _find_audits(list(args.roots) + list(args.audit))
    if not audits:
        print("No *.v3.audit.jsonl found. Pass a results root or --audit path "
              "(these live on the run host, e.g. xpu80).")
        return
    print(f"scanning {len(audits)} audit file(s)\n")

    step_classes = Counter()
    note_admit, note_drop = set(), set()           # distinct ERR| keys
    examples = defaultdict(list)                    # class -> [(sig, exe_head)]
    suspicious_admitted = Counter()                 # sig -> count, inside ADMIT
    tasks = errored_tasks = 0

    for path in audits:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            tasks += 1
            saw_error = False
            for s in d.get("steps", []) or []:
                if not s.get("is_error"):
                    continue
                app = s.get("app", "unknown")
                if args.domain and app != args.domain:
                    continue
                saw_error = True
                exe = s.get("exe_result", "") or ""
                sig = s.get("action_sig") or s.get("api_call") or ""
                cls = classify_error_step(exe, sig)
                step_classes[cls] += 1
                key = "ERR|{}|{}|{}".format(app, (s.get("pre", {}) or {}).get("state_sig", "?"), sig)
                if cls == "execution":
                    note_admit.add(key)
                    if _SUSPICIOUS_SIG.match(sig or ""):
                        suspicious_admitted[sig] += 1
                else:
                    note_drop.add(key)
                if len(examples[cls]) < args.examples:
                    examples[cls].append((sig or "<none>", exe.replace("\n", " ")[:80]))
            if saw_error:
                errored_tasks += 1

    total_steps = sum(step_classes.values())
    if not total_steps:
        print("No errored steps matched (check --domain).")
        return

    print(f"tasks audited: {tasks}   tasks with >=1 error: {errored_tasks}")
    print(f"errored steps: {total_steps}\n")

    print("ERROR STEP classes (admit only 'execution'):")
    for cls in ("execution", "representation", "no_action"):
        n = step_classes.get(cls, 0)
        tag = "ADMIT" if cls == "execution" else "DROP "
        print(f"  [{tag}] {cls:14s} {n:5d}  ({n/total_steps*100:4.1f}%)")

    drop_only = note_drop - note_admit   # keys that exist ONLY as dropped
    print(f"\nDistinct error_notes keys: admit={len(note_admit)}  "
          f"drop-only={len(drop_only)}  "
          f"(a key in both = same action errored pre- and post-boundary across tasks)")

    for cls in ("representation", "no_action", "execution"):
        if examples[cls]:
            print(f"\n  e.g. {cls}:")
            for sig, head in examples[cls]:
                print(f"    {sig:30s} | {head}")

    if suspicious_admitted:
        print(f"\n!! ADMITTED but structurally suspicious sig ([a-z][A-Z] prefix) -- "
              f"a possible non-SyntaxError mangling path leaking past the rule:")
        for sig, n in suspicious_admitted.most_common():
            print(f"    {sig:30s} x{n}")
    else:
        print("\nNo suspicious ([a-z][A-Z]) sigs in the ADMIT bucket "
              "-> the compile-time rule already catches the mangling.")


if __name__ == "__main__":
    main()
