"""
Role-motif analyzer -- the GUI-agent-GENERIC layer on top of distill_results.

Anti-overfit discipline (enforced in code):
  * The keyword map below is ONLY an adapter that translates THIS log's concrete
    actions (CalcTools.get_column_data, ImpressTools.go_to_slide, ...) into
    intent ROLES. The analysis downstream consumes ROLE SEQUENCES ONLY -- never a
    tool/API name. A different agent can swap this mapper for a tool-schema /
    LLM classifier without touching the method.
  * Implementation fingerprints are NOT positive knowledge: a mangled tool name
    like `nCalcTools.*` (leaked-newline artifact) maps to role `garbled` and is
    dropped from every motif. SyntaxError/content-block live in error steps and
    are excluded by clean_only.
  * `print_result` -> adapter (dropped). `save/export` -> commit (terminal-ish,
    not real verification). Bare `Agent.click/type` -> lowlevel (kept as real
    actions but never counted as positive procedural knowledge; only REPEATED
    bare click/type is read as a low-semantic / stuck signal).
  * A motif is only "transferable" if it is positive in >=2 domains AND the 3rd
    is not clearly negative -- single-domain lift is ignored (writer n is small).

Roles: inspect / locate / input / mutate_data / format / structural_edit /
       commit / terminal / adapter / lowlevel_select / lowlevel_input / garbled / other

Usage (domain=target_run[:baseline_run], ~ ok):
  python analyze_role_motifs.py \
    libreoffice_calc=~/...v31_calc:~/...v34_calc \
    libreoffice_impress=~/...v31_impress:~/...v34_impress \
    libreoffice_writer=~/...v31_writer:~/...v34_writer
"""
import argparse
import os
import re
from collections import defaultdict, Counter

from distill_results import (collect_scores, find_audit, load_audit,
                             paired_label, _sig, ngrams)

# ---- the (replaceable) action -> intent-role adapter --------------------------
_FORMAT_KW = ("color", "font", "align", "style", "spacing", "strike", "bold",
              "italic", "underline", "number_format", "background", "highlight",
              "border", "orientation", "indent")


def action_role(sig: str) -> str:
    s = sig or ""
    if not s or s in ("unknown", "WAIT", "DONE", "FAIL"):
        return "other" if s not in ("WAIT", "DONE", "FAIL") else "terminal"
    tool, _, method = s.partition(".")
    method = (method or tool).lower()
    # mangling fingerprint (nCalcTools / nImpressTools ...): never knowledge
    if re.match(r"^[a-z][A-Z]", tool):
        return "garbled"
    if tool in ("Agent", "pyautogui"):
        if method in ("click", "double_click", "right_click", "moveto", "drag"):
            return "lowlevel_select"
        if method in ("type", "write", "press", "hotkey", "key"):
            return "lowlevel_input"
        if method in ("exit", "wait", "stop", "done", "fail"):
            return "terminal"
    if method == "print_result":
        return "adapter"
    if any(method.startswith(v) or method == v for v in ("exit", "done", "fail", "wait", "finish", "close")):
        return "terminal"
    if method.startswith("save") or method.startswith("export"):
        return "commit"
    if "find_and_replace" in method or method.startswith("replace"):
        return "structural_edit"
    if any(method.startswith(v) for v in ("insert", "delete", "remove", "duplicate", "create", "clear", "add_slide", "new_")):
        return "structural_edit"
    if any(method.startswith(v) for v in ("get", "read", "check", "count", "list", "describe", "find", "observe", "is_")):
        return "inspect"
    if any(k in method for k in _FORMAT_KW):
        return "format"
    if any(method.startswith(v) for v in ("type", "fill", "write")):
        return "input"
    if any(method.startswith(v) for v in ("go_to", "goto", "switch", "open", "scroll", "focus", "activate", "select", "navigate")):
        return "locate"
    if method.startswith("set") or any(method.startswith(v) for v in ("merge", "sort", "rename", "update", "move", "add", "formula")):
        return "mutate_data"
    return "other"


_DROP_ROLES = {"adapter", "terminal", "garbled", "other"}
_LOWLEVEL = {"lowlevel_select", "lowlevel_input"}

MOTIFS_OF_INTEREST = [
    ("inspect", "mutate_data"), ("inspect", "structural_edit"),
    ("locate", "format"), ("locate", "input"),
    ("format", "commit"), ("mutate_data", "commit"), ("structural_edit", "commit"),
]


def role_seq(steps, clean_only=True):
    out = []
    for s in steps:
        if clean_only and s.get("is_error"):
            continue
        r = action_role(_sig(s))
        if r in _DROP_ROLES:
            continue
        out.append(r)
    return out


def _max_run(flags):
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def stuck_signals(steps):
    """Two clean stuck signals (the coarse 'same high-level role >=3' is dropped --
    it mis-flags legit batched ops like formatting several objects in a row):
      err_loop : >=3 consecutive errored steps (the repetition/death loop)
      ll_loop  : >=3 consecutive bare lowlevel click/type (low-semantic flailing)
    NOTE: collapse steps that produce no valid action often aren't recorded in the
    audit, so err_loop under-counts -- treat as a floor, not a full measure."""
    err_loop = _max_run([bool(s.get("is_error")) for s in steps]) >= 3
    ll_loop = _max_run([action_role(_sig(s)) in _LOWLEVEL for s in steps]) >= 3
    return err_loop, ll_loop


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="domain=target_dir[:baseline_dir]")
    ap.add_argument("--min", type=int, default=3, help="min tasks for a motif to be shown")
    args = ap.parse_args()

    # domain -> {base_sr, motif:{n,succ,fixed,broke}, stuck:(n,succ), roles:Counter}
    dom = {}
    for spec in args.runs:
        domain, _, rest = spec.partition("=")
        target, _, baseline = rest.partition(":")
        target = os.path.expanduser(target)
        baseline = os.path.expanduser(baseline) if baseline else None
        tgt_scores = collect_scores(target)
        base_scores = collect_scores(baseline)
        audit = load_audit(find_audit(target))

        motif = defaultdict(lambda: {"tasks": set(), "succ": set(), "fixed": 0, "broke": 0})
        roles_seen = Counter()
        stuck = {"err": 0, "err_succ": 0, "ll": 0, "ll_succ": 0, "tot": 0, "tot_succ": 0}
        n = succ_n = 0
        for tid in set(tgt_scores) | set(audit):
            rec = audit.get(tid)
            apps = [s.get("app") for s in (rec["steps"] if rec else []) if s.get("app")]
            app = max(set(apps), key=apps.count) if apps else "unknown"
            if app != domain:
                continue
            ts = tgt_scores.get(tid)
            succ = ts is not None and ts > 0
            label = paired_label(base_scores.get(tid), ts)
            steps = rec["steps"] if rec else []
            n += 1
            succ_n += int(succ)

            rs = role_seq(steps, clean_only=True)
            for r in rs:
                roles_seen[r] += 1
            for g in set(ngrams(rs, 2)):
                m = motif[g]
                m["tasks"].add(tid)
                if succ:
                    m["succ"].add(tid)
                if label == "target_fixed":
                    m["fixed"] += 1
                elif label == "target_broke":
                    m["broke"] += 1
            if steps:
                stuck["tot"] += 1
                stuck["tot_succ"] += int(succ)
                err_loop, ll_loop = stuck_signals(steps)
                if err_loop:
                    stuck["err"] += 1
                    stuck["err_succ"] += int(succ)
                if ll_loop:
                    stuck["ll"] += 1
                    stuck["ll_succ"] += int(succ)

        base_sr = succ_n / n if n else 0.0
        dom[domain] = {"base_sr": base_sr, "n": n, "motif": motif,
                       "roles": roles_seen, "stuck": stuck}

    domains = list(dom)
    print("domains:", {d: f"{dom[d]['n']}tasks/{dom[d]['base_sr']*100:.0f}%" for d in domains})
    print("role mix:", {d: dict(dom[d]["roles"].most_common(6)) for d in domains})

    def lift(domain, g):
        m = dom[domain]["motif"].get(g)
        if not m or len(m["tasks"]) < args.min:
            return None
        sr = len(m["succ"]) / len(m["tasks"])
        return (sr - dom[domain]["base_sr"]) * 100, len(m["tasks"]), m["fixed"], m["broke"]

    # ---- cross-domain matrix for the motifs of interest + discovered ones ----
    discovered = set()
    for d in domains:
        for g, m in dom[d]["motif"].items():
            if len(m["tasks"]) >= args.min:
                discovered.add(g)
    motif_list = [tuple(x) for x in MOTIFS_OF_INTEREST] + sorted(discovered - set(map(tuple, MOTIFS_OF_INTEREST)))

    print("\n" + "=" * 96)
    print("ROLE-MOTIF CROSS-DOMAIN LIFT  (lift pp [n, fx/bk] per domain | verdict)")
    print("transferable = positive in >=2 domains AND 3rd not clearly negative (< -10pp)")
    print("=" * 96)
    hdr = "  {:28s}".format("role-motif") + "".join(f"{d.split('_')[-1][:9]:>16s}" for d in domains) + "   verdict"
    print(hdr)
    for g in motif_list:
        cells, lifts = [], []
        for d in domains:
            r = lift(d, g)
            if r is None:
                cells.append(f"{'-':>16s}")
                lifts.append(None)
            else:
                lf, nt, fx, bk = r
                cells.append(f"{lf:+5.0f}[{nt},{fx}/{bk}]"[:15].rjust(16))
                lifts.append(lf)
        present = [x for x in lifts if x is not None]
        pos = sum(1 for x in present if x >= 10)
        clearly_neg = any(x <= -10 for x in present)
        is_ll = any(role in _LOWLEVEL for role in g)
        if is_ll:
            verdict = "  (lowlevel -- diagnostic only, NOT procedural)"   # never positive knowledge
        elif pos >= 2 and not clearly_neg:
            verdict = "  <-- TRANSFERABLE"
        elif pos >= 1:
            verdict = "  (single-domain)"
        else:
            verdict = ""
        star = "*" if tuple(g) in set(map(tuple, MOTIFS_OF_INTEREST)) else " "
        print(f" {star}{(' > '.join(g)):28s}" + "".join(cells) + verdict)

    print("\nSTUCK signals (diagnostic; err-loop under-counts since collapse steps may be unrecorded):")
    for d in domains:
        s = dom[d]["stuck"]
        if not s["tot"]:
            continue
        e_sr = f"{s['err_succ']/s['err']*100:.0f}%" if s["err"] else "-"
        l_sr = f"{s['ll_succ']/s['ll']*100:.0f}%" if s["ll"] else "-"
        print(f"  {d:24s} err-loop {s['err']:>2d}/{s['tot']} (succ {e_sr}) | "
              f"lowlevel-loop {s['ll']:>2d}/{s['tot']} (succ {l_sr}) | base {dom[d]['base_sr']*100:.0f}%")
    print("\n* = a-priori motif of interest. Read TRANSFERABLE rows as candidate "
          "GUI-agent procedural knowledge (role-level, not API-level).")


if __name__ == "__main__":
    main()
