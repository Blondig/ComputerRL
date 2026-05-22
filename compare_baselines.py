"""
Compare two baselines by task result status.

Usage:
    python compare_baselines.py --b1 results/baseline1 --b2 results/baseline2
    python compare_baselines.py --b1 results/b1 --b2 results/b2 --domain chrome
"""

import argparse
import sys
from pathlib import Path


def scan_baseline(results_dir: Path, domain_filter: str | None) -> dict[str, str]:
    """Returns {domain/example_id: status} where status is 'success'|'failed'|'no_result'."""
    tasks = {}
    for domain_dir in sorted(results_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        if domain_filter and domain != domain_filter:
            continue
        for example_dir in sorted(domain_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            key = f"{domain}/{example_dir.name}"
            result_file = example_dir / "result.txt"
            if not result_file.exists():
                tasks[key] = "no_result"
            else:
                try:
                    score = float(result_file.read_text().strip())
                    tasks[key] = "success" if score > 0 else "failed"
                except (ValueError, IOError):
                    tasks[key] = "no_result"
    return tasks


def print_list(title: str, items: list[str]) -> None:
    print(f"\n--- {title} ({len(items)}) ---")
    for item in sorted(items):
        print(f"  {item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b1", required=True, help="Path to baseline 1 results dir")
    parser.add_argument("--b2", required=True, help="Path to baseline 2 results dir")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--verbose", action="store_true", help="Print per-task lists")
    args = parser.parse_args()

    b1_dir = Path(args.b1)
    b2_dir = Path(args.b2)
    for d in (b1_dir, b2_dir):
        if not d.exists():
            print(f"Directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    b1 = scan_baseline(b1_dir, args.domain)
    b2 = scan_baseline(b2_dir, args.domain)

    all_tasks = sorted(set(b1) | set(b2))

    # per-baseline counts
    def counts(data):
        from collections import Counter
        return Counter(data.values())

    c1, c2 = counts(b1), counts(b2)
    print(f"\n{'':30s}  {'B1':>10}  {'B2':>10}")
    print(f"{'Baseline':30s}  {args.b1:>10}  {args.b2:>10}")
    print("-" * 56)
    for status in ("success", "failed", "no_result"):
        print(f"{status:30s}  {c1[status]:>10}  {c2[status]:>10}")
    print(f"{'total':30s}  {len(b1):>10}  {len(b2):>10}")

    # transition analysis
    transitions: dict[tuple, list] = {}
    for task in all_tasks:
        s1 = b1.get(task, "missing")
        s2 = b2.get(task, "missing")
        transitions.setdefault((s1, s2), []).append(task)

    interesting = [
        (("failed", "success"), "B1 failed -> B2 success (B2 gains)"),
        (("success", "failed"), "B1 success -> B2 failed (B2 regressions)"),
        (("no_result", "success"), "B1 no_result -> B2 success"),
        (("success", "no_result"), "B1 success -> B2 no_result"),
        (("no_result", "failed"), "B1 no_result -> B2 failed"),
        (("failed", "no_result"), "B1 failed -> B2 no_result"),
        (("missing", "success"), "Only in B2 (success)"),
        (("missing", "failed"), "Only in B2 (failed)"),
        (("success", "missing"), "Only in B1 (success)"),
        (("failed", "missing"), "Only in B1 (failed)"),
    ]

    print("\n=== Transitions ===")
    for key, label in interesting:
        items = transitions.get(key, [])
        print(f"{label}: {len(items)}")
        if args.verbose and items:
            for t in sorted(items):
                print(f"    {t}")

    # same status
    same_success = transitions.get(("success", "success"), [])
    same_failed = transitions.get(("failed", "failed"), [])
    same_no_result = transitions.get(("no_result", "no_result"), [])
    print(f"\nBoth success:    {len(same_success)}")
    print(f"Both failed:     {len(same_failed)}")
    print(f"Both no_result:  {len(same_no_result)}")


if __name__ == "__main__":
    main()
