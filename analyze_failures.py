"""
Analyze failed OSWorld tasks using a local vllm model.

Usage:
    python analyze_failures.py \
        --results_dir results/b1_chrome_qwen/pyautogui/screenshot/qwen/qwen3.5-122b-a10b \
        --domain chrome \
        --model Qwen/Qwen3.5-72B-Instruct \
        --base_url http://localhost:8000/v1 \
        --output failure_analysis.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from openai import OpenAI

ANALYSIS_PROMPT = """\
You are analyzing a failed desktop automation task.

## Task Instruction
{instruction}

## Trajectory (all steps)
{trajectory}

## Your Job
Identify WHY the task failed. Be concise and specific. Output JSON with these fields:
- "failure_type": one of ["stuck_in_loop", "wrong_target", "missing_step", "api_misuse", "task_impossible", "timeout", "other"]
- "root_cause": 1-2 sentences describing what went wrong
- "key_mistake": the specific step or pattern that caused failure (e.g. "repeatedly clicked same coordinates without checking result")
- "fix_hint": 1 sentence on what should have been done differently

Return only valid JSON, no markdown wrapper."""


def load_instruction(eval_dir: Path, domain: str, example_id: str) -> str:
    config_path = eval_dir / "examples" / domain / f"{example_id}.json"
    if not config_path.exists():
        return "(instruction not found)"
    with open(config_path) as f:
        return json.load(f).get("instruction", "(no instruction field)")


def format_trajectory(traj_path: Path, max_steps: int = 30) -> str:
    steps = []
    with open(traj_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    lines = []
    for step in steps[:max_steps]:
        n = step.get("step_num", "?")
        raw_action = step.get("action", "")
        if isinstance(raw_action, dict):
            action = json.dumps(raw_action, ensure_ascii=False)
        else:
            action = str(raw_action).replace("\n", " ").strip()
        if len(action) > 300:
            action = action[:300] + "..."
        done = step.get("done", False)
        info = step.get("info", {})
        lines.append(f"Step {n}: {action}" + (" [DONE]" if done else "") + (f" info={info}" if info else ""))

    if len(steps) > max_steps:
        lines.append(f"... ({len(steps) - max_steps} more steps truncated)")

    return "\n".join(lines) if lines else "(empty trajectory)"


def analyze_one(client: OpenAI, model: str, instruction: str, trajectory: str) -> dict:
    prompt = ANALYSIS_PROMPT.format(instruction=instruction, trajectory=trajectory)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}


def collect_failed_tasks(results_dir: Path, domain_filter: str | None) -> list[dict]:
    failed = []
    # structure: results_dir / domain / example_id / result.txt
    for result_file in sorted(results_dir.rglob("result.txt")):
        try:
            score = float(result_file.read_text().strip())
        except (ValueError, IOError):
            continue
        domain = domain_dir.name
        if domain_filter and domain != domain_filter:
            continue

        for example_dir in sorted(domain_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            traj_path = example_dir / "traj.jsonl"
            if not traj_path.exists():
                continue

            status, score = get_task_status(example_dir)
            buckets[status].append({
                "domain": domain,
                "example_id": example_dir.name,
                "example_dir": example_dir,
                "traj_path": traj_path,
                "score": score,
                "status": status,
            })

    return buckets


def aggregate_summary(results: list[dict], skipped_success: int, skipped_no_traj: int) -> dict:
    from collections import Counter
    failure_types = Counter(
        r["analysis"].get("failure_type", "unknown")
        for r in results
        if "analysis" in r and not r["analysis"].get("parse_error")
    )
    status_counts = Counter(r["status"] for r in results)
    return {
        "analyzed": len(results),
        "skipped_success": skipped_success,
        "skipped_no_traj": skipped_no_traj,
        "status_counts": dict(status_counts),
        "failure_type_counts": dict(failure_types.most_common()),
        "most_common_failure": failure_types.most_common(1)[0][0] if failure_types else "unknown",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Path to model results dir (contains domain subdirs)")
    parser.add_argument("--eval_dir", default="evaluation_examples", help="Path to evaluation_examples dir")
    parser.add_argument("--domain", default=None, help="Filter to specific domain (e.g. chrome)")
    parser.add_argument("--model", default="qwen3.5-122b-a10b", help="Model name served by vllm")
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="vllm OpenAI-compatible endpoint")
    parser.add_argument("--output", default="failure_analysis.json", help="Output JSON file")
    parser.add_argument("--max_steps", type=int, default=30, help="Max trajectory steps to send to LLM")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N failed tasks (for testing)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    eval_dir = Path(args.eval_dir)

    if not results_dir.exists():
        print(f"results_dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    buckets = collect_tasks(results_dir, args.domain)
    print(f"success={len(buckets['success'])}  failed={len(buckets['failed'])}  no_result={len(buckets['no_result'])}")

    to_analyze = buckets["failed"] + buckets["no_result"]
    if args.limit:
        to_analyze = to_analyze[: args.limit]

    print(f"Analyzing {len(to_analyze)} tasks (failed + no_result)")

    all_results = []
    for i, task in enumerate(to_analyze):
        domain = task["domain"]
        example_id = task["example_id"]
        status = task["status"]
        print(f"[{i+1}/{len(to_analyze)}] [{status}] {domain}/{example_id} ... ", end="", flush=True)

        instruction = load_instruction(eval_dir, domain, example_id)
        trajectory = format_trajectory(task["traj_path"], max_steps=args.max_steps)

        try:
            analysis = analyze_one(client, args.model, instruction, trajectory)
            print(analysis.get("failure_type", "?"))
        except Exception as e:
            analysis = {"error": str(e)}
            print(f"ERROR: {e}")

        all_results.append({
            "domain": domain,
            "example_id": example_id,
            "status": status,
            "instruction": instruction,
            "analysis": analysis,
        })

    summary = aggregate_summary(
        all_results,
        skipped_success=len(buckets["success"]),
        skipped_no_traj=0,
    )
    output = {
        "summary": summary,
        "tasks": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"Analyzed: {summary['analyzed']}  (success skipped: {summary['skipped_success']})")
    print(f"By status: {summary['status_counts']}")
    print(f"Failure types: {summary['failure_type_counts']}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
