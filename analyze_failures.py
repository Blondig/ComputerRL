"""
Analyze failed OSWorld/ComputerRL tasks using a local vllm model.

Usage:
    python analyze_failures.py \
        --results_dir results/b1_chrome_qwen/pyautogui/screenshot/qwen/qwen3.5-122b-a10b \
        --eval_dir ../OSWorld/evaluation_examples \
        --domain chrome \
        --model Qwen/Qwen3-8B-Instruct \
        --base_url http://localhost:8000/v1 \
        --output failure_analysis.json

    # test with 5 tasks first
    python analyze_failures.py ... --limit 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from openai import OpenAI

ANALYSIS_PROMPT = """\
You are a post-mortem analyst. A desktop automation agent has already finished running and FAILED. \
Do NOT continue the task. Do NOT suggest next actions. Your ONLY job is to explain why it failed.

## Task Instruction (what the agent was supposed to do)
{instruction}

## What the agent actually did (completed trajectory, task is over)
{trajectory}

## Output format
Respond with a single JSON object and nothing else:
{{
  "failure_type": "<one of: stuck_in_loop | wrong_target | missing_step | api_misuse | task_impossible | other>",
  "root_cause": "<1-2 sentences: what went wrong>",
  "key_mistake": "<the specific step or repeated pattern that caused failure>",
  "fix_hint": "<1 sentence: what the agent should have done differently>"
}}"""


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
        lines.append(
            f"Step {n}: {action}"
            + (" [DONE]" if done else "")
            + (f" info={info}" if info else "")
        )

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
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}


def get_task_status(example_dir: Path) -> tuple[str, float | None]:
    """Returns (status, score). status: 'success' | 'failed' | 'no_result'."""
    result_file = example_dir / "result.txt"
    if not result_file.exists():
        return "no_result", None
    try:
        score = float(result_file.read_text().strip())
        return ("success" if score > 0 else "failed"), score
    except (ValueError, IOError):
        return "no_result", None


def collect_tasks(results_dir: Path, domain_filter: str | None) -> dict:
    """Scan all example dirs. Returns success/failed lists and no_result count."""
    buckets: dict = {"success": [], "failed": [], "no_result": 0}

    for domain_dir in sorted(results_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        if domain_filter and domain != domain_filter:
            continue

        for example_dir in sorted(domain_dir.iterdir()):
            if not example_dir.is_dir():
                continue

            status, score = get_task_status(example_dir)
            if status == "no_result":
                buckets["no_result"] += 1
                continue

            traj_path = example_dir / "traj.jsonl"
            if not traj_path.exists():
                continue

            buckets[status].append({
                "domain": domain,
                "example_id": example_dir.name,
                "example_dir": example_dir,
                "traj_path": traj_path,
                "score": score,
            })

    return buckets


def aggregate_summary(results: list[dict], skipped_success: int) -> dict:
    failure_types = Counter(
        r["analysis"].get("failure_type", "unknown")
        for r in results
        if "analysis" in r and not r["analysis"].get("parse_error")
    )
    return {
        "analyzed": len(results),
        "skipped_success": skipped_success,
        "failure_type_counts": dict(failure_types.most_common()),
        "most_common_failure": failure_types.most_common(1)[0][0] if failure_types else "unknown",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--eval_dir", default="evaluation_examples")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--model", default="qwen3.5-122b-a10b")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--output", default="failure_analysis.json")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N tasks (for testing)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    eval_dir = Path(args.eval_dir)

    if not results_dir.exists():
        print(f"results_dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    buckets = collect_tasks(results_dir, args.domain)
    print(f"success={len(buckets['success'])}  failed={len(buckets['failed'])}  no_result={buckets['no_result']}")

    to_analyze = buckets["failed"]
    if args.limit:
        to_analyze = to_analyze[: args.limit]
    print(f"Analyzing {len(to_analyze)} failed tasks")

    all_results = []
    for i, task in enumerate(to_analyze):
        domain = task["domain"]
        example_id = task["example_id"]
        print(f"[{i+1}/{len(to_analyze)}] {domain}/{example_id} ... ", end="", flush=True)

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
            "instruction": instruction,
            "analysis": analysis,
        })

    summary = aggregate_summary(all_results, skipped_success=len(buckets["success"]))
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "tasks": all_results}, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"success={len(buckets['success'])}  failed={len(all_results)}  no_result={buckets['no_result']}")
    print(f"Failure types: {summary['failure_type_counts']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
