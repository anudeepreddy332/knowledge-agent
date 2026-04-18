"""
Evaluate harness for knowledge agent

Runs every query in tests/eval_queries.json through the live agent,
scores each result against ground truth, and prints a report.

Metrics produced:
  - Tool routing accuracy  : did the agent call the expected tool first?
  - KB retrieval hit rate  : for KB queries, did a retrieved chunk contain the expected keyword?
  - Mean best relevance    : average reranked cosine distance across KB queries (higher = more confident)
  - Overall pass rate      : queries where both routing AND content checks passed

"""
import json
import os
from pathlib import Path
from src.knowledge_agent.config import EVAL_PATH, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL
from src.knowledge_agent.tools import TOOL_EXECUTORS, TOOL_SCHEMAS
from openai import OpenAI
from dotenv import load_dotenv
from main import SYSTEM_PROMPT
load_dotenv()


# Config

EVAL_PATH = Path(EVAL_PATH)
MAX_ROUNDS = 8 # guard rail
DISTANCE_WARN_THRESHOLD = 0.65 # flag kb results > 0.65 as low-confidence


# Helpers

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return f"[error] Unknown tool: '{tool_name}'"
    try:
        return executor(**tool_input)
    except TypeError as e:
        return f"[error] Bad arguments for '{tool_name}': {e}"


def run_eval_turn(client: OpenAI, question: str) -> dict:
    """
        Run one question through the agent and collect observations.

        Returns a dict with:
          - answer        : final text response
          - tools_called  : list of tool names called, in order
          - first_tool    : the first tool called (used for routing check)
          - kb_distances  : list of distance scores from any KB calls (parsed from output)
          - raw_kb_output : raw string returned by KB tool (for keyword search)

        Same ReAct loop as main.py, but instead of printing we collect metadata.
        We parse the KB tool output for distance scores so we can measure retrieval confidence.

    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tools_called = []
    kb_distances = []
    raw_kb_output = ""
    kb_called = False


    for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=1024,
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return {
                "answer":       msg.content or "",
                "tools_called": tools_called,
                "first_tool":   tools_called[0] if tools_called else None,
                "kb_distances": kb_distances,
                "raw_kb_output": raw_kb_output,
            }

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tools_called.append(tool_name)


            if tool_name == "query_knowledge_base" and kb_called:
                result = (
                    "[SYSTEM GATE] query_knowledge_base already called this turn. "
                    "Proceed to verify_claim on the chunks above, or answer directly."
                )
            else:
                if tool_name == "query_knowledge_base":
                    kb_called = True
                result = dispatch_tool(tool_name, tool_args)

            # Parse relevance out of KB output for quality scoring
            if tool_name == "query_knowledge_base":
                raw_kb_output += result
                for line in result.splitlines():
                    if "relevance:" in line:
                        try:
                            score_str = line.split("relevance:")[-1].strip().rstrip("]")
                            kb_distances.append(float(score_str))
                        except ValueError:
                            pass

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })


    return {
        "answer": "[max rounds reached]",
        "tools_called": tools_called,
        "first_tool": tools_called[0] if tools_called else None,
        "kb_distances": kb_distances,
        "raw_kb_output": raw_kb_output,
    }


# Scoring

def score_result(case: dict, result: dict) -> dict:
    """
    Compare one result against its ground truth case.

    Checks:
        1. routing_pass: first_tool ==expected_tool
        2. content_pass: expected_keyword found (case-insensitive) in KB output,
                         OR expected_keyword is null (web queries, no content check)
        3. best_distance: lowest distance score from KB calls (None if no KB call)
        4. distance_ok: best_distance < DISTANCE_WARN_THRESHOLD (or None)

    Returns a scored dict ready for reporting.
    """

    expected_tool = case["expected_tool"]
    expected_keyword = case.get("expected_keyword")

    routing_pass = result["first_tool"] == expected_tool

    if expected_keyword is None:
        # Web search queries — no content check, just routing
        content_pass = True

    else:
        content_pass = expected_keyword.lower() in result["raw_kb_output"].lower()

    best_distance = min(result["kb_distances"]) if result["kb_distances"] else None
    distance_ok = (best_distance > 0.0) if best_distance is not None else None

    overall_pass = routing_pass and content_pass

    return {
        "id": case["id"],
        "question": case["question"],
        "expected_tool": expected_tool,
        "actual_tool": result["first_tool"],
        "routing_pass": routing_pass,
        "content_pass": content_pass,
        "best_distance": best_distance,
        "distance_ok": distance_ok,
        "overall_pass": overall_pass,
    }

def print_report(scores: list[dict]):
    """
    Print a table of per-query results and aggregate metrics.

    The per-query table shows every case with pass/fail indicators.
    The summary block shows the four aggregate metrics.
    """
    print("\n" + "=" * 72)
    print(f"{'ID':<6} {'Tool OK':>8} {'Content':>8} {'Dist':>7} {'Pass':>6}  Question")
    print("-" * 72)

    for s in scores:
        dist_str = f"{s['best_distance']:.4f}" if s["best_distance"] is not None else "  N/A "
        dist_flag = ""
        if s["best_distance"] is not None and not s["distance_ok"]:
            dist_flag = " ⚠"

        print(
            f"{s['id']:<6} "
            f"{'✓' if s['routing_pass'] else '✗':>8} "
            f"{'✓' if s['content_pass'] else '✗':>8} "
            f"{dist_str:>7}{dist_flag:<2} "
            f"{'✓' if s['overall_pass'] else '✗':>6}  "
            f"{s['question'][:52]}"
        )

    print("=" * 72)

    n = len(scores)
    routing_acc = sum(s["routing_pass"] for s in scores) / n
    content_acc = sum(s["content_pass"] for s in scores) / n
    overall_acc = sum(s["overall_pass"] for s in scores) / n

    kb_scores = [s for s in scores if s["best_distance"] is not None]
    mean_dist = sum(s["best_distance"] for s in kb_scores) / len(kb_scores) if kb_scores else None

    print(f"\nSummary ({n} queries)")
    print(f"  Tool routing accuracy : {routing_acc:.0%}")
    print(f"  Content hit rate      : {content_acc:.0%}")
    print(f"  Overall pass rate     : {overall_acc:.0%}")
    if mean_dist is not None:
        print(f"  Mean best KB relevance : {mean_dist:.4f}")
    print()


# Entry point

def main():
    cases = json.loads(EVAL_PATH.read_text())

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )

    print(f"Running {len(cases)} eval queries...\n")

    scores = []
    for case in cases:
        print(f"  [{case['id']}] {case['question'][:60]}")
        result = run_eval_turn(client=client, question=case["question"])
        score = score_result(case=case, result=result)
        scores.append(score)

        # Inline status to watch progress
        routing_marker = "✓" if score["routing_pass"] else "✗"
        content_marker = "✓" if score["content_pass"] else "✗"
        dist_str = f"{score['best_distance']:.4f}" if score["best_distance"] else "N/A"
        print(f"         routing:{routing_marker}  content:{content_marker}  dist:{dist_str}")

    print_report(scores)



if __name__ == "__main__":
    main()








