"""
EVAL RUNNER
------------
Job: run every case in eval_dataset.py through our real pipeline,
grade each answer against the expected answer, and print a score.
"""

import sys
import os

# Let this script import from src/, since tests/ is a sibling folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_dataset import HAND_WRITTEN_CASES


def grade_answer(actual_answer: str, expected_answer: str, client, model: str) -> bool:
    """
    Use an LLM-as-judge to check if actual_answer conveys the same
    factual content as expected_answer. Fuzzy, not exact string match -
    "18 days" and "You get 18 days of PTO per year" should both pass.
    """
    judge_prompt = f"""Does the ACTUAL answer contain the same key fact(s) as the EXPECTED answer? Minor wording differences are fine - only check factual content.

EXPECTED: {expected_answer}
ACTUAL: {actual_answer}

Respond with ONLY one word: "PASS" or "FAIL"."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )

    verdict = response.choices[0].message.content.strip().upper()
    return "PASS" in verdict and "FAIL" not in verdict

def run_eval():
    # Import our real pipeline pieces
    os.chdir(os.path.join(os.path.dirname(__file__), "..", "src"))  # so relative paths (DB, .env) resolve correctly
    from main import ask
    from generate import client, MODEL

    results = []

    for case in HAND_WRITTEN_CASES:
        question = case["question"]
        expected = case["expected_answer"]

        answer, _chunks = ask(question)
        passed = grade_answer(answer, expected, client, MODEL)

        results.append({
            "question": question,
            "expected": expected,
            "actual": answer,
            "passed": passed,
            "category": case["category"],
        })

    # Print a readable report
    print("\n" + "=" * 60)
    print("EVAL RESULTS")
    print("=" * 60)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n{status} [{r['category']}]")
        print(f"  Q: {r['question']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Actual:   {r['actual']}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 60)
    print(f"SCORE: {passed_count}/{total} passed ({passed_count/total*100:.0f}%)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_eval()