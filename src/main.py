from retrieve import retrieve
from generate import generate, client, MODEL
from guardrails import check_refused_appropriately, check_groundedness


def ask(question: str):
    chunks = retrieve(question)
    answer = generate(question, chunks)

    # Guardrail 1: cheap rule-based check
    refused_ok = check_refused_appropriately(question, chunks, answer)
    if not refused_ok:
        print("⚠️  [guardrail] Model may have answered without sufficient context")

    # Guardrail 2: LLM-as-judge groundedness check
    ground_check = check_groundedness(question, chunks, answer, client, MODEL)
    if not ground_check["grounded"]:
        print(f"⚠️  [guardrail] Answer may not be fully grounded (verdict: {ground_check['raw_verdict']})")

    return answer, chunks


if __name__ == "__main__":
    print("RAG Assistant — ask a question (type 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, chunks = ask(question)
        print("\nAssistant:", answer)
        print()