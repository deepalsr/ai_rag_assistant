from retrieve import retrieve
from generate import generate, client, MODEL
from guardrails import check_refused_appropriately, check_groundedness
from cache import SemanticCache
from retrieve import embedder  

cache = SemanticCache(embedder, similarity_threshold=0.2)
def ask(question: str):
    # 1. Check cache first
    cached = cache.lookup(question)
    if cached:
        print(f"⚡ [cache hit] matched: \"{cached['matched_question']}\" (score={cached['score']:.3f})")
        return cached["answer"], []

    # 2. Cache miss — do the real retrieve + generate work
    chunks = retrieve(question)
    answer = generate(question, chunks)

    refused_ok = check_refused_appropriately(question, chunks, answer)
    if not refused_ok:
        print("⚠️  [guardrail] Model may have answered without sufficient context")

    ground_check = check_groundedness(question, chunks, answer, client, MODEL)
    if not ground_check["grounded"]:
        print(f"⚠️  [guardrail] Answer may not be fully grounded (verdict: {ground_check['raw_verdict']})")

    # 3. Store for next time
    cache.store(question, answer)

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