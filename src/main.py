"""
MAIN — the chat loop
----------------------
Ties together: semantic cache -> retrieval -> grounded generation
(with conversation memory) -> guardrail checks -> caching the result.
"""

from retrieve import retrieve, embedder
from generate import generate, client, MODEL, gateway
from guardrails import check_refused_appropriately, check_groundedness
from cache import SemanticCache
from memory import ConversationMemory

cache = SemanticCache(embedder, similarity_threshold=0.45)
memory = ConversationMemory(max_turns=5)


def ask(question: str):
    # 1. Retrieve FIRST now - we need chunk IDs to check the cache properly
    chunks = retrieve(question)
    chunk_ids = {f"{c['source']}-{c.get('chunk_index', c['text'][:20])}" for c in chunks}

    # 2. Now check cache using both similarity AND chunk overlap
    cached = cache.lookup(question, chunk_ids)
    if cached:
        print(f"⚡ [cache hit] matched: \"{cached['matched_question']}\" "
              f"(score={cached['score']:.3f}, overlap={cached['overlap_ratio']:.2f})")
        return cached["answer"], chunks

    # 3. Cache miss - generate a grounded answer
    answer = generate(question, chunks, memory=memory)

    # 4. Guardrail checks
    refused_ok = check_refused_appropriately(question, chunks, answer)
    if not refused_ok:
        print("⚠️  [guardrail] Model may have answered without sufficient context")

    ground_check = check_groundedness(question, chunks, answer, client, MODEL)
    if not ground_check["grounded"]:
        print(f"⚠️  [guardrail] Answer may not be fully grounded (verdict: {ground_check['raw_verdict']})")

    # 5. Store with chunk IDs this time
    cache.store(question, answer, chunk_ids)
    memory.add(question, answer)

    return answer, chunks


if __name__ == "__main__":
    print("RAG Assistant — ask a question (type 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            gateway.summary()
            break
        if not question:
            continue

        answer, chunks = ask(question)
        print("\nAssistant:", answer)
        print()