def check_refused_appropriately(question: str, chunks: list, answer: str) -> bool:
    """
    If retrieval found nothing relevant (chunks are all very far away),
    the LLM SHOULD have said it doesn't know. This catches cases where
    it answered anyway from its own training data instead.
    """
    NO_CONTEXT_PHRASES = [
        "don't have enough information",
        "cannot find", "no information", "not mentioned",
    ]

    # If chunks are all weak matches (high distance = not similar)
    all_weak = all(c["distance"] > 1.8 for c in chunks) if chunks else True

    said_dont_know = any(phrase in answer.lower() for phrase in NO_CONTEXT_PHRASES)

    if all_weak and not said_dont_know:
        return False  # FAILED: should have refused, but didn't
    return True  # OK

def check_groundedness(question: str, chunks: list, answer: str, client, model: str) -> dict:
    """
    Ask a SEPARATE LLM call to judge whether the answer is actually
    supported by the retrieved context. This catches subtler hallucinations
    that the cheap rule-based check would miss (e.g. the model added an
    extra fact that sounds plausible but isn't in the context).
    """
    context = "\n\n".join(c["text"] for c in chunks)

    judge_prompt = f"""You are a fact-checker. Given a context and an answer, determine if the answer is FULLY supported by the context.

If the answer is a refusal (e.g. saying it doesn't have enough information), that counts as GROUNDED — refusing when the context is insufficient is the correct behavior, not a hallucination.

Context:
{context}

Answer to check:
{answer}

Respond with ONLY one word: "GROUNDED" if every factual claim in the answer is supported by the context OR the answer is an appropriate refusal, or "UNGROUNDED" if the answer contains factual claims not present in the context."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
    )

    verdict = response.choices[0].message.content.strip().upper()

    return {
        "grounded": "GROUNDED" in verdict and "UNGROUNDED" not in verdict,
        "raw_verdict": verdict,
    }