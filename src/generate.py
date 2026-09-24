"""
GENERATION
-----------
Job: take retrieved chunks + (optionally) recent conversation history,
build a grounded prompt, and call the LLM through our gateway
(rate limiting, fallback routing, cost tracking all handled there).
"""

import os
from dotenv import load_dotenv
from groq import Groq
from gateway import RateLimiter, LLMGateway

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"  # kept for guardrails.py's direct LLM-as-judge call

rate_limiter = RateLimiter(max_calls=20, per_seconds=60)  # 20 calls per minute
gateway = LLMGateway(
    client=client,
    models=["openai/gpt-oss-20b", "openai/gpt-oss-120b"],  # primary, then fallback
    rate_limiter=rate_limiter,
)


def generate(question: str, chunks: list[dict], memory=None) -> str:
    """
    Take retrieved chunks (as returned by retrieve.py) and ask the LLM
    to answer the question using ONLY that context, optionally aware
    of recent conversation history for follow-up questions.
    """
    context = "\n\n---\n\n".join(c["text"] for c in chunks)

    history_block = ""
    if memory is not None:
        history_text = memory.as_context_string()
        if history_text:
            history_block = f"""Previous conversation (for context on follow-up questions):
{history_text}

"""

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't have enough information to answer that."

{history_block}Context:
{context}

Question: {question}

Answer:"""

    result = gateway.call(prompt, temperature=0.1)
    return result["text"]


if __name__ == "__main__":
    from retrieve import retrieve

    question = "how many vacation days do I get?"
    chunks = retrieve(question)
    answer = generate(question, chunks)

    print("Question:", question)
    print("Answer:", answer)