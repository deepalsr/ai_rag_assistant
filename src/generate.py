import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-20b"

def generate(question: str, chunks: list[dict]) -> str:
    """
    Take retrieved chunks (as returned by retrieve.py) and ask the LLM
    to answer the question using ONLY that context.
    """
    context = "\n\n---\n\n".join(c["text"] for c in chunks)

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from retrieve import retrieve

    question = "how many vacation days do I get?"
    chunks = retrieve(question)
    answer = generate(question, chunks)

    print("Question:", question)
    print("Answer:", answer)