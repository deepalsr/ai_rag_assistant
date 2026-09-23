from retrieve import retrieve
from generate import generate


def ask(question: str):
    chunks = retrieve(question)
    answer = generate(question, chunks)
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