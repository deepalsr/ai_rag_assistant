"""
CONVERSATION MEMORY
---------------------
Job: keep a running history of the conversation so follow-up questions
("can I carry them over?") can be understood in context.
"""


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        """
        max_turns: how many past question-answer pairs to remember.
        Keeping this bounded matters - unlimited history eventually
        overflows the model's context window and costs more tokens
        on every single call.
        """
        self.max_turns = max_turns
        self.history = []  # list of {"question": str, "answer": str}

    def add(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_turns:
            self.history.pop(0)  # drop the oldest turn

    def as_context_string(self) -> str:
        """Format history into text we can prepend to a new prompt."""
        if not self.history:
            return ""

        lines = []
        for turn in self.history:
            lines.append(f"Q: {turn['question']}")
            lines.append(f"A: {turn['answer']}")
        return "\n".join(lines)