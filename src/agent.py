"""
AGENT / TOOL-CALLING
----------------------
Job: let the LLM decide, based on the question, whether it needs to call
a real function (a "tool") to get information our documents can't provide,
then feed the tool's result back to produce the final answer.
"""


from datetime import date, timedelta


def business_days_until(target_date_str: str) -> str:
    """
    Calculate how many business days (Mon-Fri) remain between today
    and a given target date (format: YYYY-MM-DD).
    """
    try:
        target = date.fromisoformat(target_date_str)
    except ValueError:
        return f"Error: '{target_date_str}' is not a valid date in YYYY-MM-DD format."

    today = date.today()
    if target < today:
        return "Error: target date is in the past."

    count = 0
    current = today
    while current < target:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0 ... Sunday=6, so <5 means Mon-Fri
            count += 1

    return f"{count} business days between today ({today.isoformat()}) and {target_date_str}."


def get_current_date() -> str:
    """The actual Python function that does the real work."""
    return date.today().isoformat()  # e.g. "2026-09-24"


# This is the "menu" we hand to the LLM - a structured description of
# what tools exist, in the exact format Groq's API expects (this format
# is shared across OpenAI-compatible APIs, so it's a transferable skill).
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get today's date. Use this when the question depends on knowing the current date.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "business_days_until",
            "description": "Calculate the number of business days (Mon-Fri) between today and a target date. Use this for questions about deadlines, notice periods, or 'how many business days until X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_date_str": {
                        "type": "string",
                        "description": "The target date in YYYY-MM-DD format",
                    }
                },
                "required": ["target_date_str"],
            },
        },
    },
]

AVAILABLE_TOOLS = {
    "get_current_date": get_current_date,
    "business_days_until": business_days_until,
}

def ask_with_tools(question: str, client, model: str) -> str:
    """
    Send the question to the LLM along with our tool menu. If the model
    decides it needs a tool, run it and send the result back for a
    final answer. If not, just return its direct answer.
    """
    messages = [{"role": "user", "content": question}]

    # First call: give the model the question AND the tool menu.
    # The model can either answer directly, or ask to call a tool.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )

    response_message = response.choices[0].message

    # If the model didn't ask for any tool, it answered directly - done.
    if not response_message.tool_calls:
        return response_message.content

    # The model DID ask for a tool. Append its request to the conversation,
    # so the LLM's own "I want to call X" message is part of the history.
    messages.append(response_message)

    # There can be multiple tool calls requested at once - handle each.
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_to_run = AVAILABLE_TOOLS.get(function_name)

        if function_to_run is None:
            result = f"Error: unknown tool '{function_name}'"
        else:
            import json
            arguments = json.loads(tool_call.function.arguments)
            result = function_to_run(**arguments) 

        # Feed the tool's real output back into the conversation,
        # tagged with which tool_call it's answering.
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    # Second call: now the model has the tool's real result and can
    # write the actual final answer using it.
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return final_response.choices[0].message.content


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    print(ask_with_tools("What is today's date?", client, "openai/gpt-oss-20b"))
    print(ask_with_tools("What is 2+2?", client, "openai/gpt-oss-20b"))
    print(ask_with_tools("How many business days until 2026-12-01?", client, "openai/gpt-oss-20b"))