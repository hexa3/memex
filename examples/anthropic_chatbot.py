"""Anthropic chatbot example with local memex memory.

Requires:
    pip install anthropic memex-ai
"""

from __future__ import annotations

from anthropic import Anthropic

from memex import Memory


def main() -> None:
    client = Anthropic()
    mem = Memory(namespace="anthropic-chatbot")
    user_msg = input("You: ")
    system = mem.inject_system(user_msg)
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    assistant_msg = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
    print(f"Assistant: {assistant_msg}")
    mem.learn(user_msg, assistant_msg)


if __name__ == "__main__":
    main()
