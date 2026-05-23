"""OpenAI chatbot example with local memex memory.

Requires:
    pip install "memex-ai[openai]"
"""

from __future__ import annotations

from openai import OpenAI

from memex import Memory


def main() -> None:
    client = OpenAI()
    mem = Memory(namespace="openai-chatbot")
    user_msg = input("You: ")
    system = mem.inject_system(user_msg)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    assistant_msg = response.choices[0].message.content or ""
    print(f"Assistant: {assistant_msg}")
    mem.learn(user_msg, assistant_msg)


if __name__ == "__main__":
    main()
