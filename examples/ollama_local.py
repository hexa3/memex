"""Ollama example with local memex memory.

Requires a local Ollama server.
"""

from __future__ import annotations

import json
import urllib.request

from memex import Memory


def ollama_chat(system: str, user_msg: str) -> str:
    payload = json.dumps(
        {
            "model": "llama3.2",
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["message"]["content"]


def main() -> None:
    mem = Memory(namespace="ollama-local")
    user_msg = input("You: ")
    assistant_msg = ollama_chat(mem.inject_system(user_msg), user_msg)
    print(f"Assistant: {assistant_msg}")
    mem.learn(user_msg, assistant_msg)


if __name__ == "__main__":
    main()
