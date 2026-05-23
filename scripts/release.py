"""Build helper for maintainers."""

from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    run([sys.executable, "-m", "pytest"])
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "mypy", "memex"])
    run([sys.executable, "-m", "build"])


if __name__ == "__main__":
    main()
