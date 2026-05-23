from memex import Memory


def main() -> None:
    mem = Memory(embedder="hash")
    mem.save("The user prefers short, concrete answers.")
    print(mem.inject("How should I explain the migration?"))


if __name__ == "__main__":
    main()
