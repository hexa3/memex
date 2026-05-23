from memex.embedders import HashEmbedder, create_embedder


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dimension=64)

    first = embedder.embed("User prefers dark mode")
    second = embedder.embed("User prefers dark mode")

    assert first == second
    assert len(first) == 64


def test_create_embedder_hash() -> None:
    embedder = create_embedder("hash")

    assert embedder.dimension == 384
    assert len(embedder.embed("hello")) == 384
