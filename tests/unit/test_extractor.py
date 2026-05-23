from memex.extractor import RuleBasedExtractor


def test_rule_based_extractor_finds_preferences() -> None:
    extractor = RuleBasedExtractor()

    facts = extractor.extract("My name is Sam. I prefer concise answers.", "Noted.")

    assert "User's name is Sam." in facts
    assert "User prefers concise answers." in facts


def test_rule_based_extractor_returns_empty_for_transient_turn() -> None:
    extractor = RuleBasedExtractor()

    assert extractor.extract("What is the weather today?", "Sunny.") == []
