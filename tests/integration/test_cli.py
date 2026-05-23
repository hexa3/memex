from typer.testing import CliRunner

from memex.cli import app


def test_cli_save_and_recall(tmp_path) -> None:
    runner = CliRunner()
    db = tmp_path / "memex.db"

    save_result = runner.invoke(
        app,
        ["save", "User prefers dark mode", "--db", str(db), "--embedder", "hash"],
    )
    recall_result = runner.invoke(
        app,
        ["recall", "dark mode", "--db", str(db), "--embedder", "hash"],
    )

    assert save_result.exit_code == 0
    assert recall_result.exit_code == 0
    assert "User prefers dark mode" in recall_result.output
