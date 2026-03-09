from utils import checkpoint as checkpoint_utils


def _sample_state(version: int) -> dict:
    return {
        "best_score": float(version),
        "best_version": version,
        "best_code_file": f"v{version}.py",
        "version_scores": {str(version): float(version)},
        "successful_versions": [version],
        "blacklisted_versions": [],
        "blacklisted_ideas": [],
        "successful_ideas": [],
        "global_suggestions": [],
        "input_list": [f"input-{version}"],
        "last_input_tokens": version,
        "last_suggestion": f"suggestion-{version}",
        "sota_suggestions_call_id": version,
    }


def test_save_checkpoint_prunes_old_versions(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpoint_utils, "_CHECKPOINT_RETENTION", 3)

    db_path = tmp_path / "checkpoints.db"
    conn = checkpoint_utils.create_db(db_path)

    for version in range(1, 6):
        checkpoint_utils.save_checkpoint(
            conn,
            slug="s",
            iteration="1",
            model_name="m",
            version=version,
            state=_sample_state(version),
        )

    versions = [
        row[0]
        for row in conn.execute(
            "SELECT version FROM checkpoints WHERE slug = ? AND iteration = ? ORDER BY version",
            ("s", "1"),
        ).fetchall()
    ]

    assert versions == [3, 4, 5]
