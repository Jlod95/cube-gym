import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import feedback_audit  # noqa: E402


@pytest.fixture(scope="module")
def audits(tmp_path_factory):
    return feedback_audit.audit_all(tmp_path_factory.mktemp("raw"))


def test_feedback_text_matches_harness_format():
    from cube import SOLVED

    text = feedback_audit.feedback_text("OK: turned U.", SOLVED)
    assert text.startswith("OK: turned U. stages complete: 7/7")
    assert feedback_audit.read_feedback(text) == (7, 9)


def test_every_task_audited(audits):
    assert len(audits) == 6 * 20
    assert all(a["start_stage"] == a["stage"] - 1 for a in audits)


@pytest.mark.parametrize("stage", range(2, 7))
def test_stage_count_never_drops_in_stages_2_to_6(audits, stage):
    for a in audits:
        if a["stage"] != stage:
            continue
        for i, step in enumerate(a["steps"], 1):
            assert step["valid"], f"stage {stage} seed {a['seed']} action {i} invalid"
            assert not step["stage_drop"], (
                f"stage {stage} seed {a['seed']}: stage count dropped after action {i} "
                f"{step['action']}")


@pytest.mark.parametrize("stage", range(2, 8))
def test_every_task_ends_at_target(audits, stage):
    for a in audits:
        if a["stage"] != stage:
            continue
        assert a["steps"][-1]["stages"] >= stage, f"stage {stage} seed {a['seed']}"
        summary = a["summary"]
        assert summary["target_reached"], f"stage {stage} seed {a['seed']}"
        assert summary["stop_reason"] == ("solved" if stage == 7 else "target_reached")
        assert summary["invalid"] == 0
