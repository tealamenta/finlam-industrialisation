"""La tâche ``preprocess`` sur un fascicule : images corrigées, rapport, pages sans image."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from finlam_prod.data.schema import Issue, Page  # noqa: E402
from finlam_prod.preprocess.issue import (  # noqa: E402
    REPORT_FILE,
    SKEW_WARNING_DEG,
    preprocess_issue,
)
from finlam_prod.preprocess.synthetic import folded_page  # noqa: E402


@pytest.fixture
def issue(tmp_path: Path) -> Issue:
    """Trois pages : une pliée, une inclinée sans pli, une sans image."""
    folded = folded_page(seed=11)
    cv2.imwrite(str(tmp_path / "f1.jpg"), folded.image)
    tilted = folded_page(angles=(1.5, 1.5), shadow_depth=0, seed=12)
    cv2.imwrite(str(tmp_path / "f2.jpg"), tilted.image)
    pages = [
        Page(id="p1", issue_id="x", index=1, image_path=str(tmp_path / "f1.jpg")),
        Page(id="p2", issue_id="x", index=2, image_path=str(tmp_path / "f2.jpg")),
        Page(id="p3", issue_id="x", index=3, image_path=None),
    ]
    return Issue(id="x", name="exemple", pages=pages)


def test_preprocess_issue_corrects_images_and_reports(issue, tmp_path):
    corrected, report = preprocess_issue(issue, out_dir=tmp_path / "pre")
    by_id = {p.page_id: p for p in report.pages}
    assert by_id["p1"].fold_detected and len(by_id["p1"].angles) == 2
    assert not by_id["p2"].fold_detected
    assert abs(by_id["p2"].angles[0] + 1.5) <= 0.3 and by_id["p2"].flagged
    assert by_id["p3"].skipped == "pas d'image"
    assert report.summary == {
        "preprocessed_pages": 2,
        "skipped_pages": 1,
        "folded_pages": 1,
        "skewed_pages": 2,  # la page pliée a aussi ses moitiés inclinées de plus de 1°
        "dewarped_pages": 0,
        "preprocess_seconds": report.summary["preprocess_seconds"],
    }
    # les pages pointent sur les images corrigées, l'original est intact
    assert corrected.pages[0].image_path == str(tmp_path / "pre" / "f1_pre.png")
    assert Path(corrected.pages[0].image_path).is_file()
    assert corrected.pages[2].image_path is None
    assert issue.pages[0].image_path == str(tmp_path / "f1.jpg")
    # le rapport se sérialise et se relit
    path = report.write(tmp_path / REPORT_FILE)
    loaded = json.loads(path.read_text())
    assert loaded["summary"]["folded_pages"] == 1
    assert {p["page_id"] for p in loaded["pages"]} == {"p1", "p2", "p3"}


def test_preprocess_issue_tolerates_unreadable_image(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"rien")
    issue = Issue(id="y", name="y", pages=[Page("q", "y", 1, image_path=str(bad))])
    corrected, report = preprocess_issue(issue)
    assert report.pages[0].skipped == "image illisible"
    assert corrected.pages[0].image_path == str(bad)


def test_run_command_writes_preprocess_report(tmp_path, monkeypatch):
    """``finlam-prod run`` sur un fascicule en cache avec images → preprocess.json + résumé."""
    from typer.testing import CliRunner

    import finlam_prod.cli as cli
    from finlam_prod.data.schema import save_pages

    folded = folded_page(seed=13)
    cv2.imwrite(str(tmp_path / "f1.jpg"), folded.image)
    pages = [
        Page(
            id="p1",
            issue_id="demo",
            index=1,
            width_px=folded.image.shape[1],
            height_px=folded.image.shape[0],
            image_path=str(tmp_path / "f1.jpg"),
        )
    ]
    issue = Issue(id="demo", name="demo", pages=pages)
    save_pages(pages, tmp_path / "demo.jsonl")
    monkeypatch.setattr(cli, "_load_issue", lambda *a, **k: issue)
    result = CliRunner().invoke(cli.app, ["run", "demo", "--out-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["preprocess"]["folded_pages"] == 1
    assert (tmp_path / "out" / "demo" / REPORT_FILE).is_file()
    assert (tmp_path / "f1_pre.png").is_file()
    assert SKEW_WARNING_DEG > 0
