"""Bout en bout : fascicule synthétique → doublure → contrat → ALTO/METS validés → QC → relecture.

C'est le chemin qu'emprunte le DAG et le service, exécuté ici sans Airflow ni HTTP.
"""

import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from finlam_prod.config import Settings
from finlam_prod.convert.alto import validate_alto, write_alto
from finlam_prod.convert.mets import validate_mets, write_mets
from finlam_prod.data.mets_alto import read_mets_alto
from finlam_prod.data.schema import Issue, group_by_issue
from finlam_prod.evaluation.run import evaluate_predictions
from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier
from finlam_prod.pipeline.run import Pipeline, read_issue_json, write_issue_json
from finlam_prod.qc.checks import check_issue, summarise
from finlam_prod.service.api import create_app


@pytest.fixture(scope="module")
def trained(train_pages):
    return Pipeline(
        GradientBoostingZoneClassifier().fit(train_pages),
        segmenter=BoundarySegmenter().fit(train_pages),
    )


def test_end_to_end_issue(tmp_path, trained, test_pages):
    issue = group_by_issue(test_pages)[0]
    # 1. prédiction → contrat JSON sur disque, relu tel quel
    predicted = trained.run_issue(issue)
    contract = write_issue_json(predicted, tmp_path / "pipeline.json", trained.name)
    reloaded = read_issue_json(contract)
    assert [p.id for p in reloaded.pages] == [p.id for p in issue.pages]
    # 2. conversion validée
    xsd_errors = 0
    for page in reloaded.pages:
        xsd_errors += len(validate_alto(write_alto(page, tmp_path / f"f{page.index}.alto.xml")))
    mets = write_mets(reloaded, tmp_path / "mets.xml")
    xsd_errors += len(validate_mets(mets))
    assert xsd_errors == 0
    # 3. le METS/ALTO produit se relit en la même structure
    back = read_mets_alto(mets)
    assert len(back.issue.pages) == len(issue.pages)
    assert sum(len(p.zones) for p in back.issue.pages) == sum(len(p.zones) for p in issue.pages)
    # 4. QC : résumé exploitable et routage
    summary = summarise(check_issue(reloaded), len(reloaded.pages))
    assert {"errors", "warnings", "pages_with_errors"} <= set(summary)
    route = "review" if summary["errors"] > 0 else "archive"
    assert route in ("archive", "review")
    # 5. évaluation en chaîne contre la vérité terrain, format du consortium
    report = evaluate_predictions(issue.pages, reloaded.pages, n_boot=20)
    assert {"as_f1", "jaccard_article", "bleu_block"} <= set(report.summary)
    assert 0.0 <= report.summary["as_f1"][0] <= 1.0


def test_end_to_end_through_service(trained, test_pages, tmp_path):
    """Même chemin via HTTP : /issues/predict puis /issues → zip exploitable."""
    issue: Issue = group_by_issue(test_pages)[0]
    client = TestClient(create_app(settings=Settings.from_env({}, dotenv=None)))
    from finlam_prod.pipeline.run import page_to_json

    payload = {"issue_id": issue.id, "pages": [page_to_json(p) for p in issue.pages]}
    predicted = client.post("/issues/predict", json=payload)
    assert predicted.status_code == 200
    archive = client.post("/issues", json=predicted.json())
    assert archive.status_code == 200
    assert archive.headers["X-XSD-Errors"] == "0"
    zip_path = tmp_path / "out.zip"
    zip_path.write_bytes(archive.content)
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        assert "mets.xml" in names and "qc_summary.json" in names
        assert any(n.endswith(".alto.xml") for n in names)
        summary = json.loads(z.read("qc_summary.json"))
        assert summary["mets_xsd_errors"] == 0
