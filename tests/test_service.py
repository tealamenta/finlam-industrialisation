import io
import zipfile

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from finlam_prod.data.schema import group_by_issue  # noqa: E402
from finlam_prod.pipeline.run import Pipeline, page_to_json  # noqa: E402
from finlam_prod.service.api import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def _payload(pages, name="Test"):
    return {"issue_id": pages[0].issue_id, "name": name, "pages": [page_to_json(p) for p in pages]}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_then_convert(client, test_pages):
    issue = group_by_issue(test_pages)[0]
    raw = _payload(issue.pages)
    for page in raw["pages"]:  # on retire la structure : le service doit la reconstruire
        for z in page["zones"]:
            z["class"] = None
            z["reading_order"] = None
            z["article_id"] = None
    predicted = client.post("/issues/predict", json=raw)
    assert predicted.status_code == 200
    body = predicted.json()
    assert all(z["article_id"] is not None for p in body["pages"] for z in p["zones"])

    converted = client.post("/issues", json=body)
    assert converted.status_code == 200
    assert converted.headers["X-XSD-Errors"] == "0"
    names = zipfile.ZipFile(io.BytesIO(converted.content)).namelist()
    assert "mets.xml" in names and "qc_summary.json" in names
    assert sum(n.endswith(".alto.xml") for n in names) == len(issue.pages)


def test_qc_endpoint_and_invalid_payload(client, test_pages):
    issue = Pipeline().run_issue(group_by_issue(test_pages)[0])
    r = client.post("/issues/qc", json=_payload(issue.pages))
    assert r.status_code == 200 and "summary" in r.json()
    bad = client.post("/issues/qc", json={"issue_id": "x", "pages": [{"nope": 1}]})
    assert bad.status_code == 422
