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


def test_api_key_required_when_configured(simple_page):
    from finlam_prod.config import Settings
    from finlam_prod.pipeline.run import page_to_json

    settings = Settings.from_env({"FINLAM_API_KEY": "secret"}, dotenv=None)
    client = TestClient(create_app(settings=settings))
    body = {"issue_id": "i", "pages": [page_to_json(simple_page)]}
    assert client.get("/health").status_code == 200  # santé toujours ouverte
    assert client.post("/issues/qc", json=body).status_code == 401
    assert client.post("/issues/qc", json=body, headers={"X-API-Key": "faux"}).status_code == 401
    assert client.post("/issues/qc", json=body, headers={"X-API-Key": "secret"}).status_code == 200


def test_size_limits_return_413(simple_page):
    from finlam_prod.config import Settings
    from finlam_prod.pipeline.run import page_to_json

    settings = Settings.from_env({"FINLAM_MAX_PAGES": "1"}, dotenv=None)
    client = TestClient(create_app(settings=settings))
    one = page_to_json(simple_page)
    assert (
        client.post("/issues/qc", json={"issue_id": "i", "pages": [one, one]}).status_code == 413
    )


def test_metrics_endpoint_counts_requests(simple_page):
    from finlam_prod.pipeline.run import page_to_json

    client = TestClient(create_app())
    client.post("/issues/qc", json={"issue_id": "i", "pages": [page_to_json(simple_page)]})
    text = client.get("/metrics").text
    assert 'finlam_requests_total{method="POST",route="/issues/qc",status="200"} 1' in text
    assert "finlam_pages_total 1" in text
