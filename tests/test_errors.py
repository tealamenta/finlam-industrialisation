"""Les erreurs attendues portent un message utile et sortent par la bonne porte."""

import json

import pytest
from fastapi.testclient import TestClient

from finlam_prod.errors import ContractError, DatasetError, FinlamError
from finlam_prod.pipeline.run import page_from_json, page_to_json, read_issue_json
from finlam_prod.service.api import create_app


def test_contract_missing_key_names_page_and_field(simple_page):
    payload = page_to_json(simple_page)
    del payload["zones"][0]["polygon"]
    with pytest.raises(ContractError) as info:
        page_from_json(payload)
    assert "polygon" in str(info.value)
    assert simple_page.id in str(info.value)
    assert isinstance(info.value, FinlamError)


def test_contract_unknown_class_and_bad_polygon(simple_page):
    payload = page_to_json(simple_page)
    payload["zones"][0]["class"] = "PAS-UNE-CLASSE"
    with pytest.raises(ContractError, match="classe inconnue"):
        page_from_json(payload)
    payload = page_to_json(simple_page)
    payload["zones"][0]["polygon"] = [[0, 0], [1, 1]]
    with pytest.raises(ContractError, match="trois points"):
        page_from_json(payload)


def test_contract_wrong_type_is_reported(simple_page):
    payload = page_to_json(simple_page)
    payload["page_index"] = "deux"
    with pytest.raises(ContractError, match="page_index"):
        page_from_json(payload)


def test_read_issue_json_rejects_invalid_json(tmp_path):
    bad = tmp_path / "pipeline.json"
    bad.write_text("{pas du json", encoding="utf-8")
    with pytest.raises(ContractError, match="JSON invalide"):
        read_issue_json(bad)
    missing = tmp_path / "absent.json"
    with pytest.raises(ContractError, match="illisible"):
        read_issue_json(missing)
    bad.write_text(json.dumps({"issue_id": "x"}), encoding="utf-8")
    with pytest.raises(ContractError, match="pages"):
        read_issue_json(bad)


def test_dataset_error_when_cache_missing(tmp_path):
    from finlam_prod.data.finlam import load_split

    with pytest.raises(DatasetError, match="download"):
        load_split("test", tmp_path, "la-liberte")


def test_service_maps_contract_error_to_422(simple_page):
    client = TestClient(create_app())
    payload = page_to_json(simple_page)
    del payload["zones"][0]["id"]
    response = client.post("/issues/qc", json={"issue_id": "i", "pages": [payload]})
    assert response.status_code == 422
    assert "Contrat JSON invalide" in response.json()["detail"]


def test_cli_prints_clean_error_and_exits_1(tmp_path):
    from typer.testing import CliRunner

    from finlam_prod.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["qc", str(tmp_path / "absent.json")])
    assert result.exit_code != 0


def test_gallica_4xx_becomes_gallica_error():
    from unittest import mock

    from finlam_prod.data.gallica import GallicaClient
    from finlam_prod.errors import GallicaError

    client = GallicaClient(min_interval=0)
    fake = mock.Mock(status_code=403, url="https://gallica.bnf.fr/x")
    with (
        mock.patch.object(client.session, "get", return_value=fake),
        pytest.raises(GallicaError, match="HTTP 403"),
    ):
        client.pagination("bpt6k7155522")
