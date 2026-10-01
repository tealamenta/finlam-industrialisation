"""Le DAG Airflow se charge et a la structure annoncée (test sauté sans Airflow)."""

import importlib.util
import sys
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")


def test_dag_structure(monkeypatch, tmp_path):
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "finlam_pipeline", Path(__file__).parents[1] / "dags" / "finlam_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["finlam_pipeline"] = module
    spec.loader.exec_module(module)
    dag = module.dag
    assert dag.dag_id == "finlam_pipeline"
    tasks = {t.task_id: t for t in dag.tasks}
    assert set(tasks) == {
        "fetch",
        "preprocess",
        "predict",
        "convert",
        "qc",
        "route",
        "archive",
        "review",
    }
    assert tasks["route"].downstream_task_ids == {"archive", "review"}
    assert tasks["fetch"].downstream_task_ids == {"preprocess"}
    assert tasks["preprocess"].downstream_task_ids == {"predict"}
