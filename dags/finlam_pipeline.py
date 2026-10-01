"""DAG Airflow : un fascicule entre, ALTO/METS validés et contrôlés sortent.

Chaîne par fascicule :

    fetch → predict → convert → qc → route
                                       ├── archive   (QC sans erreur bloquante)
                                       └── review    (QC bloquant : relecture humaine)

- ``fetch``   : récupère le fascicule (ARK Gallica : images + ALTO via l'API, ou
  identifiant en cache) ;
- ``predict`` : appelle le prototype FINLAM s'il est déployé (variable Airflow
  ``FINLAM_PREDICT_URL``), sinon la doublure locale ;
- ``convert`` : contrat JSON → ALTO 4.4 + METS, validation XSD ;
- ``qc``      : contrôle qualité, résumé JSON ;
- ``route``   : branchement selon ``errors`` et ``xsd_errors``.

Le DAG ne porte aucun modèle : tout passe par ``finlam_prod`` (bibliothèque
installée dans l'environnement Airflow) ou par le service HTTP. Il est
paramétré à la demande (``dag_run.conf = {"source": "bpt6k…", "pages": "1,2"}``)
et peut être planifié sur une file de fascicules.

Exécution locale (Airflow ≥ 2.9) :

    export AIRFLOW_HOME=$PWD/airflow
    airflow standalone            # UI sur http://localhost:8080
    cp dags/finlam_pipeline.py $AIRFLOW_HOME/dags/
    airflow dags trigger finlam_pipeline --conf '{"source": "bpt6k7155522", "pages": "1,2"}'
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Les variables Airflow ont priorité ; sinon la configuration du paquet (FINLAM_* dans
# l'environnement ou .env), pour que le DAG, la CLI et le service lisent les mêmes valeurs.
from finlam_prod.config import get_settings  # noqa: E402

_SETTINGS = get_settings()
WORK_DIR = Path(Variable.get("FINLAM_WORK_DIR", default_var=str(_SETTINGS.work_dir)))
MODELS_DIR = Variable.get("FINLAM_MODELS_DIR", default_var=str(_SETTINGS.models_dir or ""))
PREDICT_URL = Variable.get("FINLAM_PREDICT_URL", default_var=_SETTINGS.predict_url)


def _target(context) -> Path:
    conf = context["dag_run"].conf or {}
    source = conf.get("source", "sample")
    return WORK_DIR / context["run_id"] / source


def fetch(**context) -> str:
    from finlam_prod.data.gallica import GallicaClient, normalise_ark
    from finlam_prod.data.schema import Issue, save_pages

    conf = context["dag_run"].conf or {}
    source = conf["source"]
    pages = [int(p) for p in str(conf.get("pages", "")).split(",") if p]
    target = _target(context)
    target.mkdir(parents=True, exist_ok=True)
    fetched = GallicaClient().fetch_issue(source, target / "gallica", pages=pages or None)
    issue = Issue(id=normalise_ark(source), name=f"Gallica {normalise_ark(source)}", pages=fetched)
    save_pages(issue.pages, target / "input.jsonl")
    return str(target)


def predict(**context) -> str:
    from finlam_prod.data.schema import Issue, load_pages
    from finlam_prod.pipeline.run import Pipeline, write_issue_json

    target = Path(context["ti"].xcom_pull(task_ids="fetch"))
    pages = load_pages(target / "input.jsonl")
    issue = Issue(id=target.name, name=target.name, pages=pages)
    if PREDICT_URL:
        import requests

        from finlam_prod.pipeline.run import page_to_json, read_issue_json

        payload = {
            "issue_id": issue.id,
            "name": issue.name,
            "pages": [page_to_json(p) for p in pages],
        }
        url = f"{PREDICT_URL.rstrip('/')}/issues/predict"
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        (target / "pipeline.json").write_text(json.dumps(response.json(), ensure_ascii=False))
        read_issue_json(target / "pipeline.json")  # validation du contrat renvoyé
    else:
        pipeline = Pipeline()
        if MODELS_DIR:
            from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier

            pipeline = Pipeline(
                GradientBoostingZoneClassifier.load(Path(MODELS_DIR) / "zone_classifier.joblib"),
                segmenter=BoundarySegmenter.load(Path(MODELS_DIR) / "segmenter.joblib"),
            )
        write_issue_json(pipeline.run_issue(issue), target / "pipeline.json", pipeline.name)
    return str(target / "pipeline.json")


def convert(**context) -> dict:
    from finlam_prod.convert.alto import validate_alto, write_alto
    from finlam_prod.convert.mets import validate_mets, write_mets
    from finlam_prod.pipeline.run import read_issue_json

    json_path = Path(context["ti"].xcom_pull(task_ids="predict"))
    issue = read_issue_json(json_path)
    errors = 0
    for page in issue.pages:
        alto_path = json_path.parent / f"f{page.index}.alto.xml"
        errors += len(validate_alto(write_alto(page, alto_path)))
    errors += len(validate_mets(write_mets(issue, json_path.parent / "mets.xml")))
    return {"xsd_errors": errors, "dir": str(json_path.parent)}


def qc(**context) -> dict:
    from finlam_prod.pipeline.run import read_issue_json
    from finlam_prod.qc.checks import check_issue, findings_table, summarise

    result = context["ti"].xcom_pull(task_ids="convert")
    target = Path(result["dir"])
    issue = read_issue_json(target / "pipeline.json")
    findings = check_issue(issue)
    findings_table(findings).to_csv(target / "qc_findings.csv", index=False)
    summary = summarise(findings, len(issue.pages))
    summary["xsd_errors"] = result["xsd_errors"]
    (target / "qc_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def route(**context) -> str:
    summary = context["ti"].xcom_pull(task_ids="qc")
    blocking = summary.get("errors", 0) > 0 or summary.get("xsd_errors", 0) > 0
    return "review" if blocking else "archive"


def archive(**context) -> None:
    """Dépôt vers l'entrepôt (ici : simple journalisation ; à brancher sur le SI cible)."""
    summary = context["ti"].xcom_pull(task_ids="qc")
    print(f"ARCHIVE OK — {json.dumps(summary, ensure_ascii=False)}")


def review(**context) -> None:
    """File de relecture humaine : on laisse les fichiers en place et on notifie."""
    summary = context["ti"].xcom_pull(task_ids="qc")
    print(f"RELECTURE REQUISE — {json.dumps(summary, ensure_ascii=False)}")


with DAG(
    dag_id="finlam_pipeline",
    description="Fascicule → prototype FINLAM (ou doublure) → ALTO/METS → QC → archive/relecture",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["finlam", "bnf", "olr"],
    default_args={"retries": 1},
) as dag:
    t_fetch = PythonOperator(task_id="fetch", python_callable=fetch)
    t_predict = PythonOperator(task_id="predict", python_callable=predict)
    t_convert = PythonOperator(task_id="convert", python_callable=convert)
    t_qc = PythonOperator(task_id="qc", python_callable=qc)
    t_route = BranchPythonOperator(task_id="route", python_callable=route)
    t_archive = PythonOperator(task_id="archive", python_callable=archive)
    t_review = PythonOperator(task_id="review", python_callable=review)

    t_fetch >> t_predict >> t_convert >> t_qc >> t_route >> [t_archive, t_review]
