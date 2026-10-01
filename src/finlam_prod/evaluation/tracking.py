"""Traçage des évaluations avec MLflow (optionnel).

Chaque appel à ``log_report`` crée un run : paramètres (jeu, split,
pipeline, nombre de pages), métriques (moyenne et bornes de l'IC95 %
pour chacune), artefacts (tables CSV et Markdown). Sans MLflow installé,
la fonction n'a aucun effet et le reste de la chaîne fonctionne.

    finlam-prod evaluate --dataset la-liberte   # trace automatiquement si mlflow est présent
    mlflow ui                                    # http://localhost:5000
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def log_report(report, params: dict[str, object], experiment: str = "finlam-prod") -> str | None:
    """Enregistre un ``EvaluationReport`` dans MLflow ; renvoie l'identifiant du run."""
    try:
        import mlflow
    except ImportError:
        LOGGER.info("mlflow absent : évaluation non tracée (pip install 'finlam-prod[tracking]')")
        return None
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_param("n_pages", report.n_pages)
        for name, (mean, low, high) in report.summary.items():
            mlflow.log_metric(name, mean)
            mlflow.log_metric(f"{name}_ci_low", low)
            mlflow.log_metric(f"{name}_ci_high", high)
        mlflow.log_metric("seconds_per_page", report.seconds_per_page)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            report.per_page.to_csv(target / "per_page.csv", index=False)
            (target / "summary.md").write_text(report.markdown(), encoding="utf-8")
            mlflow.log_artifacts(str(target))
        LOGGER.info("Évaluation tracée dans MLflow, run %s", run.info.run_id)
        return run.info.run_id
