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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finlam_prod.evaluation.run import EvaluationReport
    from finlam_prod.pipeline.registry import ModelMetadata

LOGGER = logging.getLogger(__name__)


def log_report(
    report: EvaluationReport, params: dict[str, object], experiment: str = "finlam-prod"
) -> str | None:
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


def log_models(
    models_dir: str | Path, meta: ModelMetadata, experiment: str = "finlam-prod"
) -> str | None:
    """Enregistre un dossier de modèles (fichiers + metadata) comme run MLflow."""
    try:
        import mlflow
    except ImportError:
        LOGGER.info("mlflow absent : modèles non tracés")
        return None
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"train-{meta.model_version}") as run:
        mlflow.log_params(
            {
                "dataset": meta.dataset,
                "split": meta.split,
                "n_pages": meta.n_pages,
                "model_version": meta.model_version,
                "package_version": meta.package_version,
                "git_commit": meta.git_commit or "",
                "sklearn_version": meta.sklearn_version,
            }
        )
        mlflow.set_tag("stage", "train")
        mlflow.log_artifacts(str(models_dir), artifact_path="models")
        LOGGER.info("Modèles tracés dans MLflow, run %s", run.info.run_id)
        return run.info.run_id
