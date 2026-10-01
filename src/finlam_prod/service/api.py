"""Service HTTP : le contrat JSON entre, ALTO/METS validés et rapport QC sortent.

Le service ne contient aucun modèle lourd : il reçoit les sorties du
prototype (ou de la doublure) et fait la partie « archivage » de la
chaîne. C'est volontaire : la conversion et le contrôle qualité doivent
pouvoir tourner sans GPU, dans un conteneur léger, et être appelés par
un orchestrateur (voir ``dags/finlam_pipeline.py``).

Points d'entrée :

- ``GET  /health``            : état du service et version ;
- ``POST /issues``            : contrat JSON d'un fascicule → archive zip
  (ALTO par page, METS, rapport QC) ;
- ``POST /issues/qc``         : contrat JSON → rapport QC seul (JSON) ;
- ``POST /issues/predict``    : zones sans structure (JSON) → contrat JSON
  structuré par la doublure (utile pour tester la chaîne sans prototype).

Lancer : ``finlam-prod serve`` ; documentation OpenAPI sur ``/docs``.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from finlam_prod import __version__
from finlam_prod.convert.alto import validate_alto, write_alto
from finlam_prod.convert.mets import validate_mets, write_mets
from finlam_prod.data.schema import Issue
from finlam_prod.pipeline.run import Pipeline, page_from_json, page_to_json
from finlam_prod.qc.checks import check_issue, findings_table, summarise


class IssuePayload(BaseModel):
    """Contrat JSON d'un fascicule (voir :func:`finlam_prod.pipeline.run.write_issue_json`)."""

    issue_id: str
    name: str | None = None
    pages: list[dict] = Field(..., description="Une entrée par page, au format page_to_json")


def _issue_from_payload(payload: IssuePayload) -> Issue:
    try:
        pages = [page_from_json(p) for p in payload.pages]
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Contrat JSON invalide : {exc}") from exc
    return Issue(id=payload.issue_id, name=payload.name or payload.issue_id, pages=pages)


def create_app(models_dir: Path | None = None) -> FastAPI:
    """Construit l'application ; ``models_dir`` charge la doublure entraînée pour /predict."""
    app = FastAPI(
        title="finlam-prod",
        version=__version__,
        description="Conversion ALTO/METS, contrôle qualité et doublure du pipeline FINLAM.",
    )
    pipeline = Pipeline()
    if models_dir is not None:
        from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier

        pipeline = Pipeline(
            GradientBoostingZoneClassifier.load(models_dir / "zone_classifier.joblib"),
            segmenter=BoundarySegmenter.load(models_dir / "segmenter.joblib"),
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "pipeline": pipeline.name}

    @app.post("/issues/qc")
    def issue_qc(payload: IssuePayload) -> dict:
        issue = _issue_from_payload(payload)
        findings = check_issue(issue)
        return {
            "summary": summarise(findings, len(issue.pages)),
            "findings": findings_table(findings).to_dict(orient="records"),
        }

    @app.post("/issues/predict")
    def issue_predict(payload: IssuePayload) -> dict:
        issue = pipeline.run_issue(_issue_from_payload(payload))
        return {
            "issue_id": issue.id,
            "name": issue.name,
            "pages": [page_to_json(p, pipeline.name) for p in issue.pages],
        }

    @app.post("/issues")
    def issue_convert(payload: IssuePayload) -> StreamingResponse:
        issue = _issue_from_payload(payload)
        buffer = io.BytesIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive,
        ):
            target = Path(tmp)
            alto_errors = 0
            for page in issue.pages:
                path = write_alto(page, target / f"f{page.index}.alto.xml")
                alto_errors += len(validate_alto(path))
                archive.write(path, path.name)
            mets_path = write_mets(issue, target / "mets.xml")
            mets_errors = validate_mets(mets_path)
            archive.write(mets_path, mets_path.name)
            findings = check_issue(issue)
            summary = summarise(findings, len(issue.pages))
            summary.update({"alto_xsd_errors": alto_errors, "mets_xsd_errors": len(mets_errors)})
            archive.writestr("qc_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
            archive.writestr("qc_findings.csv", findings_table(findings).to_csv(index=False))
        buffer.seek(0)
        headers = {
            "Content-Disposition": f'attachment; filename="{issue.id}.zip"',
            "X-QC-Errors": str(summary["errors"]),
            "X-XSD-Errors": str(alto_errors + len(mets_errors)),
        }
        return StreamingResponse(buffer, media_type="application/zip", headers=headers)

    return app
