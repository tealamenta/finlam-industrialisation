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
import logging
import secrets
import tempfile
import time
import zipfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from finlam_prod import __version__
from finlam_prod.config import Settings, get_settings
from finlam_prod.convert.alto import validate_alto, write_alto
from finlam_prod.convert.mets import validate_mets, write_mets
from finlam_prod.data.schema import Issue
from finlam_prod.errors import ContractError, FinlamError
from finlam_prod.pipeline.run import Pipeline, page_from_json, page_to_json
from finlam_prod.qc.checks import check_issue, findings_table, summarise

LOGGER = logging.getLogger(__name__)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class IssuePayload(BaseModel):
    """Contrat JSON d'un fascicule (voir :func:`finlam_prod.pipeline.run.write_issue_json`)."""

    issue_id: str
    name: str | None = None
    pages: list[dict] = Field(..., description="Une entrée par page, au format page_to_json")


def _issue_from_payload(payload: IssuePayload, settings: Settings) -> Issue:
    if len(payload.pages) > settings.max_pages:
        raise HTTPException(
            status_code=413,
            detail=f"{len(payload.pages)} pages > FINLAM_MAX_PAGES={settings.max_pages}",
        )
    for p in payload.pages:
        zones = p.get("zones") if isinstance(p, dict) else None
        if isinstance(zones, list) and len(zones) > settings.max_zones_per_page:
            raise HTTPException(
                status_code=413,
                detail=f"page {p.get('page_id')} : {len(zones)} zones > "
                f"FINLAM_MAX_ZONES_PER_PAGE={settings.max_zones_per_page}",
            )
    try:
        pages = [page_from_json(p) for p in payload.pages]
    except ContractError as exc:
        raise HTTPException(status_code=422, detail=f"Contrat JSON invalide : {exc}") from exc
    return Issue(id=payload.issue_id, name=payload.name or payload.issue_id, pages=pages)


class _Metrics:
    """Compteurs d'exploitation, exposés en texte Prometheus sur ``/metrics``.

    Sans dépendance : le format d'exposition est du texte ligne par ligne. Si
    ``prometheus_client`` est installé, on l'utilise (histogrammes, registre).
    """

    def __init__(self) -> None:
        self.requests: dict[tuple[str, str, int], int] = {}
        self.seconds: dict[str, float] = {}
        self.pages: int = 0
        self.qc_errors: int = 0
        self.xsd_errors: int = 0

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        key = (method, route, status)
        self.requests[key] = self.requests.get(key, 0) + 1
        self.seconds[route] = self.seconds.get(route, 0.0) + seconds

    def render(self) -> str:
        lines = [
            "# HELP finlam_requests_total Requêtes HTTP par route et statut.",
            "# TYPE finlam_requests_total counter",
        ]
        for (method, route, status), n in sorted(self.requests.items()):
            lines.append(
                f'finlam_requests_total{{method="{method}",route="{route}",status="{status}"}} {n}'
            )
        lines += [
            "# HELP finlam_request_seconds_total Temps cumulé par route.",
            "# TYPE finlam_request_seconds_total counter",
        ]
        for route, total in sorted(self.seconds.items()):
            lines.append(f'finlam_request_seconds_total{{route="{route}"}} {total:.6f}')
        lines += [
            "# HELP finlam_pages_total Pages traitées.",
            "# TYPE finlam_pages_total counter",
            f"finlam_pages_total {self.pages}",
            "# HELP finlam_qc_errors_total Erreurs QC bloquantes rencontrées.",
            "# TYPE finlam_qc_errors_total counter",
            f"finlam_qc_errors_total {self.qc_errors}",
            "# HELP finlam_xsd_errors_total Erreurs de validation XSD rencontrées.",
            "# TYPE finlam_xsd_errors_total counter",
            f"finlam_xsd_errors_total {self.xsd_errors}",
        ]
        return "\n".join(lines) + "\n"


def create_app(models_dir: Path | None = None, settings: Settings | None = None) -> FastAPI:
    """Construit l'application.

    ``models_dir`` charge la doublure entraînée pour ``/issues/predict`` ; ``settings``
    porte la clé d'API, les limites de taille et le niveau de journalisation
    (par défaut :func:`~finlam_prod.config.get_settings`).
    """
    settings = settings or get_settings()
    if models_dir is None and settings.models_dir is not None:
        models_dir = settings.models_dir
    metrics = _Metrics()
    app = FastAPI(
        title="finlam-prod",
        version=__version__,
        description="Conversion ALTO/METS, contrôle qualité et doublure du pipeline FINLAM.",
    )

    def require_key(key: str | None = Depends(API_KEY_HEADER)) -> None:
        """Authentification par clé partagée, active seulement si FINLAM_API_KEY est défini."""
        if not settings.api_key:
            return
        if not key or not secrets.compare_digest(key, settings.api_key):
            raise HTTPException(status_code=401, detail="X-API-Key absente ou invalide")

    protected = [Depends(require_key)]

    @app.middleware("http")
    async def _observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.observe(request.method, path, response.status_code, elapsed)
        LOGGER.info(
            "%s %s %s %.3fs",
            request.method,
            path,
            response.status_code,
            elapsed,
            extra={"route": path, "status": response.status_code, "seconds": round(elapsed, 4)},
        )
        return response

    @app.exception_handler(FinlamError)
    async def _finlam_error(_request: Request, exc: FinlamError) -> JSONResponse:
        status = 422 if isinstance(exc, ContractError) else 500
        LOGGER.error("%s", exc, extra={"error_type": type(exc).__name__})
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics() -> str:
        """Exposition Prometheus (format texte)."""
        return metrics.render()

    pipeline = Pipeline()
    if models_dir is not None:
        from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier

        pipeline = Pipeline(
            GradientBoostingZoneClassifier.load(models_dir / "zone_classifier.joblib"),
            segmenter=BoundarySegmenter.load(models_dir / "segmenter.joblib"),
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "pipeline": pipeline.name,
            "models_dir": str(models_dir) if models_dir else None,
            "auth": bool(settings.api_key),
            "limits": {"max_pages": settings.max_pages, "max_zones": settings.max_zones_per_page},
        }

    @app.post("/issues/qc", dependencies=protected)
    def issue_qc(payload: IssuePayload) -> dict:
        issue = _issue_from_payload(payload, settings)
        findings = check_issue(issue)
        metrics.pages += len(issue.pages)
        return {
            "summary": summarise(findings, len(issue.pages)),
            "findings": findings_table(findings).to_dict(orient="records"),
        }

    @app.post("/issues/predict", dependencies=protected)
    def issue_predict(payload: IssuePayload) -> dict:
        issue = pipeline.run_issue(_issue_from_payload(payload, settings))
        metrics.pages += len(issue.pages)
        return {
            "issue_id": issue.id,
            "name": issue.name,
            "pages": [page_to_json(p, pipeline.name) for p in issue.pages],
        }

    @app.post("/issues", dependencies=protected)
    def issue_convert(payload: IssuePayload) -> StreamingResponse:
        issue = _issue_from_payload(payload, settings)
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
        metrics.pages += len(issue.pages)
        metrics.qc_errors += int(summary["errors"])
        metrics.xsd_errors += alto_errors + len(mets_errors)
        headers = {
            "Content-Disposition": f'attachment; filename="{issue.id}.zip"',
            "X-QC-Errors": str(summary["errors"]),
            "X-XSD-Errors": str(alto_errors + len(mets_errors)),
        }
        return StreamingResponse(buffer, media_type="application/zip", headers=headers)

    return app
