"""Prétraitement d'un fascicule entier : la tâche ``preprocess`` de la chaîne.

Entre ``fetch`` (images + ALTO récupérés) et ``predict`` (modèles), chaque page
qui a une image passe par :func:`finlam_prod.preprocess.preprocess_page`
(pli, éclairage, inclinaison) et, si demandé, par le dégauchissement appris
de :mod:`finlam_prod.preprocess.dewarp` (photos). L'image corrigée est
écrite à côté de l'original, ``page.image_path`` pointe dessus, et un rapport
par page (``preprocess.json``) alimente le contrôle qualité : une page pliée
ou fortement inclinée est une page à surveiller.

Les pages sans image (jeux d'évaluation sans images, ALTO seul) passent
inchangées : le prétraitement ne doit jamais bloquer la chaîne.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finlam_prod.data.schema import Issue, Page
from finlam_prod.errors import PreprocessError

LOGGER = logging.getLogger(__name__)
REPORT_FILE = "preprocess.json"
SKEW_WARNING_DEG = 1.0  # au-delà, la page est signalée au QC


@dataclass
class PageReport:
    """Ce qui a été fait à une page."""

    page_id: str
    index: int
    source: str | None
    output: str | None
    fold_detected: bool = False
    fold_x: int | None = None
    fold_depth: float | None = None
    angles: list[float] = field(default_factory=list)
    dewarped: bool = False
    seconds: float = 0.0
    skipped: str | None = None  # raison si la page n'a pas été traitée

    @property
    def flagged(self) -> bool:
        """Page à signaler au QC : pli, ou inclinaison corrigée au-delà du seuil."""
        return self.fold_detected or any(abs(a) >= SKEW_WARNING_DEG for a in self.angles)


@dataclass
class IssueReport:
    """Rapport du fascicule, sérialisable."""

    issue_id: str
    pages: list[PageReport]

    @property
    def summary(self) -> dict[str, Any]:
        """Compteurs pour le résumé QC."""
        done = [p for p in self.pages if p.skipped is None]
        return {
            "preprocessed_pages": len(done),
            "skipped_pages": len(self.pages) - len(done),
            "folded_pages": sum(p.fold_detected for p in done),
            "skewed_pages": sum(any(abs(a) >= SKEW_WARNING_DEG for a in p.angles) for p in done),
            "dewarped_pages": sum(p.dewarped for p in done),
            "preprocess_seconds": round(sum(p.seconds for p in done), 2),
        }

    def to_dict(self) -> dict[str, Any]:
        """Forme JSON."""
        return {
            "issue_id": self.issue_id,
            "summary": self.summary,
            "pages": [dataclasses.asdict(p) for p in self.pages],
        }

    def write(self, target: Path) -> Path:
        """Écrit ``preprocess.json``."""
        target.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return target


def _output_path(image_path: Path, out_dir: Path | None) -> Path:
    directory = out_dir or image_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{image_path.stem}_pre.png"


def preprocess_issue(
    issue: Issue,
    out_dir: Path | None = None,
    dewarp_model: Any = None,
    max_angle: float = 3.0,
) -> tuple[Issue, IssueReport]:
    """Prétraite chaque page qui a une image ; renvoie le fascicule corrigé et le rapport.

    ``dewarp_model`` (UVDoc chargé, voir :mod:`finlam_prod.preprocess.dewarp`)
    n'est appliqué que s'il est fourni : il ne sert qu'aux photos, pas aux
    scans à plat.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise PreprocessError(
            "OpenCV absent : installer l'extra image (pip install 'finlam-prod[image]')"
        ) from exc
    from finlam_prod.preprocess.fold import preprocess_page

    pages: list[Page] = []
    reports: list[PageReport] = []
    for page in issue.pages:
        report = PageReport(page.id, page.index, page.image_path, None)
        source = Path(page.image_path) if page.image_path else None
        if source is None or not source.is_file():
            report.skipped = "pas d'image"
            pages.append(page)
            reports.append(report)
            continue
        started = time.perf_counter()
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            report.skipped = "image illisible"
            LOGGER.warning("page %s : image illisible (%s)", page.id, source)
            pages.append(page)
            reports.append(report)
            continue
        if dewarp_model is not None:
            from finlam_prod.preprocess.dewarp import dewarp_page

            image = dewarp_page(image, dewarp_model)
            report.dewarped = True
        result = preprocess_page(image, max_angle=max_angle)
        target = _output_path(source, out_dir)
        if not cv2.imwrite(str(target), result.image):
            raise PreprocessError(f"écriture impossible : {target}")
        report.output = str(target)
        report.fold_detected = result.fold.detected
        report.fold_x = result.fold.x if result.fold.detected else None
        report.fold_depth = result.fold.depth if result.fold.detected else None
        report.angles = [float(a) for a in result.angles]
        report.seconds = round(time.perf_counter() - started, 3)
        pages.append(dataclasses.replace(page, image_path=str(target)))
        reports.append(report)
        LOGGER.info(
            "page %s prétraitée : pli=%s angles=%s %.2fs",
            page.id,
            result.fold.detected,
            result.angles,
            report.seconds,
            extra={"page_id": page.id, "fold": result.fold.to_dict(), "angles": report.angles},
        )
    issue_report = IssueReport(issue.id, reports)
    LOGGER.info("fascicule %s : %s", issue.id, issue_report.summary, extra=issue_report.summary)
    return dataclasses.replace(issue, pages=pages), issue_report
