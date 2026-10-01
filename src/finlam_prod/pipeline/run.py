"""Chaînage des trois tâches et export JSON du pipeline.

Le :class:`Pipeline` reproduit la sortie d'un pipeline d'extraction de
structure logique : chaque page entrante (zones + texte OCR) ressort
avec classe, ordre de lecture et article prédits. L'export JSON est le
format d'échange « prototype » que l'on convertit ensuite en ALTO/METS.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from finlam_prod import __version__
from finlam_prod.data.schema import Issue, Page, Zone, ZoneClass
from finlam_prod.errors import ContractError
from finlam_prod.pipeline.base import ArticleSegmenter, ReadingOrderer, ZoneClassifier
from finlam_prod.pipeline.rules import (
    HeuristicClassifier,
    TitleRuleSegmenter,
    XYCutOrderer,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """Enchaîne classification, ordre de lecture et séparation d'articles."""

    classifier: ZoneClassifier = field(default_factory=HeuristicClassifier)
    orderer: ReadingOrderer = field(default_factory=XYCutOrderer)
    segmenter: ArticleSegmenter = field(default_factory=TitleRuleSegmenter)

    @property
    def name(self) -> str:
        """Nom composite, utile dans les rapports."""
        return f"{self.classifier.name}+{self.orderer.name}+{self.segmenter.name}"

    def fit(self, pages: list[Page]) -> Pipeline:
        """Entraîne les trois composants sur des pages annotées."""
        started = time.perf_counter()
        self.classifier.fit(pages)
        self.orderer.fit(pages)
        self.segmenter.fit(pages)
        LOGGER.info(
            "pipeline %s entraîné : %d pages, %d zones, %.1fs",
            self.name,
            len(pages),
            sum(len(p.zones) for p in pages),
            time.perf_counter() - started,
            extra={"pipeline": self.name, "n_pages": len(pages)},
        )
        return self

    def run(self, page: Page) -> Page:
        """Renvoie une copie de la page portant les prédictions."""
        classes = self.classifier.predict(page)
        classified = replace(
            page,
            zones=[
                replace(z, zone_class=int(k)) for z, k in zip(page.zones, classes, strict=True)
            ],
        )
        order = self.orderer.predict(classified)
        articles = self.segmenter.predict(classified, order, classes)
        zones = [
            replace(zone, zone_class=int(k), order=int(o), article_id=int(a))
            for zone, k, o, a in zip(page.zones, classes, order, articles, strict=True)
        ]
        return replace(page, zones=zones)

    def run_issue(self, issue: Issue) -> Issue:
        """Traite toutes les pages d'un fascicule, avec des articles numérotés par fascicule."""
        started = time.perf_counter()
        pages = []
        offset = 0
        for page in issue.pages:
            predicted = self.run(page)
            LOGGER.debug(
                "page %s : %d zones, %d articles",
                page.id,
                len(predicted.zones),
                len({z.article for z in predicted.zones if z.article > 0}),
            )
            local_ids = sorted(
                {z.article_id for z in predicted.zones if z.article_id and z.article_id > 0}
            )
            mapping = {local: offset + i + 1 for i, local in enumerate(local_ids)}
            for zone in predicted.zones:
                if zone.article_id in mapping:
                    zone.article_id = mapping[zone.article_id]
            offset += len(local_ids)
            pages.append(predicted)
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "fascicule %s : %d pages, %d articles, %.2fs (%.3fs/page) par %s",
            issue.id,
            len(pages),
            offset,
            elapsed,
            elapsed / max(len(pages), 1),
            self.name,
            extra={
                "issue_id": issue.id,
                "n_pages": len(pages),
                "n_articles": offset,
                "seconds": round(elapsed, 3),
                "pipeline": self.name,
            },
        )
        return replace(issue, pages=pages)


# ---------------------------------------------------------------- export JSON
def page_to_json(page: Page, pipeline_name: str = "finlam-prod") -> dict:
    """Représentation JSON « sortie de pipeline » d'une page."""
    return {
        "page_id": page.id,
        "issue_id": page.issue_id,
        "page_index": page.index,
        "image": {
            "path": page.image_path,
            "width": page.width_px,
            "height": page.height_px,
        },
        "pipeline": {
            "name": pipeline_name,
            "version": __version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "zones": [
            {
                "id": z.id,
                "polygon": [list(p) for p in z.polygon],
                "text": z.text,
                "class": ZoneClass(z.zone_class).label if z.zone_class is not None else None,
                "reading_order": z.order,
                "article_id": z.article_id,
                "confidence": z.confidence,
            }
            for z in page.zones_in_order()
        ],
    }


def _require(payload: dict, key: str, kind: type | tuple[type, ...], page_id: str | None) -> Any:
    """Lit une clé obligatoire du contrat en traduisant les erreurs en :class:`ContractError`."""
    if key not in payload:
        raise ContractError("clé manquante", page_id, key)
    value = payload[key]
    if not isinstance(value, kind):
        raise ContractError(f"type {type(value).__name__} inattendu", page_id, key)
    return value


def _zone_from_json(z: dict, page_id: str) -> Zone:
    zone_id = str(_require(z, "id", (str, int), page_id))
    polygon = _require(z, "polygon", list, page_id)
    try:
        points = [(float(x), float(y)) for x, y in polygon]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"polygone illisible ({exc})", page_id, f"zone {zone_id}") from exc
    if len(points) < 3:
        raise ContractError("polygone à moins de trois points", page_id, f"zone {zone_id}")
    label = z.get("class")
    try:
        zone_class = ZoneClass.from_label(label) if label else None
    except (KeyError, ValueError) as exc:
        raise ContractError(f"classe inconnue {label!r}", page_id, f"zone {zone_id}") from exc
    return Zone(
        id=zone_id,
        polygon=points,
        text=z.get("text") or "",
        zone_class=zone_class,
        order=z.get("reading_order"),
        article_id=z.get("article_id"),
        confidence=z.get("confidence"),
    )


def page_from_json(payload: dict) -> Page:
    """Relit une page exportée par :func:`page_to_json`.

    Toute entorse au contrat (clé absente, type inattendu, polygone ou classe
    invalides) lève :class:`~finlam_prod.errors.ContractError` avec la page et
    le champ fautifs, plutôt qu'une ``KeyError`` anonyme.
    """
    if not isinstance(payload, dict):
        raise ContractError(f"page attendue sous forme d'objet, reçu {type(payload).__name__}")
    page_id = str(payload.get("page_id", "?"))
    zones_raw = _require(payload, "zones", list, page_id)
    zones = [_zone_from_json(z, page_id) for z in zones_raw]
    image = payload.get("image") or {}
    return Page(
        id=str(_require(payload, "page_id", (str, int), page_id)),
        issue_id=str(_require(payload, "issue_id", (str, int), page_id)),
        index=int(_require(payload, "page_index", int, page_id)),
        zones=zones,
        width_px=image.get("width"),
        height_px=image.get("height"),
        image_path=image.get("path"),
        source=(payload.get("pipeline") or {}).get("name", "json"),
    )


def write_issue_json(issue: Issue, path: str | Path, pipeline_name: str = "finlam-prod") -> Path:
    """Écrit un fascicule prédit en un fichier JSON (une entrée par page)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue_id": issue.id,
        "name": issue.name,
        "pages": [page_to_json(p, pipeline_name) for p in issue.pages],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_issue_json(path: str | Path) -> Issue:
    """Relit un fascicule écrit par :func:`write_issue_json`.

    Lève :class:`~finlam_prod.errors.ContractError` si le fichier n'est pas un
    JSON valide ou ne respecte pas le contrat.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"fichier illisible {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"JSON invalide dans {path} : {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"fascicule attendu sous forme d'objet dans {path}")
    issue_id = str(_require(payload, "issue_id", (str, int), None))
    pages = _require(payload, "pages", list, None)
    return Issue(
        id=issue_id,
        name=payload.get("name") or issue_id,
        pages=[page_from_json(p) for p in pages],
    )
