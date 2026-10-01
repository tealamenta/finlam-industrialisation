"""Chaînage des trois tâches et export JSON du pipeline.

Le :class:`Pipeline` reproduit la sortie d'un pipeline d'extraction de
structure logique : chaque page entrante (zones + texte OCR) ressort
avec classe, ordre de lecture et article prédits. L'export JSON est le
format d'échange « prototype » que l'on convertit ensuite en ALTO/METS.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from finlam_prod import __version__
from finlam_prod.data.schema import Issue, Page, Zone, ZoneClass
from finlam_prod.pipeline.base import ArticleSegmenter, ReadingOrderer, ZoneClassifier
from finlam_prod.pipeline.rules import (
    HeuristicClassifier,
    TitleRuleSegmenter,
    XYCutOrderer,
)


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
        self.classifier.fit(pages)
        self.orderer.fit(pages)
        self.segmenter.fit(pages)
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
        pages = []
        offset = 0
        for page in issue.pages:
            predicted = self.run(page)
            local_ids = sorted(
                {z.article_id for z in predicted.zones if z.article_id and z.article_id > 0}
            )
            mapping = {local: offset + i + 1 for i, local in enumerate(local_ids)}
            for zone in predicted.zones:
                if zone.article_id in mapping:
                    zone.article_id = mapping[zone.article_id]
            offset += len(local_ids)
            pages.append(predicted)
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


def page_from_json(payload: dict) -> Page:
    """Relit une page exportée par :func:`page_to_json`."""
    zones = [
        Zone(
            id=z["id"],
            polygon=[tuple(p) for p in z["polygon"]],
            text=z.get("text", ""),
            zone_class=ZoneClass.from_label(z["class"]) if z.get("class") else None,
            order=z.get("reading_order"),
            article_id=z.get("article_id"),
            confidence=z.get("confidence"),
        )
        for z in payload["zones"]
    ]
    image = payload.get("image", {})
    return Page(
        id=payload["page_id"],
        issue_id=payload["issue_id"],
        index=payload["page_index"],
        zones=zones,
        width_px=image.get("width"),
        height_px=image.get("height"),
        image_path=image.get("path"),
        source=payload.get("pipeline", {}).get("name", "json"),
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
    """Relit un fascicule écrit par :func:`write_issue_json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Issue(
        id=payload["issue_id"],
        name=payload.get("name", payload["issue_id"]),
        pages=[page_from_json(p) for p in payload["pages"]],
    )
