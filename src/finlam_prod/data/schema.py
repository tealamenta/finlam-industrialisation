"""Modèle de données commun à toute la chaîne.

Les coordonnées des zones sont exprimées en pourcentage de la page
(0-100 sur chaque axe), comme dans le jeu FINLAM. Cela rend les
géométries indépendantes de la résolution de numérisation ; la
conversion en pixels n'a lieu qu'à l'export ALTO.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path

NO_ARTICLE = -1
"""Identifiant d'article des zones hors article (en-tête de page, etc.)."""


class ZoneClass(IntEnum):
    """Les 13 classes de zones du jeu FINLAM, dans l'ordre des identifiants."""

    HEADER_TITLE = 0
    HEADER_TEXT = 1
    ARTICLE_ILLUSTRATION = 2
    ADVERTISEMENT = 3
    ANNOUNCEMENT = 4
    ARTICLE_TITLE = 5
    ARTICLE_TEXT = 6
    ARTICLE_SUBTITLE = 7
    ARTICLE_INSIDEHEADING = 8
    CAPTION = 9
    AUTHOR = 10
    ARTICLE_TABLE = 11
    SECTION_TITLE = 12

    @property
    def label(self) -> str:
        """Libellé FINLAM (« ARTICLE-TITLE », ...)."""
        return self.name.replace("_", "-")

    @classmethod
    def from_label(cls, label: str) -> ZoneClass:
        """Construit la classe à partir d'un libellé FINLAM."""
        return cls[label.replace("-", "_").upper()]


TITLE_CLASSES = frozenset({ZoneClass.ARTICLE_TITLE, ZoneClass.SECTION_TITLE})
"""Classes qui, par convention, ouvrent un nouvel article."""

HEADER_CLASSES = frozenset({ZoneClass.HEADER_TITLE, ZoneClass.HEADER_TEXT})
"""Classes hors article (identifiant d'article NO_ARTICLE)."""

NON_TEXT_CLASSES = frozenset({ZoneClass.ARTICLE_ILLUSTRATION})
"""Classes dont le contenu textuel est normalement vide."""


@dataclass
class Zone:
    """Une zone de page (bloc de mise en page) et ses annotations.

    Attributes:
        id: identifiant stable de la zone.
        polygon: sommets (x, y) en pourcentage de la page.
        text: transcription OCR de la zone (vide pour une illustration).
        zone_class: classe de la zone (vérité terrain ou prédiction).
        order: rang dans l'ordre de lecture de la page (0 = première zone).
        article_id: identifiant de l'article auquel la zone appartient.
        confidence: confiance du modèle, si disponible.

    """

    id: str
    polygon: list[tuple[float, float]]
    text: str = ""
    zone_class: int | None = None
    order: int | None = None
    article_id: int | None = None
    confidence: float | None = None
    section_id: int | None = None
    raw_class: str | None = None

    @property
    def order_key(self) -> int:
        """Rang de lecture utilisable pour trier (zones sans rang en dernier)."""
        return self.order if self.order is not None else sys.maxsize

    @property
    def article(self) -> int:
        """Identifiant d'article, ``NO_ARTICLE`` si la zone n'en a pas."""
        return self.article_id if self.article_id is not None else NO_ARTICLE

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Boîte englobante (x0, y0, x1, y1) en pourcentage."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def x0(self) -> float:  # noqa: D102
        return self.bbox[0]

    @property
    def y0(self) -> float:  # noqa: D102
        return self.bbox[1]

    @property
    def x1(self) -> float:  # noqa: D102
        return self.bbox[2]

    @property
    def y1(self) -> float:  # noqa: D102
        return self.bbox[3]

    @property
    def width(self) -> float:  # noqa: D102
        return self.x1 - self.x0

    @property
    def height(self) -> float:  # noqa: D102
        return self.y1 - self.y0

    @property
    def cx(self) -> float:  # noqa: D102
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:  # noqa: D102
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:  # noqa: D102
        return self.width * self.height

    def iou(self, other: Zone) -> float:
        """Intersection sur union des boîtes englobantes."""
        ax0, ay0, ax1, ay1 = self.bbox
        bx0, by0, bx1, by1 = other.bbox
        inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = inter_w * inter_h
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Page:
    """Une page de fascicule et ses zones."""

    id: str
    issue_id: str
    index: int
    zones: list[Zone] = field(default_factory=list)
    width_px: int | None = None
    height_px: int | None = None
    image_path: str | None = None
    source: str = "unknown"

    def zones_in_order(self, key: str = "order") -> list[Zone]:
        """Zones triées selon l'ordre de lecture (les zones sans rang à la fin)."""
        ranked = [z for z in self.zones if getattr(z, key) is not None]
        unranked = [z for z in self.zones if getattr(z, key) is None]
        return sorted(ranked, key=lambda z: getattr(z, key)) + unranked

    def articles(self) -> dict[int, list[Zone]]:
        """Zones regroupées par identifiant d'article (hors NO_ARTICLE)."""
        groups: dict[int, list[Zone]] = {}
        for zone in self.zones_in_order():
            if zone.article_id is None or zone.article_id == NO_ARTICLE:
                continue
            groups.setdefault(zone.article_id, []).append(zone)
        return groups


@dataclass
class Issue:
    """Un fascicule (numéro de journal) : une suite de pages."""

    id: str
    name: str
    pages: list[Page] = field(default_factory=list)

    def __iter__(self) -> Iterator[Page]:  # noqa: D105
        return iter(self.pages)


def _page_from_dict(payload: dict) -> Page:
    zones = [
        Zone(
            id=z["id"],
            polygon=[tuple(p) for p in z["polygon"]],
            text=z.get("text", ""),
            zone_class=z.get("zone_class"),
            order=z.get("order"),
            article_id=z.get("article_id"),
            confidence=z.get("confidence"),
            section_id=z.get("section_id"),
            raw_class=z.get("raw_class"),
        )
        for z in payload["zones"]
    ]
    return Page(
        id=payload["id"],
        issue_id=payload["issue_id"],
        index=payload["index"],
        zones=zones,
        width_px=payload.get("width_px"),
        height_px=payload.get("height_px"),
        image_path=payload.get("image_path"),
        source=payload.get("source", "unknown"),
    )


def save_pages(pages: list[Page], path: str | Path) -> None:
    """Sérialise des pages en JSON Lines (une page par ligne)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")


def load_pages(path: str | Path) -> list[Page]:
    """Relit des pages écrites par :func:`save_pages`."""
    with Path(path).open(encoding="utf-8") as handle:
        return [_page_from_dict(json.loads(line)) for line in handle if line.strip()]


def group_by_issue(pages: list[Page]) -> list[Issue]:
    """Regroupe des pages en fascicules, dans l'ordre des pages."""
    issues: dict[str, Issue] = {}
    for page in pages:
        issue = issues.setdefault(page.issue_id, Issue(id=page.issue_id, name=page.issue_id))
        issue.pages.append(page)
    for issue in issues.values():
        issue.pages.sort(key=lambda p: p.index)
    return list(issues.values())
