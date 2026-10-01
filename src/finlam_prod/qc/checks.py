"""Contrôle qualité des sorties du pipeline.

Sans vérité terrain (cas de la production), on ne peut pas mesurer la
justesse ; on peut en revanche détecter les sorties *incohérentes*, qui
sont presque toujours fausses. Chaque vérification produit des
:class:`Finding` de sévérité ``error`` (à bloquer) ou ``warning`` (à
revoir), agrégés par page et par fascicule pour alimenter un tableau de
bord ou une reprise manuelle.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pandas as pd

from finlam_prod.data.schema import (
    HEADER_CLASSES,
    NO_ARTICLE,
    NON_TEXT_CLASSES,
    TITLE_CLASSES,
    Issue,
    Page,
    ZoneClass,
)


@dataclass(frozen=True)
class Finding:
    """Une anomalie détectée sur une page."""

    severity: str  # "error" | "warning"
    code: str
    page_id: str
    zone_id: str | None
    message: str


@dataclass
class QCConfig:
    """Seuils des vérifications."""

    max_overlap_iou: float = 0.3
    min_zone_area: float = 0.01  # % de page
    min_confidence: float = 0.6
    max_text_zones_without_title: int = 12
    max_low_confidence_ratio: float = 0.2
    max_table_area_ratio: float = 0.25
    max_oov_ratio: float = 0.35
    min_words_for_oov: int = 30


def check_geometry(page: Page, config: QCConfig) -> list[Finding]:
    """Zones hors page, dégénérées ou minuscules."""
    findings = []
    for zone in page.zones:
        x0, y0, x1, y1 = zone.bbox
        if len(zone.polygon) < 3:
            findings.append(
                Finding(
                    "error",
                    "polygon_degenerate",
                    page.id,
                    zone.id,
                    "polygone à moins de 3 sommets",
                )
            )
        if x0 < -0.5 or y0 < -0.5 or x1 > 100.5 or y1 > 100.5:
            findings.append(
                Finding(
                    "error",
                    "out_of_page",
                    page.id,
                    zone.id,
                    f"bbox hors page {zone.bbox}",
                )
            )
        if zone.width <= 0 or zone.height <= 0:
            findings.append(
                Finding("error", "empty_bbox", page.id, zone.id, "largeur ou hauteur nulle")
            )
        elif zone.area < config.min_zone_area:
            findings.append(
                Finding(
                    "warning",
                    "tiny_zone",
                    page.id,
                    zone.id,
                    f"aire {zone.area:.3f} % de page",
                )
            )
    return findings


def check_overlaps(page: Page, config: QCConfig) -> list[Finding]:
    """Chevauchements importants entre zones (segmentation douteuse)."""
    findings = []
    zones = page.zones
    for i, a in enumerate(zones):
        for b in zones[i + 1 :]:
            iou = a.iou(b)
            if iou > config.max_overlap_iou:
                findings.append(
                    Finding(
                        "warning",
                        "overlap",
                        page.id,
                        a.id,
                        f"IoU {iou:.2f} avec {b.id}",
                    )
                )
    return findings


def check_text(page: Page, config: QCConfig) -> list[Finding]:
    """Cohérence entre classe et contenu textuel."""
    findings = []
    for zone in page.zones:
        if zone.zone_class is None:
            findings.append(Finding("error", "missing_class", page.id, zone.id, "classe absente"))
            continue
        empty = not zone.text.strip()
        if empty and zone.zone_class not in NON_TEXT_CLASSES:
            findings.append(
                Finding(
                    "warning",
                    "empty_text",
                    page.id,
                    zone.id,
                    f"texte vide pour une zone {ZoneClass(zone.zone_class).label}",
                )
            )
        if not empty and zone.zone_class in NON_TEXT_CLASSES:
            findings.append(
                Finding(
                    "warning",
                    "text_in_illustration",
                    page.id,
                    zone.id,
                    "texte dans une illustration",
                )
            )
        if zone.zone_class == ZoneClass.ARTICLE_TITLE and len(zone.text) > 400:
            findings.append(
                Finding(
                    "warning",
                    "long_title",
                    page.id,
                    zone.id,
                    f"titre de {len(zone.text)} caractères",
                )
            )
    return findings


def check_reading_order(page: Page, config: QCConfig) -> list[Finding]:
    """Ordre de lecture complet et sans doublon."""
    orders = [z.order for z in page.zones]
    if any(o is None for o in orders):
        return [
            Finding(
                "error",
                "missing_order",
                page.id,
                None,
                "ordre de lecture absent sur certaines zones",
            )
        ]
    if sorted(orders) != list(range(len(orders))):
        return [
            Finding(
                "error",
                "order_not_permutation",
                page.id,
                None,
                "les rangs ne forment pas une permutation 0..n-1",
            )
        ]
    return []


def check_articles(page: Page, config: QCConfig) -> list[Finding]:
    """Structure des articles : titre, en-têtes hors article, taille plausible."""
    findings = []
    for zone in page.zones:
        if zone.article_id is None:
            findings.append(
                Finding(
                    "error",
                    "missing_article",
                    page.id,
                    zone.id,
                    "identifiant d'article absent",
                )
            )
        elif zone.zone_class in HEADER_CLASSES and zone.article_id != NO_ARTICLE:
            findings.append(
                Finding(
                    "warning",
                    "header_in_article",
                    page.id,
                    zone.id,
                    "zone d'en-tête rattachée à un article",
                )
            )
    for article_id, zones in page.articles().items():
        has_title = any(z.zone_class in TITLE_CLASSES for z in zones)
        n_text = sum(z.zone_class == ZoneClass.ARTICLE_TEXT for z in zones)
        if not has_title and n_text > config.max_text_zones_without_title:
            findings.append(
                Finding(
                    "warning",
                    "article_without_title",
                    page.id,
                    zones[0].id,
                    f"article {article_id} : {n_text} zones de texte sans titre",
                )
            )
        ranks = sorted(z.order for z in zones if z.order is not None)
        if ranks and ranks[-1] - ranks[0] + 1 > 3 * len(ranks):
            findings.append(
                Finding(
                    "warning",
                    "article_fragmented",
                    page.id,
                    zones[0].id,
                    f"article {article_id} très dispersé dans l'ordre de lecture",
                )
            )
    return findings


def check_confidence(page: Page, config: QCConfig) -> list[Finding]:
    """Part de zones à faible confiance OCR."""
    scored = [z for z in page.zones if z.confidence is not None]
    if not scored:
        return []
    low = [z for z in scored if z.confidence < config.min_confidence]
    findings = [
        Finding("warning", "low_confidence", page.id, z.id, f"confiance {z.confidence:.2f}")
        for z in low
    ]
    if len(low) / len(scored) > config.max_low_confidence_ratio:
        findings.append(
            Finding(
                "error",
                "page_low_confidence",
                page.id,
                None,
                f"{len(low)}/{len(scored)} zones sous {config.min_confidence}",
            )
        )
    return findings


def check_tables(page: Page, config: QCConfig) -> list[Finding]:
    """Pages dominées par des tableaux : point de rupture connu de l'OLR industriel.

    Neudecker & Antonacopoulos (DAS 2016), sur 12 M de pages d'Europeana
    Newspapers, rapportent que la segmentation d'articles par prestataire
    échoue presque systématiquement sur les grands tableaux (cours de
    bourse, résultats sportifs). On signale les pages où la surface des
    zones ARTICLE-TABLE dépasse ``max_table_area_ratio`` pour relecture.
    """
    total = sum(z.area for z in page.zones) or 1.0
    tables = sum(z.area for z in page.zones if z.zone_class == ZoneClass.ARTICLE_TABLE)
    ratio = tables / total
    if ratio > config.max_table_area_ratio:
        return [
            Finding(
                "warning",
                "table_heavy_page",
                page.id,
                None,
                f"{ratio:.0%} de la surface en tableaux : segmentation d'articles à relire",
            )
        ]
    return []


def _lexicon_ratio(text: str) -> tuple[int, int]:
    """(mots, mots « suspects ») : un mot est suspect s'il mêle lettres et chiffres ou symboles."""
    words = re.findall(r"\S+", text)
    suspect = 0
    for w in words:
        core = w.strip(".,;:!?()[]«»\"'-")
        if not core:
            continue
        has_alpha = any(c.isalpha() for c in core)
        has_odd = any(c.isdigit() or not (c.isalnum() or c in "'-") for c in core)
        if (has_alpha and has_odd) or (len(core) == 1 and not core.isalpha()):
            suspect += 1
    return len(words), suspect


def check_text_quality(page: Page, config: QCConfig) -> list[Finding]:
    """Qualité OCR estimée sans vérité terrain (mots suspects), par page.

    Ben Salah, Ragot, Paquet & Moreux (ICDAR 2015) estiment la qualité
    OCR en production sans vérité terrain (alignement croisé de moteurs).
    À défaut de second moteur, on utilise un signal plus simple : la part
    de mots mêlant lettres et chiffres ou symboles, typique des erreurs
    OCR sur presse ancienne. Au-dessus de ``max_oov_ratio`` (sur au moins
    ``min_words_for_oov`` mots), la page est signalée : la séparation
    d'articles fondée sur le texte y sera moins fiable.
    """
    text = "\n".join(z.text for z in page.zones if z.zone_class not in NON_TEXT_CLASSES)
    n_words, suspect = _lexicon_ratio(text)
    if n_words < config.min_words_for_oov:
        return []
    ratio = suspect / n_words
    if ratio > config.max_oov_ratio:
        return [
            Finding(
                "warning",
                "low_text_quality",
                page.id,
                None,
                f"{ratio:.0%} de mots suspects sur {n_words} : OCR dégradé probable",
            )
        ]
    return []


CHECKS = [
    check_geometry,
    check_overlaps,
    check_text,
    check_reading_order,
    check_articles,
    check_confidence,
    check_tables,
    check_text_quality,
]


def check_page(page: Page, config: QCConfig | None = None) -> list[Finding]:
    """Toutes les vérifications d'une page."""
    config = config or QCConfig()
    findings: list[Finding] = []
    for check in CHECKS:
        findings.extend(check(page, config))
    return findings


def check_issue(issue: Issue, config: QCConfig | None = None) -> list[Finding]:
    """Toutes les vérifications d'un fascicule, plus la continuité des articles entre pages."""
    findings = [f for page in issue.pages for f in check_page(page, config)]
    seen_pages: dict[int, list[int]] = {}
    for page in issue.pages:
        for article_id in page.articles():
            seen_pages.setdefault(article_id, []).append(page.index)
    for article_id, pages in seen_pages.items():
        if len(pages) > 1 and pages != list(range(pages[0], pages[0] + len(pages))):
            findings.append(
                Finding(
                    "warning",
                    "article_non_contiguous_pages",
                    issue.pages[0].id,
                    None,
                    f"article {article_id} sur des pages non contiguës {pages}",
                )
            )
    return findings


def findings_table(findings: Iterable[Finding]) -> pd.DataFrame:
    """Table des anomalies (une ligne par anomalie)."""
    rows = [asdict(f) for f in findings]
    return pd.DataFrame(rows, columns=["severity", "code", "page_id", "zone_id", "message"])


def summarise(findings: Iterable[Finding], n_pages: int) -> dict[str, float | int]:
    """Résumé : nombre d'anomalies par sévérité et par code, pages bloquantes."""
    table = findings_table(findings)
    errors = table[table.severity == "error"]
    summary: dict[str, float | int] = {
        "pages": n_pages,
        "errors": int(len(errors)),
        "warnings": int((table.severity == "warning").sum()),
        "pages_with_errors": int(errors.page_id.nunique()) if len(errors) else 0,
    }
    for code, count in table.code.value_counts().items():
        summary[f"n_{code}"] = int(count)
    return summary
