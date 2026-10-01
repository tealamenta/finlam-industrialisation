"""Descripteurs géométriques et textuels des zones.

Deux niveaux de description :

- :func:`zone_features` décrit une zone seule (position, taille, forme
  du texte) ; sert à la classification de zone.
- :func:`pair_features` décrit deux zones consécutives dans l'ordre de
  lecture ; sert à décider si la seconde ouvre un nouvel article.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence

import numpy as np

from finlam_prod.data.schema import TITLE_CLASSES, Page, Zone

ZONE_FEATURE_NAMES = [
    "x0",
    "y0",
    "x1",
    "y1",
    "width",
    "height",
    "area",
    "aspect",
    "cx",
    "cy",
    "n_chars",
    "n_words",
    "n_lines",
    "upper_ratio",
    "digit_ratio",
    "punct_ratio",
    "mean_word_len",
    "is_empty",
    "chars_per_area",
    "rel_width",
    "starts_upper",
    "ends_period",
    "column_estimate",
    "n_zones_page",
    "n_columns_spanned",
]

PAIR_FEATURE_NAMES = [
    "vertical_gap",
    "horizontal_overlap",
    "same_column",
    "b_above_a",
    "b_left_of_a",
    "dx_center",
    "dy_center",
    "a_is_title",
    "b_is_title",
    "a_class",
    "b_class",
    "b_width_rel",
    "a_ends_period",
    "b_starts_upper",
    "b_n_chars",
    "a_n_chars",
    "b_is_empty",
]

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def estimate_columns(
    page: Page,
    spanning_factor: float = 1.6,
    split_factor: float = 0.35,
    min_zones: int = 2,
) -> list[tuple[float, float]]:
    """Estime les colonnes de la page (voir :func:`columns_from_zones`)."""
    return columns_from_zones(page.zones, spanning_factor, split_factor, min_zones)


def columns_from_zones(
    all_zones: Sequence[Zone],
    spanning_factor: float = 1.6,
    split_factor: float = 0.35,
    min_zones: int = 2,
) -> list[tuple[float, float]]:
    """Estime les colonnes en regroupant les bords gauches des zones.

    Sur les pages FINLAM, les polygones de zones voisines se touchent
    bord à bord (pas de gouttière visible dans les coordonnées), ce qui
    rend inopérante toute détection par « vide » horizontal. En revanche
    les bords gauches ``x0`` se concentrent sur quelques valeurs, une par
    colonne : on trie les ``x0`` des zones « étroites » (largeur ≤
    ``spanning_factor`` × largeur médiane) et on coupe dès qu'un écart
    dépasse ``split_factor`` × largeur médiane. Chaque groupe d'au moins
    ``min_zones`` zones donne une colonne ``(x0 médian, x1 médian)``.
    """
    zones = [z for z in all_zones if z.width > 0]
    if len(zones) < min_zones:
        return [(0.0, 100.0)]
    med_w = statistics.median(z.width for z in zones)
    narrow = sorted((z for z in zones if z.width <= spanning_factor * med_w), key=lambda z: z.x0)
    if not narrow:
        return [(0.0, 100.0)]
    groups: list[list[Zone]] = [[narrow[0]]]
    for zone, previous in zip(narrow[1:], narrow, strict=False):
        if zone.x0 - previous.x0 > split_factor * med_w:
            groups.append([zone])
        else:
            groups[-1].append(zone)
    columns = [
        (statistics.median(z.x0 for z in g), statistics.median(z.x1 for z in g))
        for g in groups
        if len(g) >= min_zones
    ]
    return columns or [(0.0, 100.0)]


def column_index(zone: Zone, columns: Sequence[tuple[float, float]]) -> int:
    """Index de la colonne dont le bord gauche est le plus proche de celui de la zone."""
    if not columns:
        return -1
    return min(range(len(columns)), key=lambda i: abs(columns[i][0] - zone.x0))


def columns_spanned(
    zone: Zone, columns: Sequence[tuple[float, float]], min_cover: float = 0.3
) -> list[int]:
    """Colonnes dont la zone couvre au moins ``min_cover`` de la largeur."""
    covered = []
    for i, (x0, x1) in enumerate(columns):
        overlap = max(0.0, min(zone.x1, x1) - max(zone.x0, x0))
        if overlap >= min_cover * max(x1 - x0, 1e-6):
            covered.append(i)
    return covered


def _text_stats(text: str) -> dict[str, float]:
    letters = [c for c in text if c.isalpha()]
    words = text.split()
    return {
        "n_chars": float(len(text)),
        "n_words": float(len(words)),
        "n_lines": float(text.count("\n") + 1 if text else 0),
        "upper_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "digit_ratio": sum(c.isdigit() for c in text) / len(text) if text else 0.0,
        "punct_ratio": len(_PUNCT.findall(text)) / len(text) if text else 0.0,
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "is_empty": float(not text.strip()),
        "starts_upper": float(bool(text) and text.lstrip()[:1].isupper()),
        "ends_period": float(text.rstrip().endswith((".", "!", "?"))),
    }


def zone_features(
    zone: Zone, page: Page, columns: Sequence[tuple[float, float]] | None = None
) -> list[float]:
    """Vecteur de descripteurs d'une zone (ordre : :data:`ZONE_FEATURE_NAMES`)."""
    columns = columns if columns is not None else estimate_columns(page)
    stats = _text_stats(zone.text)
    area = max(zone.area, 1e-6)
    return [
        zone.x0,
        zone.y0,
        zone.x1,
        zone.y1,
        zone.width,
        zone.height,
        zone.area,
        zone.width / max(zone.height, 1e-6),
        zone.cx,
        zone.cy,
        stats["n_chars"],
        stats["n_words"],
        stats["n_lines"],
        stats["upper_ratio"],
        stats["digit_ratio"],
        stats["punct_ratio"],
        stats["mean_word_len"],
        stats["is_empty"],
        stats["n_chars"] / area,
        zone.width / 100.0,
        stats["starts_upper"],
        stats["ends_period"],
        float(column_index(zone, columns)),
        float(len(page.zones)),
        float(len(columns_spanned(zone, columns))),
    ]


def page_feature_matrix(page: Page) -> np.ndarray:
    """Matrice (n_zones × n_features) d'une page, dans l'ordre de ``page.zones``."""
    columns = estimate_columns(page)
    return np.array([zone_features(z, page, columns) for z in page.zones], dtype=float)


def pair_features(
    a: Zone,
    b: Zone,
    page: Page,
    columns: Sequence[tuple[float, float]] | None = None,
    class_key: str = "zone_class",
) -> list[float]:
    """Descripteurs d'un couple (a, b) de zones consécutives dans l'ordre de lecture.

    ``class_key`` indique quel attribut porte la classe utilisée (vérité
    terrain ou prédiction) : la séparation d'articles s'appuie sur les
    classes prédites en production.
    """
    columns = columns if columns is not None else estimate_columns(page)
    a_class = getattr(a, class_key)
    b_class = getattr(b, class_key)
    a_class = -1 if a_class is None else int(a_class)
    b_class = -1 if b_class is None else int(b_class)
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) / max(min(a.width, b.width), 1e-6)
    a_stats, b_stats = _text_stats(a.text), _text_stats(b.text)
    return [
        b.y0 - a.y1,
        overlap,
        float(column_index(a, columns) == column_index(b, columns)),
        float(b.y0 < a.y0),
        float(b.x1 <= a.x0 + 1),
        b.cx - a.cx,
        b.cy - a.cy,
        float(a_class in TITLE_CLASSES),
        float(b_class in TITLE_CLASSES),
        float(a_class),
        float(b_class),
        b.width / 100.0,
        a_stats["ends_period"],
        b_stats["starts_upper"],
        b_stats["n_chars"],
        a_stats["n_chars"],
        b_stats["is_empty"],
    ]
