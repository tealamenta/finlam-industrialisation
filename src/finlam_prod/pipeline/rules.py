"""Méthodes à base de règles (sans apprentissage).

Elles servent de référence basse dans le benchmark et de solution de
repli en production quand aucun modèle n'est disponible.
"""

from __future__ import annotations

from collections.abc import Sequence

from finlam_prod.data.schema import (
    HEADER_CLASSES,
    NO_ARTICLE,
    TITLE_CLASSES,
    Page,
    Zone,
    ZoneClass,
)
from finlam_prod.pipeline.features import (
    column_index,
    columns_from_zones,
    columns_spanned,
    estimate_columns,
)


# ------------------------------------------------------------- ordre de lecture
def _gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intervalles vides entre des intervalles 1D (triés, fusionnés)."""
    intervals = sorted(intervals)
    gaps = []
    reach = intervals[0][1]
    for start, end in intervals[1:]:
        if start > reach:
            gaps.append((reach, start))
        reach = max(reach, end)
    return gaps


def xy_cut(
    zones: Sequence[Zone],
    columns: Sequence[tuple[float, float]] | None = None,
    dy: float = 0.1,
) -> list[Zone]:
    """Ordre de lecture par découpage XY récursif, guidé par les colonnes.

    À chaque étape :

    1. s'il existe une bande horizontale vide en travers de la région
       (intervalles verticaux rétrécis de ``dy`` pour tolérer les
       chevauchements d'annotation), on coupe haut / bas au niveau de la
       plus large ;
    2. sinon on coupe gauche / droite à la première frontière entre deux
       colonnes qu'aucune zone de la région ne chevauche ;
    3. sinon une zone chevauche toutes les frontières (titre pleine
       largeur) : ce qui est entièrement au-dessus de la plus haute est
       lu d'abord, puis elle, puis le reste ;
    4. à défaut, lecture colonne par colonne, de haut en bas.

    Les colonnes sont estimées sur les zones si elles ne sont pas fournies.
    Sur les pages de presse, où les polygones se touchent bord à bord, la
    règle 2 est celle qui fait le travail : une détection de gouttière
    par « vide » horizontal n'y trouverait rien.
    """
    if len(zones) <= 1:
        return list(zones)
    if columns is None:
        columns = columns_from_zones(zones)
    h_gaps = _gaps([(z.y0 + dy, z.y1 - dy) for z in zones])
    if h_gaps:
        cut = max(h_gaps, key=lambda g: g[1] - g[0])
        middle = (cut[0] + cut[1]) / 2
        top = [z for z in zones if z.cy < middle]
        bottom = [z for z in zones if z.cy >= middle]
        if top and bottom:
            return xy_cut(top, columns, dy) + xy_cut(bottom, columns, dy)
    spans = {id(z): columns_spanned(z, columns) for z in zones}
    present = sorted(
        {c for s in spans.values() for c in s} | {column_index(z, columns) for z in zones}
    )
    for left_col, right_col in zip(present, present[1:], strict=False):
        blocked = any(left_col in s and right_col in s for s in spans.values())
        if not blocked:
            left = [z for z in zones if column_index(z, columns) <= left_col]
            right = [z for z in zones if column_index(z, columns) > left_col]
            if left and right:
                return xy_cut(left, columns, dy) + xy_cut(right, columns, dy)
    spanning = [z for z in zones if len(spans[id(z)]) >= 2]
    if spanning:
        top_zone = min(spanning, key=lambda z: z.y0)
        above = [z for z in zones if z is not top_zone and z.y1 <= top_zone.y0 + dy]
        rest = [z for z in zones if z is not top_zone and z not in above]
        if above or rest:
            return xy_cut(above, columns, dy) + [top_zone] + xy_cut(rest, columns, dy)
    return sorted(zones, key=lambda z: (column_index(z, columns), z.y0, z.x0))


class XYCutOrderer:
    """Ordre de lecture par découpage XY (règle)."""

    name = "xy_cut"

    def __init__(self, dy: float = 0.1) -> None:
        self.dy = dy

    def fit(self, pages: list[Page]) -> XYCutOrderer:  # noqa: D102
        return self

    def predict(self, page: Page) -> list[int]:  # noqa: D102
        ordered = xy_cut(page.zones, estimate_columns(page), self.dy)
        rank = {id(z): i for i, z in enumerate(ordered)}
        return [rank[id(z)] for z in page.zones]


class ColumnOrderer:
    """Ordre de lecture naïf : colonne par colonne, de haut en bas.

    Référence « plancher » : ignore les titres pleine largeur, qui se
    retrouvent lus au milieu d'une colonne.
    """

    name = "columns_top_down"

    def fit(self, pages: list[Page]) -> ColumnOrderer:  # noqa: D102
        return self

    def predict(self, page: Page) -> list[int]:  # noqa: D102
        columns = estimate_columns(page)
        keyed = sorted(
            range(len(page.zones)),
            key=lambda i: (
                column_index(page.zones[i], columns),
                page.zones[i].y0,
                page.zones[i].x0,
            ),
        )
        rank = [0] * len(page.zones)
        for position, i in enumerate(keyed):
            rank[i] = position
        return rank


# ------------------------------------------------------------ classe de zone
class HeuristicClassifier:
    """Classe de zone par règles simples sur la géométrie et le texte.

    Volontairement grossière : elle fixe le niveau minimal que doit
    dépasser toute méthode apprise.
    """

    name = "heuristic"

    def fit(self, pages: list[Page]) -> HeuristicClassifier:  # noqa: D102
        return self

    def predict(self, page: Page) -> list[int]:  # noqa: D102
        return [self._classify(z, page) for z in page.zones]

    @staticmethod
    def _classify(zone: Zone, page: Page) -> int:
        text = zone.text.strip()
        letters = [c for c in text if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
        if not text:
            return ZoneClass.ARTICLE_ILLUSTRATION
        if page.index == 1 and zone.y0 < 8 and zone.width > 40:
            return ZoneClass.HEADER_TITLE
        if zone.y0 < 4 and zone.height < 3:
            return ZoneClass.HEADER_TEXT
        if len(text) < 80 and upper_ratio > 0.6:
            return ZoneClass.ARTICLE_TITLE
        if len(text) < 60 and zone.height < 2.5:
            return ZoneClass.ARTICLE_SUBTITLE
        return ZoneClass.ARTICLE_TEXT


# -------------------------------------------------------- séparation d'articles
class TitleRuleSegmenter:
    """Un nouvel article commence à chaque titre rencontré dans l'ordre de lecture.

    Règles :
    - les zones d'en-tête (HEADER-*) sont hors article ;
    - un ARTICLE-TITLE ou SECTION-TITLE ouvre un article ;
    - optionnellement, un « saut arrière » (zone qui remonte au-dessus de
      la précédente dans une autre colonne) ouvre aussi un article, car
      il signale un changement de bloc éditorial.
    """

    def __init__(self, split_on_back_jump: bool = False) -> None:
        self.split_on_back_jump = split_on_back_jump
        self.name = "title_rule+back_jump" if split_on_back_jump else "title_rule"

    def fit(self, pages: list[Page]) -> TitleRuleSegmenter:  # noqa: D102
        return self

    def predict(self, page: Page, order: list[int], classes: list[int]) -> list[int]:  # noqa: D102
        indices = sorted(range(len(page.zones)), key=lambda i: order[i])
        columns = estimate_columns(page)
        article = [NO_ARTICLE] * len(page.zones)
        current = 0
        previous: Zone | None = None
        for i in indices:
            zone, klass = page.zones[i], classes[i]
            if klass in HEADER_CLASSES:
                continue
            opens = klass in TITLE_CLASSES
            if self.split_on_back_jump and previous is not None:
                back_jump = zone.y0 < previous.y0 and column_index(zone, columns) != column_index(
                    previous, columns
                )
                opens = opens or back_jump
            if opens or current == 0:
                current += 1
            article[i] = current
            previous = zone
        return article
