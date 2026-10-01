"""Interfaces communes aux méthodes.

Les méthodes ne modifient jamais les pages en entrée : elles renvoient
des listes alignées sur ``page.zones``. C'est :class:`finlam_prod.pipeline.run.Pipeline`
qui écrit les prédictions dans les zones.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from finlam_prod.data.schema import Page


@runtime_checkable
class ZoneClassifier(Protocol):
    """Prédit la classe de chaque zone d'une page."""

    name: str

    def fit(self, pages: list[Page]) -> ZoneClassifier:  # noqa: D102
        ...

    def predict(self, page: Page) -> list[int]:
        """Classe prédite pour chaque zone, dans l'ordre de ``page.zones``."""
        ...


@runtime_checkable
class ReadingOrderer(Protocol):
    """Prédit l'ordre de lecture des zones d'une page."""

    name: str

    def fit(self, pages: list[Page]) -> ReadingOrderer:  # noqa: D102
        ...

    def predict(self, page: Page) -> list[int]:
        """Rang de lecture de chaque zone (0 = première), dans l'ordre de ``page.zones``."""
        ...


@runtime_checkable
class ArticleSegmenter(Protocol):
    """Regroupe les zones d'une page en articles."""

    name: str

    def fit(self, pages: list[Page]) -> ArticleSegmenter:  # noqa: D102
        ...

    def predict(self, page: Page, order: list[int], classes: list[int]) -> list[int]:
        """Identifiant d'article de chaque zone (NO_ARTICLE hors article).

        Args:
            page: la page.
            order: rang de lecture de chaque zone (prédit ou vérité terrain).
            classes: classe de chaque zone (prédite ou vérité terrain).

        """
        ...
