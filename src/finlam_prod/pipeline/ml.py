"""Méthodes par apprentissage classique (scikit-learn).

Ces modèles s'entraînent en quelques minutes sur CPU avec les 623
pages d'entraînement de FINLAM, ce qui en fait des candidats réalistes
pour une chaîne de production à 20 millions de pages par an.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from finlam_prod.data.schema import HEADER_CLASSES, NO_ARTICLE, Page
from finlam_prod.pipeline.features import (
    estimate_columns,
    page_feature_matrix,
    pair_features,
)


def _boundaries_from_gold(page: Page, indices: list[int]) -> list[int]:
    """Étiquette « ouvre un article » (1/0) pour chaque position dans l'ordre donné."""
    labels = []
    previous = None
    for i in indices:
        current = page.zones[i].article_id
        if current is None or current == NO_ARTICLE:
            labels.append(0)
            continue
        labels.append(int(current != previous))
        previous = current
    return labels


class GradientBoostingZoneClassifier:
    """Classe de zone : gradient boosting sur géométrie + texte (TF-IDF caractères réduit)."""

    name = "hgb_zone"

    def __init__(self, n_text_components: int = 40, random_state: int = 0) -> None:
        self.text_pipe = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=50000),
            TruncatedSVD(n_components=n_text_components, random_state=random_state),
        )
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=random_state
        )

    def _matrix(self, pages: list[Page], fit_text: bool = False) -> np.ndarray:
        geometry = np.vstack([page_feature_matrix(p) for p in pages])
        texts = [z.text for p in pages for z in p.zones]
        text = self.text_pipe.fit_transform(texts) if fit_text else self.text_pipe.transform(texts)
        # Arrondi volontaire : les composantes SVD portent un bruit numérique
        # (~1e-17) qui perturbe le binning en quantiles du gradient boosting.
        return np.hstack([geometry, np.round(text, 6)])

    def fit(self, pages: list[Page]) -> GradientBoostingZoneClassifier:  # noqa: D102
        x = self._matrix(pages, fit_text=True)
        y = np.array([z.zone_class for p in pages for z in p.zones])
        self.model.fit(x, y)
        return self

    def predict(self, page: Page) -> list[int]:  # noqa: D102
        if not page.zones:
            return []
        return [int(c) for c in self.model.predict(self._matrix([page]))]

    def predict_proba(self, page: Page) -> np.ndarray:
        """Probabilités par classe (n_zones × n_classes)."""
        return self.model.predict_proba(self._matrix([page]))

    def save(self, path: str | Path) -> None:  # noqa: D102
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> GradientBoostingZoneClassifier:  # noqa: D102
        return joblib.load(path)


class BoundarySegmenter:
    """Séparation d'articles apprise : « cette zone ouvre-t-elle un article ? ».

    Dans l'ordre de lecture, chaque zone est comparée à la précédente
    (descripteurs de couple, dont les classes). Un classifieur prédit
    une frontière ; les identifiants d'article sont la somme cumulée des
    frontières. Les zones d'en-tête restent hors article.
    """

    name = "boundary_hgb"

    def __init__(self, class_key: str = "zone_class", random_state: int = 0) -> None:
        self.class_key = class_key
        self.model = HistGradientBoostingClassifier(max_iter=300, random_state=random_state)

    def fit(self, pages: list[Page]) -> BoundarySegmenter:  # noqa: D102
        rows, labels = [], []
        for page in pages:
            indices = sorted(range(len(page.zones)), key=lambda i: page.zones[i].order)
            columns = estimate_columns(page)
            gold = _boundaries_from_gold(page, indices)
            for pos in range(1, len(indices)):
                a, b = page.zones[indices[pos - 1]], page.zones[indices[pos]]
                rows.append(pair_features(a, b, page, columns, class_key=self.class_key))
                labels.append(gold[pos])
        self.model.fit(np.array(rows), np.array(labels))
        return self

    def predict(self, page: Page, order: list[int], classes: list[int]) -> list[int]:  # noqa: D102
        n = len(page.zones)
        if n == 0:
            return []
        indices = sorted(range(n), key=lambda i: order[i])
        columns = estimate_columns(page)
        # Les descripteurs de couple lisent la classe sur les zones : on
        # travaille sur des copies portant les classes fournies.
        zones = []
        for zone, klass in zip(page.zones, classes, strict=True):
            copy = type(zone)(**{**zone.__dict__, "zone_class": klass})
            zones.append(copy)
        scratch = type(page)(**{**page.__dict__, "zones": zones})
        rows = [
            pair_features(zones[indices[pos - 1]], zones[indices[pos]], scratch, columns)
            for pos in range(1, n)
        ]
        boundaries = [1] + ([int(b) for b in self.model.predict(np.array(rows))] if rows else [])
        article = [NO_ARTICLE] * n
        current = 0
        for pos, i in enumerate(indices):
            if classes[i] in HEADER_CLASSES:
                continue
            if boundaries[pos] or current == 0:
                current += 1
            article[i] = current
        return article

    def save(self, path: str | Path) -> None:  # noqa: D102
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BoundarySegmenter:  # noqa: D102
        return joblib.load(path)


__all__ = ["BoundarySegmenter", "GradientBoostingZoneClassifier"]
