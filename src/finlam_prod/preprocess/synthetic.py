"""Pages pliées à vérité connue, pour tests et démonstration.

Deux sources :

- :func:`folded_page` fabrique une page à colonnes de « texte » (traits de
  mots) — reproductible, sans donnée externe, pour les tests ;
- :func:`simulate_fold` applique le même défaut à une **vraie page scannée**
  (La Liberté, Gallica…) : texte, mise en page et bruit réels, défaut
  contrôlé — pour mesurer ce que le prétraitement récupère sur de vraies
  images, qu'on n'a pas en version pliée *et* à plat.

Dans les deux cas : chaque moitié inclinée d'un angle différent, ombre
gaussienne le long du pli, léger vignettage, bruit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldedPage:
    """Image synthétique et ses paramètres de fabrication."""

    image: np.ndarray  # gris uint8
    clean: np.ndarray  # la même page sans pli, sans inclinaison, sans ombre
    fold_x: int
    angles: tuple[float, float]


def _text_block(rng: np.random.Generator, height: int, width: int, columns: int) -> np.ndarray:
    """Colonnes de lignes de mots gris foncé sur fond blanc."""
    page = np.full((height, width), 245, dtype=np.uint8)
    gutter = width // (columns * 12)
    col_w = (width - gutter * (columns + 1)) // columns
    line_h, gap = max(6, height // 120), max(3, height // 240)
    for c in range(columns):
        x0 = gutter + c * (col_w + gutter)
        y = gutter * 2
        while y + line_h < height - gutter * 2:
            x = x0
            while x < x0 + col_w - 4:
                word = int(rng.integers(line_h, line_h * 4))
                word = min(word, x0 + col_w - x)
                page[y : y + line_h, x : x + word] = int(rng.integers(20, 60))
                x += word + max(2, line_h // 2)
            y += line_h + gap
            if rng.random() < 0.04:  # saut de paragraphe
                y += line_h
    return page


def simulate_fold(
    clean: np.ndarray,
    fold: float = 0.5,
    angles: tuple[float, float] = (1.5, -1.0),
    shadow_depth: int = 70,
    shadow_width: int | None = None,
    seed: int = 0,
) -> FoldedPage:
    """Plie une page à plat (gris uint8) : inclinaison par moitié, ombre, vignettage, bruit."""
    import cv2

    rng = np.random.default_rng(seed)
    height, width = clean.shape[:2]
    fold_x = int(width * fold)
    shadow_width = shadow_width or max(20, width // 30)

    def tilt(part: np.ndarray, angle: float, hinge_x: float) -> np.ndarray:
        # le papier pivote autour du pli : le point (hinge_x, h/2) reste en place
        h, w = part.shape
        matrix = cv2.getRotationMatrix2D((hinge_x, h / 2), angle, 1.0)
        return cv2.warpAffine(
            part, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    image = np.hstack(
        [
            tilt(clean[:, :fold_x], angles[0], hinge_x=fold_x),
            tilt(clean[:, fold_x:], angles[1], hinge_x=0.0),
        ]
    )
    xs = np.arange(width, dtype=np.float32)
    shadow = shadow_depth * np.exp(-0.5 * ((xs - fold_x) / (shadow_width / 2.355)) ** 2)
    ys = np.linspace(-1, 1, height, dtype=np.float32)
    vignette = 1.0 - 0.08 * (ys[:, None] ** 2 + ((xs[None, :] - width / 2) / (width / 2)) ** 2)
    out = (image.astype(np.float32) - shadow[None, :]) * vignette
    out += rng.normal(0, 3, out.shape).astype(np.float32)
    return FoldedPage(np.clip(out, 0, 255).astype(np.uint8), clean, fold_x, angles)


def folded_page(
    width: int = 1200,
    height: int = 1600,
    columns: int = 6,
    fold: float = 0.5,
    angles: tuple[float, float] = (1.5, -1.0),
    shadow_depth: int = 70,
    shadow_width: int = 40,
    seed: int = 0,
) -> FoldedPage:
    """Fabrique une page pliée : inclinaison par moitié, ombre du pli, vignettage."""
    rng = np.random.default_rng(seed)
    clean = _text_block(rng, height, width, columns)
    return simulate_fold(clean, fold, angles, shadow_depth, shadow_width, seed)
