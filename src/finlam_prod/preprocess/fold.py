"""Journaux pliés et scannés : détection du pli, éclairage, redressement par moitié.

Un journal plié en deux puis numérisé à plat présente trois défauts qui
dégradent l'OCR et la segmentation en amont du pipeline FINLAM :

1. une **bande d'ombre** (ou de reflet) le long du pli, que les modèles
   prennent pour un trait vertical ou une bordure de colonne ;
2. une **inclinaison différente de chaque côté** du pli, le papier ne
   reposant pas à plat de la même façon ;
3. parfois une **perte de texte** dans le pli lui-même, irrécupérable.

Ce module traite les deux premiers avec OpenCV, sur CPU, en quelques
dizaines de millisecondes par page :

- :func:`detect_fold` cherche une bande verticale anormalement sombre ou
  claire dans le profil des colonnes, vérifie qu'elle traverse la page de
  haut en bas, et renvoie sa position et sa force ;
- :func:`normalise_illumination` divise l'image par son fond estimé
  (fermeture morphologique puis flou), ce qui gomme l'ombre du pli et le
  vignettage sans toucher à l'encre ;
- :func:`deskew_angle` estime l'inclinaison d'un bloc de texte par le
  contraste du profil des lignes (maximal quand les lignes sont
  horizontales) ;
- :func:`preprocess_page` enchaîne les trois et, si un pli est détecté,
  redresse chaque moitié séparément.

Le troisième défaut ne se corrige pas : il se **signale**. Le rapport
:class:`FoldReport` est fait pour alimenter le contrôle qualité (page à
relire) et les journaux, pas pour être ignoré.

Limites assumées : pas de dégauchissement 3D (livre gondolé), qui relève de
modèles appris (DocUNet, DewarpNet) ; pas de pli horizontal, rare sur la
presse (il suffirait de transposer l'image).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from finlam_prod.errors import PreprocessError

LOGGER = logging.getLogger(__name__)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise PreprocessError(
            "OpenCV absent : installer l'extra image (pip install 'finlam-prod[image]')"
        ) from exc
    return cv2


@dataclass(frozen=True)
class FoldReport:
    """Résultat de la détection de pli sur une page."""

    detected: bool
    x: int  # colonne du pli (pixels)
    depth: float  # écart de la bande au fond, en niveaux de gris (négatif : ombre)
    strength: float  # le même écart, en écarts-types du profil résiduel
    width: int  # largeur de la bande (pixels)
    dark: bool  # ombre (True) ou reflet (False)

    def to_dict(self) -> dict[str, Any]:
        """Forme sérialisable (rapport QC, journaux)."""
        return asdict(self)


@dataclass(frozen=True)
class PreprocessResult:
    """Image prétraitée et ce qui lui a été fait."""

    image: np.ndarray
    fold: FoldReport
    angles: tuple[float, ...]  # rotation appliquée à chaque moitié (ou à la page entière)

    def to_dict(self) -> dict[str, Any]:
        """Forme sérialisable, sans l'image."""
        return {"fold": self.fold.to_dict(), "angles": list(self.angles)}


def to_gray(image: np.ndarray) -> np.ndarray:
    """Niveaux de gris 8 bits, quelle que soit l'entrée (gris, BGR, BGRA)."""
    cv2 = _cv2()
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def default_radius(gray: np.ndarray) -> int:
    """Rayon de fermeture : ~1/40 de la largeur, impair (31 px pour une page de 1200 px).

    Il doit dépasser la hauteur d'une ligne de texte (pour la combler) et
    rester inférieur à la largeur de l'ombre d'un pli (pour la conserver).
    """
    return max(15, (gray.shape[1] // 40) | 1)


def estimate_background(gray: np.ndarray, radius: int | None = None) -> np.ndarray:
    """Fond de la page sans l'encre : fermeture morphologique puis léger flou.

    L'encre, plus fine que ``radius``, est comblée ; l'ombre du pli, plus
    large, et le vignettage restent. C'est sur ce fond que l'on cherche le
    pli et par lui que l'on normalise l'éclairage. Le flou est volontairement
    faible : trop fort, il effacerait l'ombre du fond et elle resterait
    dans l'image corrigée.
    """
    cv2 = _cv2()
    radius = max(3, (radius or default_radius(gray)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(background, (0, 0), max(1.0, radius / 8))


def normalise_illumination(gray: np.ndarray, radius: int | None = None) -> np.ndarray:
    """Divise l'image par son fond estimé : ombre du pli et vignettage disparaissent."""
    background = estimate_background(gray, radius)
    ratio = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    return np.clip(ratio * 255.0, 0, 255).astype(np.uint8)


def _column_residual(background: np.ndarray) -> np.ndarray:
    """Profil des colonnes du fond moins sa tendance lente : les bandes étroites ressortent."""
    cv2 = _cv2()
    profile = background.mean(axis=0).astype(np.float32)
    sigma = max(5.0, background.shape[1] / 12)
    trend = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigma).ravel()
    return profile - trend


def detect_fold(
    gray: np.ndarray,
    search: tuple[float, float] = (0.2, 0.8),
    min_strength: float = 4.0,
    min_depth: float = 10.0,
    allow_bright: bool = False,
) -> FoldReport:
    """Cherche une bande verticale sombre ou claire traversant la page.

    ``search`` borne la zone de recherche en fraction de la largeur (un pli
    est central ; les marges, elles, sont pleines de bordures). La bande est
    retenue si elle sort de ``min_strength`` écarts-types du profil résiduel,
    si elle s'écarte du fond d'au moins ``min_depth`` niveaux de gris (une
    page déjà plate ne doit rien déclencher) et si elle est présente dans
    les trois tiers de la hauteur. Un pli est une **ombre** : une bande
    *claire* est presque toujours de la mise en page (colonne vide d'une
    pleine page de publicité, marge) et n'est retenue que si
    ``allow_bright`` est vrai.
    """
    height, width = gray.shape[:2]
    background = estimate_background(gray)
    residual = _column_residual(background)
    lo, hi = int(width * search[0]), int(width * search[1])
    window = residual[lo:hi]
    x = lo + int(np.argmax(np.abs(window)))
    depth = float(residual[x])
    dark = bool(depth < 0)
    # Échelle robuste (MAD) mesurée hors de la bande candidate : une ombre profonde
    # gonflerait l'écart-type classique et se masquerait elle-même.
    exclude = max(5, width // 20)
    outside = np.concatenate([residual[: max(0, x - exclude)], residual[x + exclude :]])
    scale = 1.4826 * float(np.median(np.abs(outside - np.median(outside)))) + 1e-6
    strength = abs(depth) / scale

    half = max(1, int(abs(residual[x]) * 0.5))
    inside = np.abs(residual) >= half
    left = x
    while left > 0 and inside[left - 1]:
        left -= 1
    right = x
    while right < width - 1 and inside[right + 1]:
        right += 1
    band_width = right - left + 1

    # La bande doit traverser la page : même signe et au moins la moitié de la force
    # dans chacun des trois tiers.
    consistent = True
    for k in range(3):
        part = background[k * height // 3 : (k + 1) * height // 3]
        part_residual = _column_residual(part)
        value = float(part_residual[x])
        if (value < 0) != dark or abs(value) < abs(float(residual[x])) * 0.5:
            consistent = False
            break

    detected = (
        strength >= min_strength
        and abs(depth) >= min_depth
        and consistent
        and band_width < width * 0.1
        and (dark or allow_bright)
    )
    report = FoldReport(detected, x, round(depth, 1), round(strength, 2), band_width, dark)
    LOGGER.debug("détection de pli : %s", report.to_dict())
    return report


def _text_mask(gray: np.ndarray, target_width: int = 1000, margin: float = 0.05) -> np.ndarray:
    """Masque de l'encre (Otsu inversé), réduit à ``target_width`` ; bords de scan écartés."""
    cv2 = _cv2()
    scale = min(1.0, target_width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    m = int(margin * min(mask.shape))
    return mask[m : mask.shape[0] - m, m : mask.shape[1] - m] if m else mask


def deskew_angle(gray: np.ndarray, max_angle: float = 3.0, step: float = 0.1) -> float:
    """Angle (degrés) à appliquer pour rendre les lignes de texte horizontales.

    Balaye ``[-max_angle, max_angle]`` et retient l'angle qui rend le profil
    des lignes le plus contrasté (somme des carrés de ses différences) : des
    lignes bien horizontales donnent des pics nets séparés par des creux. Un
    bloc trop pauvre en texte renvoie 0.
    """
    cv2 = _cv2()
    mask = _text_mask(gray)
    if mask.mean() < 1.0:  # moins de 0,4 % de pixels d'encre : rien à redresser
        return 0.0
    h, w = mask.shape
    centre = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step / 2, step):
        matrix = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rotated = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
        profile = rotated.sum(axis=1, dtype=np.float64)
        score = float(np.sum(np.diff(profile) ** 2))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return round(best_angle, 2)


def rotate(
    image: np.ndarray, angle: float, centre: tuple[float, float] | None = None
) -> np.ndarray:
    """Rotation autour de ``centre`` (défaut : centre de l'image), fond blanc, mêmes dimensions.

    Pour une moitié de page pliée, le centre est un point **sur le pli** : le pli
    reste en place et le coin vide créé par la rotation part vers la marge
    extérieure, pas vers le texte du milieu de page.
    """
    cv2 = _cv2()
    if abs(angle) < 1e-3:
        return image.copy()
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(centre or (w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def straighten_halves(image: np.ndarray, fold_x: int, angles: tuple[float, float]) -> np.ndarray:
    """Redresse chaque côté du pli de son propre angle, en une seule image.

    Chaque côté tourne autour d'un point du pli, mais les pixels sont pris dans
    l'image **entière** : près du pli, un côté redressé emprunte ce dont il a
    besoin de l'autre côté, là où ce texte se trouve réellement — pas une bande
    blanche. Seules les marges extérieures reçoivent du blanc.
    """
    cv2 = _cv2()
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x, map_y = xs.copy(), ys.copy()
    for side, angle in zip(((0, fold_x), (fold_x, w)), angles, strict=True):
        lo, hi = side
        if hi <= lo or abs(angle) < 1e-3:
            continue
        # matrice inverse : pour chaque pixel de destination, d'où vient-il ?
        inv = cv2.getRotationMatrix2D((float(fold_x), h / 2), -angle, 1.0)
        dx, dy = xs[:, lo:hi], ys[:, lo:hi]
        map_x[:, lo:hi] = inv[0, 0] * dx + inv[0, 1] * dy + inv[0, 2]
        map_y[:, lo:hi] = inv[1, 0] * dx + inv[1, 1] * dy + inv[1, 2]
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def preprocess_page(image: np.ndarray, max_angle: float = 3.0) -> PreprocessResult:
    """Chaîne complète : gris → détection du pli → éclairage → redressement par moitié."""
    gray = to_gray(image)
    fold = detect_fold(gray)
    flat = normalise_illumination(gray)
    angles: tuple[float, ...]
    if fold.detected:
        left, right = flat[:, : fold.x], flat[:, fold.x :]
        # la bande du pli, même atténuée, fausse l'estimation : on l'exclut de la mesure
        margin = fold.width
        angles = (
            deskew_angle(left[:, : max(1, left.shape[1] - margin)], max_angle),
            deskew_angle(right[:, min(margin, right.shape[1] - 1) :], max_angle),
        )
        out = straighten_halves(flat, fold.x, (angles[0], angles[1]))
    else:
        angles = (deskew_angle(flat, max_angle),)
        out = rotate(flat, angles[0])
    LOGGER.info(
        "prétraitement : pli=%s x=%d force=%.1f angles=%s",
        fold.detected,
        fold.x,
        fold.strength,
        angles,
        extra={"fold": fold.to_dict(), "angles": list(angles)},
    )
    return PreprocessResult(out, fold, angles)
