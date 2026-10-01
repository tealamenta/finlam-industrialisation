"""Dégauchissement appris d'une page photographiée (pliée en 3D, gondolée, en perspective).

Le prétraitement OpenCV de :mod:`finlam_prod.preprocess.fold` traite le cas
« journal plié puis scanné à plat » : une ombre, deux inclinaisons. Une
**photo** d'une page pliée ou gondolée est un autre problème : la surface est
en 3D et la géométrie entière est déformée. Il faut un modèle appris.

On utilise **UVDoc** (Verhoeven, Magne & Sorkine-Hornung, *UVDoc: Neural
Grid-based Document Unwarping*, SIGGRAPH Asia 2023 — licence MIT) : un réseau
convolutif qui prédit, pour une image réduite à 488 × 712, une grille 2D de
45 × 31 points ; l'image pleine résolution est rééchantillonnée sur cette
grille (``grid_sample``). CPU, ~1 s par page. DocTr (Feng et al., 2021) fait
la même chose avec un transformeur, mais sous licence non commerciale : il
peut remplacer UVDoc derrière la même interface si la BnF obtient l'accord
de l'auteur.

Chaîne complète pour une photo : :func:`dewarp_page` (géométrie) puis
:func:`finlam_prod.preprocess.preprocess_page` (éclairage, pli résiduel,
inclinaison). Dépendances : extra ``dewarp`` (torch) ; les poids (32 Mo) sont
téléchargés une fois par :func:`fetch_weights` et vérifiés par empreinte.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from finlam_prod.errors import PreprocessError

LOGGER = logging.getLogger(__name__)

WEIGHTS_URL = "https://github.com/tanguymagne/UVDoc/raw/main/model/best_model.pkl"
WEIGHTS_SHA256 = "7e90861b8a516eb4bc51f84bd889cb77275743d2d1d3ca8091951ec9f2b7da23"
WEIGHTS_NAME = "uvdoc_best_model.pkl"
INPUT_SIZE = (488, 712)  # largeur, hauteur attendues par le réseau
GRID_SIZE = (45, 31)  # lignes, colonnes de la grille prédite


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise PreprocessError(
            "torch absent : installer l'extra dewarp (pip install 'finlam-prod[dewarp]')"
        ) from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_weights(models_dir: str | Path, url: str = WEIGHTS_URL) -> Path:
    """Télécharge les poids UVDoc dans ``models_dir`` s'ils n'y sont pas ; vérifie l'empreinte."""
    import requests

    target = Path(models_dir) / WEIGHTS_NAME
    if target.is_file() and _sha256(target) == WEIGHTS_SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("téléchargement des poids UVDoc depuis %s", url)
    try:
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
    except requests.RequestException as exc:
        raise PreprocessError(f"poids UVDoc introuvables ({url}) : {exc}") from exc
    actual = _sha256(target)
    if actual != WEIGHTS_SHA256:
        target.unlink(missing_ok=True)
        raise PreprocessError(
            f"poids UVDoc corrompus : sha256 {actual[:12]}… ≠ attendu {WEIGHTS_SHA256[:12]}…"
        )
    return target


def load_model(weights: str | Path) -> Any:
    """Charge UVDocnet sur CPU (ou GPU s'il y en a un) en mode évaluation."""
    torch = _torch()
    from finlam_prod.preprocess.uvdoc_net import UVDocnet

    weights = Path(weights)
    if not weights.is_file():
        raise PreprocessError(f"poids UVDoc absents : {weights} (voir fetch_weights)")
    model = UVDocnet(num_filter=32, kernel_size=5)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    LOGGER.info("UVDoc chargé (%s)", device)
    return model


def predict_grid(model: Any, image_bgr: np.ndarray) -> np.ndarray:
    """Grille 2D (2 × 45 × 31, coordonnées normalisées [-1, 1]) prédite pour l'image."""
    torch = _torch()
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    small = cv2.resize(rgb, INPUT_SIZE).transpose(2, 0, 1)
    device = next(model.parameters()).device
    with torch.no_grad():
        grid, _ = model(torch.from_numpy(small).unsqueeze(0).to(device))
    result: np.ndarray = grid[0].detach().cpu().numpy()
    return result


def unwarp(image_bgr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Rééchantillonne l'image pleine résolution sur la grille prédite."""
    torch = _torch()
    import torch.nn.functional as functional

    h, w = image_bgr.shape[:2]
    img = torch.from_numpy(image_bgr.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
    points = torch.from_numpy(grid).unsqueeze(0)
    dense = functional.interpolate(points, size=(h, w), mode="bilinear", align_corners=True)
    out = functional.grid_sample(
        img, dense.transpose(1, 2).transpose(2, 3), align_corners=True, padding_mode="border"
    )
    result: np.ndarray = (out[0].numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return result


def dewarp_page(image: np.ndarray, model: Any) -> np.ndarray:
    """Image (BGR ou gris) → page dégauchie, mêmes dimensions, BGR."""
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3]
    grid = predict_grid(model, bgr)
    LOGGER.debug(
        "UVDoc : grille %s, étendue x [%.2f, %.2f]", grid.shape, grid[0].min(), grid[0].max()
    )
    return unwarp(bgr, grid)
