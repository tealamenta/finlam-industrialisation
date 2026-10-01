"""Chargement du jeu FINLAM (Teklia/Newspapers-finlam, licence MIT).

Le jeu est diffusé sur Hugging Face au format Parquet. Une ligne = une
page, avec des colonnes « en parallèle » (une liste par attribut de
zone). Ce module convertit ces lignes en objets :class:`Page` et met le
résultat en cache au format JSON Lines, afin que le reste du projet
n'ait plus besoin de la bibliothèque ``datasets`` ni du réseau.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from finlam_prod.data.schema import Page, Zone, load_pages, save_pages
from finlam_prod.errors import DatasetError

LOGGER = logging.getLogger(__name__)

HF_DATASET = "Teklia/Newspapers-finlam"
SPLITS = ("train", "val", "test")


def row_to_page(row: dict[str, Any], image_dir: Path | None = None) -> Page:
    """Convertit une ligne du jeu Hugging Face en :class:`Page`.

    Args:
        row: ligne du jeu (dict de colonnes).
        image_dir: si fourni, l'image de la page y est enregistrée en JPEG.

    """
    page_id = row["page_arkindex_id"]
    image = row.get("page_image")
    width = height = None
    image_path = None
    if image is not None:
        width, height = image.size
        if image_dir is not None:
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(image_dir / f"{page_id}.jpg")
            if not Path(image_path).exists():
                image.convert("RGB").save(image_path, quality=90)

    zones = [
        Zone(
            id=zone_id,
            polygon=[(float(x), float(y)) for x, y in polygon],
            text=text or "",
            zone_class=int(zone_class),
            order=int(order),
            article_id=int(article_id),
        )
        for zone_id, polygon, text, zone_class, order, article_id in zip(
            row["zone_arkindex_ids"],
            row["zone_polygons"],
            row["zone_texts"],
            row["zone_classes"],
            row["zone_orders"],
            row["article_id"],
            strict=True,
        )
    ]
    return Page(
        id=page_id,
        issue_id=row["newspaper_arkindex_id"],
        index=int(row["page_index"]),
        zones=zones,
        width_px=width,
        height_px=height,
        image_path=image_path,
        source=row.get("newspaper_name", HF_DATASET),
    )


def download_split(
    split: str,
    cache_dir: str | Path,
    limit: int | None = None,
    with_images: bool = True,
) -> list[Page]:
    """Télécharge un split depuis Hugging Face et le met en cache.

    Nécessite l'extra ``data`` (``pip install finlam-prod[data]``) et un
    accès réseau à huggingface.co. Le cache est un fichier
    ``<cache_dir>/finlam_<split>.jsonl`` et, optionnellement, un dossier
    d'images ``<cache_dir>/images``.
    """
    from datasets import load_dataset  # import paresseux : dépendance optionnelle

    cache_dir = Path(cache_dir)
    target = cache_dir / f"finlam_{split}.jsonl"
    if target.exists() and limit is None:
        LOGGER.info("Cache présent, rechargement de %s", target)
        return load_pages(target)

    LOGGER.info("Téléchargement de %s (%s)", HF_DATASET, split)
    dataset = load_dataset(HF_DATASET, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    image_dir = cache_dir / "images" if with_images else None
    pages = [row_to_page(row, image_dir) for row in dataset]
    save_pages(pages, target)
    LOGGER.info("%d pages enregistrées dans %s", len(pages), target)
    return pages


def load_split(split: str, cache_dir: str | Path, dataset: str = "finlam") -> list[Page]:
    """Charge un split depuis le cache local (sans réseau).

    ``dataset`` vaut ``finlam`` (721 pages multi-titres) ou ``la-liberte``
    (8 836 pages du quotidien La Liberté, jeu des résultats de l'article).
    """
    prefix = dataset.replace("-", "_")
    target = Path(cache_dir) / f"{prefix}_{split}.jsonl"
    if not target.exists():
        raise DatasetError(
            f"{target} introuvable : lancez d'abord "
            f"`finlam-prod download --dataset {dataset} --split {split}`."
        )
    return load_pages(target)


def describe(pages: Iterable[Page]) -> dict[str, float]:
    """Statistiques descriptives simples sur un ensemble de pages."""
    pages = list(pages)
    n_zones = [len(p.zones) for p in pages]
    n_articles = [len(p.articles()) for p in pages]
    return {
        "pages": len(pages),
        "issues": len({p.issue_id for p in pages}),
        "zones": sum(n_zones),
        "zones_par_page_moy": sum(n_zones) / max(len(pages), 1),
        "articles_par_page_moy": sum(n_articles) / max(len(pages), 1),
    }
