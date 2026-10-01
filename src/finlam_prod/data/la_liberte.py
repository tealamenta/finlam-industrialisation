"""Chargement du jeu « Finlam La Liberté » (Teklia/Newspapers-finlam-La-Liberte, MIT).

C'est le jeu sur lequel l'article du consortium (Mocaër et al., 2026,
arXiv 2607.15082) calcule ses résultats : 1 500 numéros du quotidien
*La Liberté* (1925-1928), 8 836 pages, splits train 7 957 / val 446 /
test 433 pages. Pour comparer nos chiffres aux leurs, il faut évaluer
sur **son** split test, pas sur le petit jeu multi-titres.

Différences avec ``Newspapers-finlam`` :

- 16 classes de zones au lieu de 13 : on les ramène aux 13 classes du
  paquet par une table de correspondance **par nom** (la liste des noms
  est lue dans les métadonnées du jeu au téléchargement et enregistrée
  à côté du cache, ``la_liberte_classes.json``) ; le libellé d'origine
  est conservé dans ``Zone.raw_class`` ;
- un identifiant de **section** par zone (``zone_section_ids``), exposé
  dans ``Zone.section_id`` ;
- les noms de colonnes diffèrent (``zone_article_ids`` au lieu de
  ``article_id``) ; le chargeur accepte les deux.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from finlam_prod.data.schema import (
    NO_ARTICLE,
    Page,
    Zone,
    ZoneClass,
    load_pages,
    save_pages,
)
from finlam_prod.errors import DatasetError

LOGGER = logging.getLogger(__name__)

HF_DATASET = "Teklia/Newspapers-finlam-La-Liberte"
SPLITS = ("train", "validation", "test")
CACHE_PREFIX = "la_liberte"
SPLIT_FILE_PREFIX = {"validation": "val"}  # nom du split et des parquets sur le Hub

CLASS_MAPPING: dict[str, ZoneClass] = {
    # même nom, avec ou sans préfixe ARTICLE-
    "HEADER-TITLE": ZoneClass.HEADER_TITLE,
    "HEADER-TEXT": ZoneClass.HEADER_TEXT,
    "ARTICLE-ILLUSTRATION": ZoneClass.ARTICLE_ILLUSTRATION,
    "ILLUSTRATION": ZoneClass.ARTICLE_ILLUSTRATION,
    "ADVERTISEMENT": ZoneClass.ADVERTISEMENT,
    "ANNOUNCEMENT": ZoneClass.ANNOUNCEMENT,
    "FREEAD": ZoneClass.ANNOUNCEMENT,
    "ARTICLE-TITLE": ZoneClass.ARTICLE_TITLE,
    "TITLE": ZoneClass.ARTICLE_TITLE,
    "ARTICLE-TEXT": ZoneClass.ARTICLE_TEXT,
    "TEXT": ZoneClass.ARTICLE_TEXT,
    "ILLUSTRATEDTEXT": ZoneClass.ARTICLE_TEXT,
    "ARTICLE-ILLUSTRATEDTEXT": ZoneClass.ARTICLE_TEXT,
    "SECTION-ILLUSTRATEDTEXT": ZoneClass.ARTICLE_TEXT,
    "ARTICLE-SUBTITLE": ZoneClass.ARTICLE_SUBTITLE,
    "SUBTITLE": ZoneClass.ARTICLE_SUBTITLE,
    "ARTICLE-INSIDEHEADING": ZoneClass.ARTICLE_INSIDEHEADING,
    "INSIDEHEADING": ZoneClass.ARTICLE_INSIDEHEADING,
    "CAPTION": ZoneClass.CAPTION,
    "AUTHOR": ZoneClass.AUTHOR,
    "ARTICLE-TABLE": ZoneClass.ARTICLE_TABLE,
    "TABLE": ZoneClass.ARTICLE_TABLE,
    "SECTION-TITLE": ZoneClass.SECTION_TITLE,
    "SECTION-SUBTITLE": ZoneClass.SECTION_TITLE,
    "ARTICLE": ZoneClass.ARTICLE_TEXT,
}
"""Correspondance nom de classe La Liberté → classe du paquet.

Les noms viennent de la taxonomie décrite dans l'article (HEADER-*,
SECTION-*, TITLE, SUBTITLE, ILLUSTRATEDTEXT, TEXT, ILLUSTRATION, CAPTION,
TABLE, AUTHOR, INSIDEHEADING, ARTICLE, ADVERTISEMENT, FREEAD). Un nom
absent de la table est signalé une fois et la zone garde ``zone_class=None``
avec son ``raw_class`` : compléter la table plutôt que de deviner.
"""


def class_names_from_features(features: Any) -> list[str] | None:
    """Noms de classes déclarés par le jeu (ClassLabel), s'il les expose."""
    try:
        feature = features["zone_classes"].feature
        names = getattr(feature, "names", None)
        return list(names) if names else None
    except (KeyError, AttributeError):
        return None


def map_class(
    raw: Any, names: list[str] | None, unknown: set[str]
) -> tuple[int | None, str | None]:
    """Convertit une classe brute (entier ou nom) en classe du paquet et libellé d'origine."""
    if isinstance(raw, str):
        label = raw
    elif names is not None and 0 <= int(raw) < len(names):
        label = names[int(raw)]
    else:
        # sans liste de noms, on suppose l'ordre des 13 classes du paquet
        try:
            return int(ZoneClass(int(raw))), None
        except ValueError:
            unknown.add(str(raw))
            return None, str(raw)
    key = label.upper().replace("_", "-")
    mapped = CLASS_MAPPING.get(key)
    if mapped is None:
        unknown.add(label)
        return None, label
    return int(mapped), label


def polygon_scale(polygons: list) -> float:
    """Facteur ramenant les polygones en pourcentage de page.

    Le jeu multi-titres stocke les coordonnées en pourcentage (0-100),
    La Liberté en fraction (0-1) : on détecte l'échelle sur le maximum observé.
    """
    top = max((float(v) for poly in polygons for point in poly for v in point), default=0.0)
    return 100.0 if top <= 1.5 else 1.0


def row_to_page(
    row: dict[str, Any],
    names: list[str] | None,
    unknown: set[str],
    image_dir: Path | None = None,
) -> Page:
    """Convertit une ligne du jeu La Liberté en :class:`Page`."""
    page_id = (
        row.get("page_arkindex_id") or row.get("page_id") or f"page_{row.get('page_index', 0)}"
    )
    issue_id = (
        row.get("newspaper_arkindex_id")
        or row.get("issue_id")
        or row.get("newspaper_name")
        or "la-liberte"
    )
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

    articles = row.get("zone_article_ids", row.get("article_id")) or [None] * len(
        row["zone_polygons"]
    )
    sections = row.get("zone_section_ids") or [None] * len(row["zone_polygons"])
    zone_ids = row.get("zone_arkindex_ids") or [
        f"{page_id}_z{i}" for i in range(len(row["zone_polygons"]))
    ]
    zones = []
    scale = polygon_scale(row["zone_polygons"])
    for zid, polygon, text, raw_class, order, article, section in zip(
        zone_ids,
        row["zone_polygons"],
        row["zone_texts"],
        row["zone_classes"],
        row["zone_orders"],
        articles,
        sections,
        strict=True,
    ):
        klass, label = map_class(raw_class, names, unknown)
        zones.append(
            Zone(
                id=str(zid),
                polygon=[(float(x) * scale, float(y) * scale) for x, y in polygon],
                text=text or "",
                zone_class=klass,
                order=int(order),
                article_id=int(article) if article is not None else NO_ARTICLE,
                section_id=int(section) if section is not None else None,
                raw_class=label,
            )
        )
    return Page(
        id=str(page_id),
        issue_id=str(issue_id),
        index=int(row.get("page_index", 0)),
        zones=zones,
        width_px=width,
        height_px=height,
        image_path=image_path,
        source=f"La Liberté — {row.get('newspaper_name', issue_id)}",
    )


def download_split(
    split: str,
    cache_dir: str | Path,
    limit: int | None = None,
    with_images: bool = False,
) -> list[Page]:
    """Télécharge un split de La Liberté et le met en cache (JSON Lines).

    Le jeu complet pèse 10 Go ; par défaut les images ne sont pas
    enregistrées (elles ne servent qu'aux notebooks). Nécessite l'extra
    ``data`` et un accès à huggingface.co.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetError("extra 'data' absent : pip install 'finlam-prod[data]'") from exc

    cache_dir = Path(cache_dir)
    suffix = f"_n{limit}" if limit is not None else ""
    # un --limit écrit un cache à part : il n'écrase pas le split complet
    target = cache_dir / f"{CACHE_PREFIX}_{split}{suffix}.jsonl"
    if target.exists():
        LOGGER.info("Cache présent, rechargement de %s", target)
        return load_pages(target)
    LOGGER.info("Téléchargement de %s (%s)", HF_DATASET, split)
    # Seuls les parquets du split demandé sont téléchargés et convertis : le jeu
    # complet (10 Go, images incluses) ne tient pas forcément sur un poste.
    prefix = SPLIT_FILE_PREFIX.get(split, split)
    dataset = load_dataset(
        HF_DATASET,
        split=prefix,
        data_files={prefix: f"data/{prefix}-*.parquet"},
        verification_mode="no_checks",  # sinon datasets exige les autres splits
    )
    names = class_names_from_features(dataset.features)
    if names is None:
        # les parquets seuls peuvent perdre les ClassLabel : on relit la carte du jeu
        from datasets import load_dataset_builder

        names = class_names_from_features(load_dataset_builder(HF_DATASET).info.features)
    if names is None:
        raise DatasetError(
            "Noms de classes de La Liberté introuvables : sans eux les entiers seraient "
            "mal interprétés (16 classes ≠ 13 du paquet). Voir la carte du jeu sur le Hub."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{CACHE_PREFIX}_classes.json").write_text(
        json.dumps(
            {
                "class_names": names,
                "mapping": {k: v.label for k, v in CLASS_MAPPING.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    unknown: set[str] = set()
    image_dir = cache_dir / "images_la_liberte" if with_images else None
    pages = [row_to_page(row, names, unknown, image_dir) for row in dataset]
    if unknown:
        LOGGER.warning(
            "Classes La Liberté sans correspondance (zone_class=None, raw_class conservé) : %s "
            "— compléter CLASS_MAPPING dans finlam_prod/data/la_liberte.py",
            sorted(unknown),
        )
    save_pages(pages, target)
    LOGGER.info("%d pages enregistrées dans %s", len(pages), target)
    return pages
