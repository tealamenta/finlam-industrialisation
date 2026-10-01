"""Lecture d'un fascicule METS/ALTO « mode article » en :class:`Issue`.

Sert à deux choses :

- relire ce que produit :mod:`finlam_prod.convert` (test aller-retour) ;
- relire les METS/ALTO d'Europeana Newspapers déjà présents dans Gallica
  (jeu « Documents de presse numérisés en mode article », api.bnf.fr),
  pour comparer l'OLR existant à la sortie du nouveau pipeline sur un
  même fascicule, avec les mêmes métriques.

Le lecteur ne présuppose pas de profil précis : il cherche dans le METS
une carte de structure logique (``structMap TYPE="LOGICAL"``), y prend
les ``div`` de type article (``TYPE`` contenant « article », insensible
à la casse) et suit leurs ``area`` / ``fptr`` vers les blocs ALTO
(``BEGIN`` = identifiant de ``TextBlock``, ``FILEID`` → fichier ALTO du
``fileSec``). Les profils diffèrent dans les détails (ALTO 1.4 chez
Europeana Newspapers, attributs supplémentaires) ; ce qui n'est pas
reconnu est ignoré et compté dans ``warnings``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from finlam_prod.data.gallica import parse_bnf_alto
from finlam_prod.data.schema import NO_ARTICLE, Issue, Page

LOGGER = logging.getLogger(__name__)


def _local(tag) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def _iter(element, name: str):
    return (e for e in element.iter() if _local(e.tag) == name)


def _href(element) -> str | None:
    for key, value in element.attrib.items():
        if key.endswith("}href") or key == "href":
            return value
    return None


@dataclass
class MetsAltoIssue:
    """Résultat de lecture : fascicule + avertissements de parsing."""

    issue: Issue
    warnings: list[str] = field(default_factory=list)


def read_mets_alto(mets_path: str | Path, issue_id: str | None = None) -> MetsAltoIssue:
    """Lit un METS et ses ALTO (résolus relativement au METS) en fascicule structuré."""
    mets_path = Path(mets_path)
    root = etree.parse(str(mets_path)).getroot()
    warnings: list[str] = []

    # fileSec : FILEID → chemin ALTO
    files: dict[str, Path] = {}
    for file_el in _iter(root, "file"):
        fid = file_el.get("ID")
        loc = next(_iter(file_el, "FLocat"), None)
        href = _href(loc) if loc is not None else None
        if fid and href:
            files[fid] = (
                (mets_path.parent / href).resolve()
                if not href.startswith(("http://", "https://"))
                else Path(href)
            )

    # ALTO → pages (ordre physique)
    physical = next(
        (sm for sm in _iter(root, "structMap") if (sm.get("TYPE") or "").upper() == "PHYSICAL"),
        None,
    )
    page_files: list[tuple[int, str]] = []
    if physical is not None:
        for index, div in enumerate(
            d for d in _iter(physical, "div") if (d.get("TYPE") or "").lower() == "page"
        ):
            order = int(div.get("ORDER", index + 1))
            for fptr in _iter(div, "fptr"):
                fid = fptr.get("FILEID")
                if fid in files and files[fid].suffix.lower() in (".xml", ".alto"):
                    page_files.append((order, fid))
    if not page_files:
        alto_ids = [
            fid
            for fid, p in files.items()
            if p.suffix.lower() in (".xml", ".alto") and "mets" not in p.name.lower()
        ]
        page_files = [(i + 1, fid) for i, fid in enumerate(alto_ids)]
        warnings.append("structMap PHYSICAL absente : pages déduites du fileSec")

    issue_id = issue_id or root.get("OBJID") or mets_path.stem
    pages: dict[str, Page] = {}
    block_owner: dict[str, tuple[str, int]] = {}
    for order, fid in sorted(page_files):
        path = files[fid]
        if not path.exists():
            warnings.append(f"ALTO introuvable : {path}")
            continue
        page = parse_bnf_alto(
            path.read_bytes(), page_id=f"{issue_id}_f{order}", issue_id=issue_id, index=order
        )
        for position, zone in enumerate(page.zones):
            zone.article_id = NO_ARTICLE
            zone.order = position  # l'ordre des blocs dans le PrintSpace vaut ordre de lecture
        pages[fid] = page
        for i, zone in enumerate(page.zones):
            block_owner[(fid, zone.id)] = (fid, i)

    # structMap LOGICAL : articles → blocs
    logical = next(
        (sm for sm in _iter(root, "structMap") if (sm.get("TYPE") or "").upper() == "LOGICAL"),
        None,
    )
    n_articles = 0
    if logical is None:
        warnings.append("structMap LOGICAL absente : aucun article")
    else:
        for div in _iter(logical, "div"):
            if "article" not in (div.get("TYPE") or "").lower():
                continue
            n_articles += 1
            for area in _iter(div, "area"):
                key = (area.get("FILEID"), area.get("BEGIN"))
                owner = block_owner.get(key)
                if owner is None:
                    warnings.append(f"area non résolue : {key}")
                    continue
                pages[owner[0]].zones[owner[1]].article_id = n_articles
    ordered = [pages[fid] for _, fid in sorted(page_files) if fid in pages]
    LOGGER.info(
        "%s : %d pages, %d articles, %d avertissements",
        mets_path.name,
        len(ordered),
        n_articles,
        len(warnings),
    )
    return MetsAltoIssue(
        Issue(id=issue_id, name=root.get("LABEL") or issue_id, pages=ordered), warnings
    )
