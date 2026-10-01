"""Conversion d'un fascicule prédit en METS et validation XSD.

Le METS produit suit le profil courant des projets de presse (Europeana
Newspapers, chaîne OLR de la BnF) :

- ``fileSec`` : un groupe ``IMAGE`` et un groupe ``ALTO`` (un fichier par page) ;
- ``structMap TYPE="PHYSICAL"`` : la suite des pages, chacune pointant
  vers son image et son ALTO ;
- ``structMap TYPE="LOGICAL"`` : le fascicule, puis un ``div
  TYPE="article"`` par article, dont les ``area`` pointent vers les
  blocs ALTO (``BETYPE="IDREF"``), dans l'ordre de lecture et à travers
  les pages ;
- ``structLink`` : relie chaque article aux pages qu'il occupe.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache
from importlib import resources
from pathlib import Path

from lxml import etree

from finlam_prod import __version__
from finlam_prod.convert.alto import _block_id
from finlam_prod.data.schema import NO_ARTICLE, Issue, Page, Zone, ZoneClass
from finlam_prod.errors import ConversionError

LOGGER = logging.getLogger(__name__)

METS_NS = "http://www.loc.gov/METS/"
XLINK_NS = "http://www.w3.org/1999/xlink"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
DC_NS = "http://purl.org/dc/elements/1.1/"
METS_SCHEMA_LOCATION = "http://www.loc.gov/standards/mets/mets.xsd"
NSMAP = {None: METS_NS, "xlink": XLINK_NS, "xsi": XSI_NS, "dc": DC_NS}


def _el(parent: etree._Element, tag: str, text: str | None = None, **attrs: str) -> etree._Element:
    element = etree.SubElement(parent, f"{{{METS_NS}}}{tag}", **attrs)
    if text is not None:
        element.text = text
    return element


def default_alto_name(page: Page) -> str:  # noqa: D103
    return f"f{page.index}.alto.xml"


def default_image_name(page: Page) -> str:  # noqa: D103
    return Path(page.image_path).name if page.image_path else f"f{page.index}.jpg"


def article_title(zones: list[Zone]) -> str:
    """Titre d'un article : sa première zone de type titre, sinon ses premiers mots."""
    for zone in zones:
        if (
            zone.zone_class in (ZoneClass.ARTICLE_TITLE, ZoneClass.SECTION_TITLE)
            and zone.text.strip()
        ):
            return " ".join(zone.text.split())[:120]
    for zone in zones:
        if zone.text.strip():
            return " ".join(zone.text.split())[:60] + "…"
    return "(sans titre)"


def issue_to_mets(
    issue: Issue,
    alto_names: dict[str, str] | None = None,
    image_names: dict[str, str] | None = None,
) -> etree._Element:
    """Construit l'arbre METS d'un fascicule prédit.

    Args:
        issue: fascicule dont les pages portent classes, ordre et articles.
        alto_names: nom de fichier ALTO par ``page.id`` (défaut ``f<n>.alto.xml``).
        image_names: nom de fichier image par ``page.id``.

    """
    alto_names = alto_names or {}
    image_names = image_names or {}
    root = etree.Element(
        f"{{{METS_NS}}}mets",
        nsmap=NSMAP,
        OBJID=issue.id,
        LABEL=issue.name,
        TYPE="newspaper_issue",
    )
    root.set(f"{{{XSI_NS}}}schemaLocation", f"{METS_NS} {METS_SCHEMA_LOCATION}")

    header = _el(
        root,
        "metsHdr",
        CREATEDATE=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    agent = _el(header, "agent", ROLE="CREATOR", TYPE="OTHER", OTHERTYPE="SOFTWARE")
    _el(agent, "name", f"finlam-prod {__version__}")

    dmd = _el(root, "dmdSec", ID="DMD_ISSUE")
    wrap = _el(dmd, "mdWrap", MDTYPE="DC")
    xml_data = _el(wrap, "xmlData")
    etree.SubElement(xml_data, f"{{{DC_NS}}}title").text = issue.name
    etree.SubElement(xml_data, f"{{{DC_NS}}}identifier").text = issue.id

    file_sec = _el(root, "fileSec")
    image_grp = _el(file_sec, "fileGrp", USE="IMAGE")
    alto_grp = _el(file_sec, "fileGrp", USE="ALTO")
    for page in issue.pages:
        img = _el(image_grp, "file", ID=f"IMG_{page.index}", MIMETYPE="image/jpeg")
        loc = _el(img, "FLocat", LOCTYPE="URL")
        loc.set(f"{{{XLINK_NS}}}href", image_names.get(page.id, default_image_name(page)))
        alto = _el(alto_grp, "file", ID=f"ALTO_{page.index}", MIMETYPE="text/xml")
        loc = _el(alto, "FLocat", LOCTYPE="URL")
        loc.set(f"{{{XLINK_NS}}}href", alto_names.get(page.id, default_alto_name(page)))

    physical = _el(root, "structMap", TYPE="PHYSICAL")
    phys_issue = _el(physical, "div", ID="PHYS_ISSUE", TYPE="issue")
    for page in issue.pages:
        div = _el(
            phys_issue,
            "div",
            ID=f"PHYS_{page.index}",
            TYPE="page",
            ORDER=str(page.index),
        )
        _el(div, "fptr", FILEID=f"IMG_{page.index}")
        _el(div, "fptr", FILEID=f"ALTO_{page.index}")

    logical = _el(root, "structMap", TYPE="LOGICAL")
    log_issue = _el(
        logical,
        "div",
        ID="LOG_ISSUE",
        TYPE="issue",
        DMDID="DMD_ISSUE",
        LABEL=issue.name,
    )
    articles: dict[int, list[tuple[Page, Zone]]] = {}
    for page in issue.pages:
        for zone in page.zones_in_order():
            if zone.article == NO_ARTICLE:
                continue
            articles.setdefault(zone.article, []).append((page, zone))
    links: list[tuple[str, str]] = []
    for order, (article_id, members) in enumerate(sorted(articles.items()), start=1):
        div = _el(
            log_issue,
            "div",
            ID=f"LOG_ART_{article_id}",
            TYPE="article",
            ORDER=str(order),
            LABEL=article_title([z for _, z in members]),
        )
        pages_seen: list[int] = []
        for page, zone in members:
            fptr = _el(div, "fptr")
            _el(
                fptr,
                "area",
                FILEID=f"ALTO_{page.index}",
                BETYPE="IDREF",
                BEGIN=_block_id(zone),
            )
            if page.index not in pages_seen:
                pages_seen.append(page.index)
        links.extend((f"LOG_ART_{article_id}", f"PHYS_{index}") for index in pages_seen)

    if links:
        struct_link = _el(root, "structLink")
        for source, target in links:
            link = _el(struct_link, "smLink")
            link.set(f"{{{XLINK_NS}}}from", source)
            link.set(f"{{{XLINK_NS}}}to", target)
    return root


def write_mets(
    issue: Issue,
    path: str | Path,
    alto_names: dict[str, str] | None = None,
    image_names: dict[str, str] | None = None,
) -> Path:
    """Écrit le METS d'un fascicule."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        etree.ElementTree(issue_to_mets(issue, alto_names, image_names)).write(
            str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True
        )
    except (OSError, etree.LxmlError) as exc:
        raise ConversionError(
            f"METS du fascicule {issue.id} non écrit dans {path} : {exc}"
        ) from exc
    LOGGER.info("METS écrit : %s (%d pages)", path, len(issue.pages), extra={"issue_id": issue.id})
    return path


@lru_cache(maxsize=1)
def mets_schema() -> etree.XMLSchema:
    """Schéma METS 1.x embarqué (avec import xlink local)."""
    with resources.as_file(resources.files("finlam_prod.schemas") / "mets.xsd") as xsd:
        return etree.XMLSchema(etree.parse(str(xsd)))


def validate_mets(document: etree._Element | bytes | str | Path) -> list[str]:
    """Valide un METS contre le XSD ; renvoie la liste des erreurs (vide = valide)."""
    if isinstance(document, Path):
        tree = etree.parse(str(document))
    elif isinstance(document, (bytes, str)):
        tree = etree.fromstring(document)
    else:
        tree = document
    schema = mets_schema()
    schema.validate(tree)
    return [f"ligne {e.line}: {e.message}" for e in schema.error_log]
