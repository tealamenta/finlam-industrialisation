"""Conversion d'une page prédite en ALTO 4.4 et validation XSD.

Choix d'encodage :

- ``MeasurementUnit`` = pixel ; les pourcentages sont convertis avec les
  dimensions de l'image (``Page.width_px`` / ``height_px``) ;
- chaque zone devient un ``TextBlock`` (ou une ``Illustration`` pour les
  images) ; le polygone d'origine est conservé dans ``Shape/Polygon`` ;
- la classe de zone est un ``LayoutTag`` référencé par ``TAGREFS`` ;
- l'appartenance à un article est un ``StructureTag`` (``ART_<n>``),
  également référencé par ``TAGREFS`` ; le METS reprend cette
  information dans sa carte de structure logique ;
- l'ordre de lecture est encodé deux fois, par l'ordre des blocs dans
  ``PrintSpace`` et par un ``ReadingOrder/OrderedGroup`` explicite ;
- la confiance de zone, si elle existe, est portée par un ``OtherTag``
  ``CONF_<n>`` ; le texte est découpé en ``TextLine`` (un par ligne de la
  transcription) puis en ``String`` par mot, avec des positions estimées
  par répartition uniforme dans le bloc, ce qui est signalé dans
  ``processingStepDescription``.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from importlib import resources
from pathlib import Path

from lxml import etree

from finlam_prod import __version__
from finlam_prod.data.schema import NO_ARTICLE, Page, Zone, ZoneClass

ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ALTO_SCHEMA_LOCATION = "http://www.loc.gov/standards/alto/v4/alto-4-4.xsd"
NSMAP = {None: ALTO_NS, "xsi": XSI_NS}


def _el(parent: etree._Element, tag: str, text: str | None = None, **attrs: str) -> etree._Element:
    element = etree.SubElement(parent, f"{{{ALTO_NS}}}{tag}", **attrs)
    if text is not None:
        element.text = text
    return element


def _fmt(value: float) -> str:
    return f"{value:.0f}"


def _to_px(zone: Zone, page: Page) -> tuple[float, float, float, float]:
    width = page.width_px or 100
    height = page.height_px or 100
    x0, y0, x1, y1 = zone.bbox
    return (
        x0 * width / 100,
        y0 * height / 100,
        (x1 - x0) * width / 100,
        (y1 - y0) * height / 100,
    )


def _polygon_points(zone: Zone, page: Page) -> str:
    width = page.width_px or 100
    height = page.height_px or 100
    return " ".join(f"{_fmt(x * width / 100)},{_fmt(y * height / 100)}" for x, y in zone.polygon)


def _add_text(block: etree._Element, zone: Zone, page: Page) -> None:
    """Répartit le texte en lignes et mots de positions estimées."""
    hpos, vpos, width, height = _to_px(zone, page)
    lines = [line for line in zone.text.split("\n") if line.strip()] or (
        [zone.text] if zone.text.strip() else []
    )
    if not lines:
        return
    line_h = height / len(lines)
    for li, line in enumerate(lines):
        words = line.split()
        text_line = _el(
            block,
            "TextLine",
            HPOS=_fmt(hpos),
            VPOS=_fmt(vpos + li * line_h),
            WIDTH=_fmt(width),
            HEIGHT=_fmt(line_h),
        )
        total = sum(len(w) for w in words) + max(len(words) - 1, 0)
        cursor = hpos
        for wi, word in enumerate(words):
            w_width = width * len(word) / max(total, 1)
            attrs = {
                "CONTENT": word,
                "HPOS": _fmt(cursor),
                "VPOS": _fmt(vpos + li * line_h),
                "WIDTH": _fmt(w_width),
                "HEIGHT": _fmt(line_h),
            }
            if zone.confidence is not None:
                attrs["WC"] = f"{min(max(zone.confidence, 0.0), 1.0):.2f}"
            _el(text_line, "String", **attrs)
            cursor += w_width
            if wi < len(words) - 1:
                sp_width = width / max(total, 1)
                _el(
                    text_line,
                    "SP",
                    HPOS=_fmt(cursor),
                    VPOS=_fmt(vpos + li * line_h),
                    WIDTH=_fmt(sp_width),
                )
                cursor += sp_width


def page_to_alto(page: Page, image_filename: str | None = None) -> etree._Element:
    """Construit l'arbre ALTO 4.4 d'une page.

    Args:
        page: page portant (au moins) des zones ; classe, ordre et article
            sont optionnels mais attendus en sortie de pipeline.
        image_filename: nom de fichier image à déclarer dans
            ``sourceImageInformation`` (par défaut ``page.image_path``).

    """
    root = etree.Element(f"{{{ALTO_NS}}}alto", nsmap=NSMAP)
    root.set(f"{{{XSI_NS}}}schemaLocation", f"{ALTO_NS} {ALTO_SCHEMA_LOCATION}")

    description = _el(root, "Description")
    _el(description, "MeasurementUnit", "pixel")
    source = _el(description, "sourceImageInformation")
    _el(
        source,
        "fileName",
        image_filename or Path(page.image_path or f"{page.id}.jpg").name,
    )
    processing = _el(description, "Processing", ID="PROC_OLR")
    _el(processing, "processingCategory", "contentGeneration")
    _el(
        processing,
        "processingDateTime",
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    _el(
        processing,
        "processingStepDescription",
        "Classification de zones, ordre de lecture et separation d'articles (finlam-prod). "
        "Positions des TextLine/String estimees par repartition uniforme dans le bloc.",
    )
    software = _el(processing, "processingSoftware")
    _el(software, "softwareName", "finlam-prod")
    _el(software, "softwareVersion", __version__)

    ordered = page.zones_in_order()
    tags = _el(root, "Tags")
    used_classes = sorted({z.zone_class for z in ordered if z.zone_class is not None})
    for klass in used_classes:
        label = ZoneClass(klass).label
        _el(tags, "LayoutTag", ID=f"LT_{label}", LABEL=label, TYPE="zone_class")
    article_ids = sorted({z.article_id for z in ordered if z.article_id not in (None, NO_ARTICLE)})
    for article in article_ids:
        _el(
            tags,
            "StructureTag",
            ID=f"ART_{article}",
            LABEL=f"article {article}",
            TYPE="article",
        )
    conf_tags: dict[str, str] = {}
    for zone in ordered:
        if zone.confidence is not None:
            tag_id = f"CONF_{zone.id}".replace(" ", "_")
            _el(
                tags,
                "OtherTag",
                ID=tag_id,
                LABEL=f"{zone.confidence:.3f}",
                TYPE="confidence",
            )
            conf_tags[zone.id] = tag_id

    reading_order = _el(root, "ReadingOrder")
    group = _el(reading_order, "OrderedGroup", ID="RO_1")

    layout = _el(root, "Layout")
    width, height = page.width_px or 100, page.height_px or 100
    page_el = _el(
        layout,
        "Page",
        ID=f"PAGE_{page.index}",
        PHYSICAL_IMG_NR=str(page.index),
        WIDTH=_fmt(width),
        HEIGHT=_fmt(height),
    )
    print_space = _el(
        page_el,
        "PrintSpace",
        HPOS="0",
        VPOS="0",
        WIDTH=_fmt(width),
        HEIGHT=_fmt(height),
    )

    for position, zone in enumerate(ordered):
        hpos, vpos, w, h = _to_px(zone, page)
        refs = []
        if zone.zone_class is not None:
            refs.append(f"LT_{ZoneClass(zone.zone_class).label}")
        if zone.article_id not in (None, NO_ARTICLE):
            refs.append(f"ART_{zone.article_id}")
        if zone.id in conf_tags:
            refs.append(conf_tags[zone.id])
        attrs = {
            "ID": _block_id(zone),
            "HPOS": _fmt(hpos),
            "VPOS": _fmt(vpos),
            "WIDTH": _fmt(w),
            "HEIGHT": _fmt(h),
        }
        if refs:
            attrs["TAGREFS"] = " ".join(refs)
        is_illustration = zone.zone_class == ZoneClass.ARTICLE_ILLUSTRATION
        block = _el(print_space, "Illustration" if is_illustration else "TextBlock", **attrs)
        shape = _el(block, "Shape")
        _el(shape, "Polygon", POINTS=_polygon_points(zone, page))
        if not is_illustration:
            _add_text(block, zone, page)
        _el(group, "ElementRef", ID=f"RO_1_{position}", REF=attrs["ID"])
    return root


def _block_id(zone: Zone) -> str:
    """Identifiant XML valide (NCName) dérivé de l'identifiant de zone."""
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in zone.id)
    return safe if safe[:1].isalpha() or safe[:1] == "_" else f"B_{safe}"


def write_alto(page: Page, path: str | Path, image_filename: str | None = None) -> Path:
    """Écrit l'ALTO d'une page (UTF-8, indenté)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(page_to_alto(page, image_filename))
    tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return path


@lru_cache(maxsize=1)
def alto_schema() -> etree.XMLSchema:
    """Schéma ALTO 4.4 embarqué (avec import xlink local)."""
    with resources.as_file(resources.files("finlam_prod.schemas") / "alto-4-4.xsd") as xsd:
        return etree.XMLSchema(etree.parse(str(xsd)))


def validate_alto(
    document: etree._Element | etree._ElementTree | bytes | str | Path,
) -> list[str]:
    """Valide un ALTO contre le XSD 4.4 ; renvoie la liste des erreurs (vide = valide)."""
    if isinstance(document, (bytes, str, Path)) and not isinstance(document, etree._Element):
        tree = (
            etree.parse(str(document))
            if isinstance(document, Path)
            else etree.fromstring(document)
        )
    else:
        tree = document
    schema = alto_schema()
    schema.validate(tree)
    return [f"ligne {e.line}: {e.message}" for e in schema.error_log]
