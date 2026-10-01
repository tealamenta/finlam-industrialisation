"""Client minimal pour les API de Gallica (IIIF, pagination, OCR ALTO).

Gallica limite fortement le débit (quelques requêtes ALTO par minute
depuis une même adresse). Le client applique donc un intervalle minimal
entre deux requêtes et réessaie avec attente exponentielle sur HTTP 429.

Références :
- API Document : https://api.bnf.fr/fr/api-document-de-gallica
- IIIF Image : https://gallica.bnf.fr/iiif/ark:/12148/<ark>/f<n>/full/full/0/native.jpg
- OCR ALTO : https://gallica.bnf.fr/RequestDigitalElement?O=<ark>&E=ALTO&Deb=<n>
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from pathlib import Path

import requests
from lxml import etree

from finlam_prod.data.schema import Page, Zone
from finlam_prod.errors import ConversionError, GallicaError

LOGGER = logging.getLogger(__name__)

GALLICA = "https://gallica.bnf.fr"
ARK_PREFIX = "ark:/12148/"
# Gallica refuse le User-Agent python par défaut (403) : chaîne de navigateur, identifiée
USER_AGENT = "Mozilla/5.0 (compatible; finlam-prod/0.1; +https://github.com)"


def normalise_ark(ark: str) -> str:
    """Ramène « ark:/12148/bpt6k…», une URL Gallica ou « bpt6k… » au seul nom ARK."""
    match = re.search(r"(bpt6k[0-9a-z]+|btv1b[0-9a-z]+|cb[0-9a-z]+)", ark)
    if not match:
        raise GallicaError(f"Identifiant ARK non reconnu : {ark!r}")
    return match.group(1)


class GallicaClient:
    """Accès aux images et à l'OCR d'un document Gallica, avec limitation de débit."""

    def __init__(
        self,
        min_interval: float = 3.0,
        max_retries: int = 5,
        timeout: float = 60.0,
        session: requests.Session | None = None,
    ) -> None:
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self._last_call = 0.0

    # ------------------------------------------------------------------ réseau
    def _get(self, url: str, **params: str | int) -> requests.Response:
        """GET avec intervalle minimal et réessais sur 429/5xx."""
        delay = self.min_interval
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = self.session.get(url, params=params or None, timeout=self.timeout)
            except requests.RequestException as exc:  # réseau coupé, DNS, délai dépassé
                LOGGER.warning("%s (essai %d/%d)", exc, attempt, self.max_retries)
                time.sleep(delay)
                delay *= 2
                continue
            if response.status_code == 429 or response.status_code >= 500:
                LOGGER.warning(
                    "HTTP %s sur %s (essai %d/%d), attente %.0fs",
                    response.status_code,
                    url,
                    attempt,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
                continue
            if response.status_code >= 400:  # 4xx : inutile de réessayer, on explique
                hint = {
                    403: "accès refusé (document non diffusable, OCR absent ou blocage du client)",
                    404: "document ou vue introuvable",
                }.get(response.status_code, "requête refusée")
                raise GallicaError(f"HTTP {response.status_code}, {hint} : {response.url}")
            return response
        raise GallicaError(f"Abandon après {self.max_retries} essais : {url}")

    # --------------------------------------------------------------- services
    def pagination(self, ark: str) -> int:
        """Nombre de vues du document (service Pagination)."""
        ark = normalise_ark(ark)
        response = self._get(f"{GALLICA}/services/Pagination", ark=ark)
        root = etree.fromstring(response.content)
        nb = root.findtext(".//nbVueImages")
        if nb is None:
            raise GallicaError(f"Pagination indisponible pour {ark}")
        return int(nb)

    def manifest(self, ark: str) -> dict:
        """Manifeste IIIF (Presentation API) du document."""
        ark = normalise_ark(ark)
        return self._get(f"{GALLICA}/iiif/{ARK_PREFIX}{ark}/manifest.json").json()

    def page_image(self, ark: str, page: int, dest: Path, max_width: int | None = None) -> Path:
        """Télécharge l'image d'une vue via IIIF (pleine taille ou largeur bornée)."""
        ark = normalise_ark(ark)
        size = f"{max_width}," if max_width else "full"
        url = f"{GALLICA}/iiif/{ARK_PREFIX}{ark}/f{page}/full/{size}/0/native.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._get(url).content)
        return dest

    def page_alto(self, ark: str, page: int) -> bytes:
        """OCR ALTO d'une vue (format ALTO BnF)."""
        ark = normalise_ark(ark)
        response = self._get(f"{GALLICA}/RequestDigitalElement", O=ark, E="ALTO", Deb=page)
        if b"<alto" not in response.content[:2000]:
            raise GallicaError(f"Pas d'ALTO pour {ark} vue {page} (document sans OCR ?)")
        return response.content

    def fetch_issue(
        self,
        ark: str,
        dest_dir: Path,
        pages: list[int] | None = None,
        with_images: bool = True,
        max_width: int | None = 2000,
    ) -> list[Page]:
        """Récupère l'ALTO (et les images) d'un fascicule et le convertit en pages."""
        ark = normalise_ark(ark)
        dest_dir = Path(dest_dir) / ark
        dest_dir.mkdir(parents=True, exist_ok=True)
        if pages is None:
            pages = list(range(1, self.pagination(ark) + 1))
        result: list[Page] = []
        for number in pages:
            alto_path = dest_dir / f"f{number}.alto.xml"
            if not alto_path.exists():
                alto_path.write_bytes(self.page_alto(ark, number))
            page = parse_bnf_alto(
                alto_path.read_bytes(),
                page_id=f"{ark}_f{number}",
                issue_id=ark,
                index=number,
            )
            if with_images:
                image_path = dest_dir / f"f{number}.jpg"
                if not image_path.exists():
                    self.page_image(ark, number, image_path, max_width=max_width)
                page.image_path = str(image_path)
            result.append(page)
        return result


# ----------------------------------------------------------------- ALTO BnF
def _local(tag: str) -> str:
    return etree.QName(tag).localname


def _iter(element: etree._Element, name: str) -> Iterator[etree._Element]:
    return (child for child in element.iter() if _local(child.tag) == name)


def parse_bnf_alto(xml: bytes, page_id: str, issue_id: str, index: int) -> Page:
    """Convertit un fichier ALTO (BnF ou standard) en :class:`Page`.

    Chaque ``TextBlock`` et chaque ``Illustration`` devient une zone ; les coordonnées sont
    ramenées en pourcentage de la page à partir des attributs WIDTH et
    HEIGHT de l'élément ``Page``, ce qui rend le résultat indépendant de
    l'unité de mesure déclarée (pixel, mm10, inch1200).

    L'ALTO Gallica ne porte ni classe de zone, ni article : ces champs
    restent ``None`` et sont à prédire par la chaîne.
    """
    root = etree.fromstring(xml)
    page_el = next(_iter(root, "Page"), None)
    if page_el is None:
        raise ConversionError("ALTO sans élément Page")
    page_w = float(page_el.get("WIDTH", 0)) or None
    page_h = float(page_el.get("HEIGHT", 0)) or None
    if not page_w or not page_h:
        raise ConversionError("ALTO sans dimensions de page")

    zones: list[Zone] = []
    blocks = (e for e in root.iter() if _local(e.tag) in ("TextBlock", "Illustration"))
    for order, block in enumerate(blocks):
        hpos, vpos = float(block.get("HPOS", 0)), float(block.get("VPOS", 0))
        width, height = float(block.get("WIDTH", 0)), float(block.get("HEIGHT", 0))
        x0, y0 = 100 * hpos / page_w, 100 * vpos / page_h
        x1, y1 = 100 * (hpos + width) / page_w, 100 * (vpos + height) / page_h
        lines = []
        confidences = []
        for line in _iter(block, "TextLine"):
            words = []
            for string in _iter(line, "String"):
                words.append(string.get("CONTENT", ""))
                if string.get("WC"):
                    confidences.append(float(string.get("WC")))
            lines.append(" ".join(words))
        zones.append(
            Zone(
                id=block.get("ID") or f"{page_id}_b{order}",
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                text="\n".join(lines).strip(),
                confidence=sum(confidences) / len(confidences) if confidences else None,
            )
        )
    return Page(
        id=page_id,
        issue_id=issue_id,
        index=index,
        zones=zones,
        width_px=int(page_w),
        height_px=int(page_h),
        source="gallica",
    )
