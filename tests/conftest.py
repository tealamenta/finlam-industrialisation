import pytest

from finlam_prod.data.schema import NO_ARTICLE, Page, Zone, ZoneClass
from finlam_prod.data.synthetic import make_pages


def _box(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def simple_page() -> Page:
    """Page à deux colonnes avec un titre pleine largeur, annotée à la main.

    Ordre attendu : en-tête, titre de une, colonne gauche (haut, bas),
    colonne droite (haut, bas).
    """
    zones = [
        Zone(
            "hdr",
            _box(10, 1, 90, 5),
            "LE JOURNAL",
            ZoneClass.HEADER_TITLE,
            0,
            NO_ARTICLE,
        ),
        Zone("t1", _box(5, 8, 95, 11), "GRAND TITRE", ZoneClass.ARTICLE_TITLE, 1, 1),
        Zone(
            "l1",
            _box(5, 12, 48, 40),
            "Texte gauche haut.\nSuite.",
            ZoneClass.ARTICLE_TEXT,
            2,
            1,
        ),
        Zone("l2", _box(5, 41, 48, 90), "Texte gauche bas.", ZoneClass.ARTICLE_TEXT, 3, 1),
        Zone("r1", _box(52, 12, 95, 15), "AUTRE TITRE", ZoneClass.ARTICLE_TITLE, 4, 2),
        Zone("r2", _box(52, 16, 95, 90), "Texte droite.", ZoneClass.ARTICLE_TEXT, 5, 2),
    ]
    return Page(id="p1", issue_id="issue-1", index=1, zones=zones, width_px=1000, height_px=1500)


@pytest.fixture(scope="session")
def train_pages() -> list[Page]:
    return make_pages(24, seed=1)


@pytest.fixture(scope="session")
def test_pages() -> list[Page]:
    return make_pages(6, seed=2)
