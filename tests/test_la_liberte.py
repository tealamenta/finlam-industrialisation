from finlam_prod.data.la_liberte import CLASS_MAPPING, map_class, row_to_page
from finlam_prod.data.schema import NO_ARTICLE, ZoneClass


def test_map_class_by_name_and_by_index():
    names = ["HEADER-TITLE", "TEXT", "FREEAD", "SECTION-SUBTITLE", "MYSTERY"]
    unknown: set[str] = set()
    assert map_class(1, names, unknown) == (ZoneClass.ARTICLE_TEXT, "TEXT")
    assert map_class("FreeAd", names, unknown) == (ZoneClass.ANNOUNCEMENT, "FreeAd")
    assert map_class(3, names, unknown) == (ZoneClass.SECTION_TITLE, "SECTION-SUBTITLE")
    assert map_class(4, names, unknown) == (None, "MYSTERY") and unknown == {"MYSTERY"}
    assert map_class(6, None, set()) == (
        ZoneClass.ARTICLE_TEXT,
        None,
    )  # sans noms : ordre du paquet
    assert all(isinstance(v, ZoneClass) for v in CLASS_MAPPING.values())


def test_row_to_page_la_liberte_fields():
    row = {
        "page_arkindex_id": "p-1",
        "page_index": 2,
        "newspaper_name": "La Liberté - 1926-03-01",
        "zone_polygons": [
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[0, 6], [10, 6], [10, 20], [0, 20]],
        ],
        "zone_texts": ["TITRE", "corps"],
        "zone_classes": ["TITLE", "TEXT"],
        "zone_orders": [0, 1],
        "zone_article_ids": [3, 3],
        "zone_section_ids": [1, 1],
    }
    unknown: set[str] = set()
    page = row_to_page(row, None, unknown)
    assert page.index == 2 and len(page.zones) == 2 and not unknown
    assert page.zones[0].zone_class == ZoneClass.ARTICLE_TITLE
    assert page.zones[0].raw_class == "TITLE" and page.zones[0].section_id == 1
    assert page.zones[1].article_id == 3
    assert page.articles() and NO_ARTICLE not in page.articles()


def test_polygon_scale_detects_fractions():
    from finlam_prod.data.la_liberte import polygon_scale

    assert polygon_scale([[[0.1, 0.2], [0.9, 0.95]]]) == 100.0
    assert polygon_scale([[[10.0, 20.0], [90.0, 95.0]]]) == 1.0
    assert polygon_scale([]) == 100.0
