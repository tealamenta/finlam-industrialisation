from finlam_prod.data.schema import (
    NO_ARTICLE,
    Zone,
    ZoneClass,
    group_by_issue,
    load_pages,
    save_pages,
)


def test_zone_geometry():
    zone = Zone("z", [(10, 20), (30, 20), (30, 60), (10, 60)])
    assert zone.bbox == (10, 20, 30, 60)
    assert zone.width == 20 and zone.height == 40
    assert zone.cx == 20 and zone.cy == 40
    assert zone.area == 800


def test_iou():
    a = Zone("a", [(0, 0), (10, 0), (10, 10), (0, 10)])
    b = Zone("b", [(5, 0), (15, 0), (15, 10), (5, 10)])
    c = Zone("c", [(50, 50), (60, 50), (60, 60), (50, 60)])
    assert abs(a.iou(b) - 50 / 150) < 1e-9
    assert a.iou(c) == 0.0
    assert a.iou(a) == 1.0


def test_zone_class_labels():
    assert ZoneClass.ARTICLE_TITLE.label == "ARTICLE-TITLE"
    assert ZoneClass.from_label("ARTICLE-TITLE") is ZoneClass.ARTICLE_TITLE
    assert len(ZoneClass) == 13
    assert ZoneClass(6) is ZoneClass.ARTICLE_TEXT


def test_page_helpers(simple_page):
    ordered = [z.id for z in simple_page.zones_in_order()]
    assert ordered == ["hdr", "t1", "l1", "l2", "r1", "r2"]
    articles = simple_page.articles()
    assert set(articles) == {1, 2}
    assert [z.id for z in articles[1]] == ["t1", "l1", "l2"]
    assert all(z.article_id != NO_ARTICLE for group in articles.values() for z in group)


def test_jsonl_roundtrip(tmp_path, simple_page):
    path = tmp_path / "pages.jsonl"
    save_pages([simple_page], path)
    (loaded,) = load_pages(path)
    assert loaded.id == simple_page.id
    assert [z.id for z in loaded.zones] == [z.id for z in simple_page.zones]
    assert loaded.zones[1].polygon == simple_page.zones[1].polygon
    assert loaded.zones[1].zone_class == ZoneClass.ARTICLE_TITLE


def test_group_by_issue(train_pages):
    issues = group_by_issue(train_pages)
    assert sum(len(i.pages) for i in issues) == len(train_pages)
    for issue in issues:
        assert [p.index for p in issue.pages] == sorted(p.index for p in issue.pages)
