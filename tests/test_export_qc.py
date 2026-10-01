from lxml import etree

from finlam_prod.convert.alto import ALTO_NS, page_to_alto, validate_alto, write_alto
from finlam_prod.convert.mets import METS_NS, issue_to_mets, validate_mets, write_mets
from finlam_prod.data.gallica import normalise_ark, parse_bnf_alto
from finlam_prod.data.schema import NO_ARTICLE, Issue, Page, Zone, ZoneClass, group_by_issue
from finlam_prod.pipeline.run import (
    Pipeline,
    page_from_json,
    page_to_json,
    read_issue_json,
    write_issue_json,
)
from finlam_prod.qc.checks import (
    QCConfig,
    check_issue,
    check_page,
    findings_table,
    summarise,
)

NS = {"a": ALTO_NS, "m": METS_NS}


# ---------------------------------------------------------------- pipeline
def test_pipeline_fills_predictions(simple_page):
    predicted = Pipeline().run(simple_page)
    assert all(
        z.zone_class is not None and z.order is not None and z.article_id is not None
        for z in predicted.zones
    )
    assert simple_page.zones[0].article_id == NO_ARTICLE  # l'entrée n'est pas modifiée
    assert sorted(z.order for z in predicted.zones) == list(range(len(predicted.zones)))


def test_pipeline_issue_numbers_articles_across_pages(test_pages):
    issue = group_by_issue(test_pages)[0]
    predicted = Pipeline().run_issue(issue)
    ids_per_page = [
        {z.article_id for z in p.zones if z.article_id != NO_ARTICLE} for p in predicted.pages
    ]
    for i in range(len(ids_per_page) - 1):
        assert not (ids_per_page[i] & ids_per_page[i + 1])


def test_json_roundtrip(tmp_path, simple_page):
    payload = page_to_json(simple_page)
    assert payload["zones"][0]["class"] == "HEADER-TITLE"
    back = page_from_json(payload)
    assert [z.id for z in back.zones] == [z.id for z in simple_page.zones_in_order()]
    assert back.zones[1].zone_class == ZoneClass.ARTICLE_TITLE
    issue = Issue(id="issue-1", name="Test", pages=[simple_page])
    path = write_issue_json(issue, tmp_path / "issue.json")
    assert read_issue_json(path).pages[0].id == "p1"


# --------------------------------------------------------------------- ALTO
def test_alto_is_valid_and_ordered(simple_page):
    alto = page_to_alto(simple_page)
    assert validate_alto(alto) == []
    blocks = alto.xpath("//a:PrintSpace/*", namespaces=NS)
    assert [b.get("ID") for b in blocks] == ["hdr", "t1", "l1", "l2", "r1", "r2"]
    refs = alto.xpath("//a:ReadingOrder//a:ElementRef/@REF", namespaces=NS)
    assert refs == ["hdr", "t1", "l1", "l2", "r1", "r2"]
    title = alto.xpath("//a:TextBlock[@ID='t1']", namespaces=NS)[0]
    assert "LT_ARTICLE-TITLE" in title.get("TAGREFS") and "ART_1" in title.get("TAGREFS")
    assert title.get("HPOS") == "50" and title.get("WIDTH") == "900"  # 5 % et 90 % de 1000 px
    assert alto.xpath("count(//a:TextBlock[@ID='l1']/a:TextLine)", namespaces=NS) == 2


def test_alto_illustration_and_confidence(simple_page):
    simple_page.zones.append(
        Zone(
            "img",
            [(5, 91), (48, 91), (48, 96), (5, 96)],
            "",
            ZoneClass.ARTICLE_ILLUSTRATION,
            6,
            1,
            confidence=0.42,
        )
    )
    alto = page_to_alto(simple_page)
    assert validate_alto(alto) == []
    assert alto.xpath("//a:Illustration[@ID='img']", namespaces=NS)
    assert alto.xpath("//a:OtherTag[@TYPE='confidence']/@LABEL", namespaces=NS) == ["0.420"]


def test_alto_invalid_document_reports_errors():
    bad = etree.fromstring(f'<alto xmlns="{ALTO_NS}"><Layout/></alto>')
    assert validate_alto(bad)


def test_write_alto_file(tmp_path, simple_page):
    path = write_alto(simple_page, tmp_path / "f1.alto.xml")
    assert validate_alto(path) == []


# --------------------------------------------------------------------- METS
def test_mets_is_valid_with_logical_articles(test_pages, tmp_path):
    issue = Pipeline().run_issue(group_by_issue(test_pages)[0])
    mets = issue_to_mets(issue)
    assert validate_mets(mets) == []
    articles = mets.xpath("//m:structMap[@TYPE='LOGICAL']//m:div[@TYPE='article']", namespaces=NS)
    assert len(articles) == len(
        {z.article_id for p in issue.pages for z in p.zones if z.article_id != NO_ARTICLE}
    )
    areas = mets.xpath("//m:area/@BEGIN", namespaces=NS)
    block_ids = {z.id for p in issue.pages for z in p.zones}
    assert set(areas) <= block_ids
    pages = mets.xpath("//m:structMap[@TYPE='PHYSICAL']//m:div[@TYPE='page']", namespaces=NS)
    assert len(pages) == len(issue.pages)
    assert mets.xpath("//m:smLink", namespaces=NS)
    assert validate_mets(write_mets(issue, tmp_path / "mets.xml")) == []


# ----------------------------------------------------------------------- QC
def test_qc_clean_page_has_no_findings(simple_page):
    assert check_page(simple_page) == []


def test_qc_detects_anomalies(simple_page):
    simple_page.zones.append(
        Zone(
            "bad",
            [(90, 95), (120, 95), (120, 99), (90, 99)],
            "",
            ZoneClass.ARTICLE_TEXT,
            6,
            2,
        )
    )
    simple_page.zones.append(
        Zone(
            "dup",
            [(5, 12), (48, 12), (48, 40), (5, 40)],
            "x",
            ZoneClass.ARTICLE_TEXT,
            7,
            1,
            0.1,
        )
    )
    codes = {f.code for f in check_page(simple_page)}
    assert {"out_of_page", "empty_text", "overlap", "low_confidence"} <= codes
    table = findings_table(check_page(simple_page))
    assert set(table.columns) == {"severity", "code", "page_id", "zone_id", "message"}
    summary = summarise(check_page(simple_page), 1)
    assert summary["errors"] >= 1 and summary["pages_with_errors"] == 1


def test_qc_missing_predictions(simple_page):
    for zone in simple_page.zones:
        zone.order = None
    codes = {f.code for f in check_page(simple_page, QCConfig())}
    assert "missing_order" in codes


def test_qc_issue_non_contiguous_article(test_pages):
    issue = group_by_issue(test_pages)[0]
    issue.pages[0].zones[-1].article_id = 999
    issue.pages[-1].zones[-1].article_id = 999
    codes = {f.code for f in check_issue(issue)}
    assert "article_non_contiguous_pages" in codes


# ------------------------------------------------------------------ Gallica
BNF_ALTO = b"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://bibnum.bnf.fr/ns/alto_prod">
  <Description><MeasurementUnit>mm10</MeasurementUnit></Description>
  <Layout>
    <Page ID="PAG_1" WIDTH="2000" HEIGHT="3000">
      <PrintSpace>
        <TextBlock ID="TB_1" HPOS="100" VPOS="150" WIDTH="800" HEIGHT="60">
          <TextLine>
            <String CONTENT="LE" WC="0.9"/><SP/><String CONTENT="TITRE" WC="0.7"/>
          </TextLine>
        </TextBlock>
        <TextBlock ID="TB_2" HPOS="100" VPOS="300" WIDTH="800" HEIGHT="900">
          <TextLine><String CONTENT="Premier" WC="0.5"/></TextLine>
          <TextLine><String CONTENT="second"/></TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""


def test_parse_bnf_alto():
    page = parse_bnf_alto(BNF_ALTO, page_id="x_f1", issue_id="x", index=1)
    assert page.width_px == 2000 and page.height_px == 3000
    assert [z.id for z in page.zones] == ["TB_1", "TB_2"]
    assert page.zones[0].text == "LE TITRE"
    assert page.zones[0].bbox == (5.0, 5.0, 45.0, 7.0)
    assert abs(page.zones[0].confidence - 0.8) < 1e-9
    assert page.zones[1].text == "Premier\nsecond"
    assert page.zones[0].zone_class is None


def test_normalise_ark():
    assert (
        normalise_ark("https://gallica.bnf.fr/ark:/12148/bpt6k7155522/f3.item") == "bpt6k7155522"
    )
    assert normalise_ark("bpt6k7155522") == "bpt6k7155522"


def test_qc_literature_rules(simple_page):
    from finlam_prod.qc.checks import check_tables, check_text_quality

    assert check_tables(simple_page, QCConfig()) == []
    simple_page.zones.append(
        Zone("tab", [(5, 50), (95, 50), (95, 97), (5, 97)], "1 2 3", ZoneClass.ARTICLE_TABLE, 6, 2)
    )
    for z in simple_page.zones[:-1]:  # réduire les autres zones pour que le tableau domine
        z.polygon = [(x, y * 0.3) for x, y in z.polygon]
    assert [f.code for f in check_tables(simple_page, QCConfig())] == ["table_heavy_page"]

    noisy = Page(id="n", issue_id="i", index=1)
    noisy.zones.append(
        Zone(
            "z",
            [(0, 0), (50, 0), (50, 50), (0, 50)],
            " ".join(["l3s", "m0ts", "s0nt", "f4ux"] * 10),
            ZoneClass.ARTICLE_TEXT,
            0,
            1,
        )
    )
    assert [f.code for f in check_text_quality(noisy, QCConfig())] == ["low_text_quality"]
    assert check_text_quality(simple_page, QCConfig()) == []
