from finlam_prod.convert.alto import write_alto
from finlam_prod.convert.mets import write_mets
from finlam_prod.data.mets_alto import read_mets_alto
from finlam_prod.data.schema import NO_ARTICLE, group_by_issue
from finlam_prod.evaluation.finlam import article_separation
from finlam_prod.pipeline.run import Pipeline


def test_mets_alto_roundtrip(tmp_path, test_pages):
    """Ce que le convertisseur écrit, le lecteur le relit avec les mêmes articles."""
    issue = Pipeline().run_issue(group_by_issue(test_pages)[0])
    for page in issue.pages:
        write_alto(page, tmp_path / f"f{page.index}.alto.xml")
    write_mets(issue, tmp_path / "mets.xml")

    result = read_mets_alto(tmp_path / "mets.xml")
    assert result.warnings == []
    assert len(result.issue.pages) == len(issue.pages)
    for original, back in zip(issue.pages, result.issue.pages, strict=True):
        assert len(back.zones) == len(original.zones)
        gold = [z.article_id for z in original.zones_in_order()]
        pred = [
            z.article_id for z in back.zones
        ]  # lus dans l'ordre du PrintSpace = ordre de lecture
        areas = [max(z.area, 1e-9) for z in original.zones_in_order()]
        scores = article_separation(gold, pred, areas)
        assert scores["as_f1"] == 1.0
        assert all((g == NO_ARTICLE) == (p == NO_ARTICLE) for g, p in zip(gold, pred, strict=True))
