from finlam_prod.data.schema import NO_ARTICLE
from finlam_prod.evaluation.finlam import (
    article_separation,
    bleu_block,
    bleu_sequence,
    bootstrap_ci,
    jaccard_count,
)


def test_article_separation_perfect_and_relabelled():
    gold = [NO_ARTICLE, 1, 1, 2, 2, 3]
    areas = [1, 1, 1, 1, 1, 1]
    perfect = article_separation(gold, [NO_ARTICLE, 9, 9, 7, 7, 5], areas)
    assert perfect["as_f1"] == 1.0 and perfect["as_miou"] == 1.0


def test_article_separation_merge_and_split():
    gold = [1, 1, 2, 2]
    merged = article_separation(gold, [1, 1, 1, 1], [1, 1, 1, 1])
    assert (
        merged["as_precision"] == 1.0 and merged["as_recall"] == 0.5
    )  # un seul apparié à IoU 0,5
    gold3 = [1, 1, 1, 2, 2, 2]
    split = article_separation(gold3, [1, 2, 3, 4, 5, 6], [1] * 6)
    assert split["as_f1"] == 0.0  # chaque singleton a une IoU de 1/3 < seuil
    assert jaccard_count(gold3, [1, 2, 3, 4, 5, 6]) == 2 / 6


def test_bleu_sequence():
    assert abs(bleu_sequence([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) - 1.0) < 1e-9
    assert bleu_sequence([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]) < 0.15  # unigrammes seuls, lissés
    assert 0 < bleu_sequence([0, 1, 2, 3, 4, 5], [0, 1, 2, 5, 3, 4]) < 1
    assert bleu_block([0, 1, 2], [0, 1, 2]) == 1.0


def test_bootstrap_ci_contains_mean():
    per_page = [{"m": v} for v in [0.2, 0.4, 0.6, 0.8]]
    mean, low, high = bootstrap_ci(per_page, n_boot=200)["m"]
    assert abs(mean - 0.5) < 1e-9 and low <= mean <= high


def test_viz_helpers_run_headless(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from finlam_prod.data.synthetic import make_pages
    from finlam_prod.evaluation import viz
    from finlam_prod.pipeline.run import Pipeline

    pages = make_pages(4, seed=3)
    preds = [Pipeline().run(p) for p in pages]
    table = pd.DataFrame({"page_id": [p.id for p in pages], "as_f1": [0.2, 0.4, 0.6, 0.8]})
    viz.plot_metric_distribution({"a": table, "b": table}, "as_f1")
    viz.plot_metric_by(table, pages, "as_f1", "zones")
    matrix = viz.confusion_table(pages, preds)
    assert matrix.values.sum() == sum(len(p.zones) for p in pages)
    viz.plot_confusion(matrix)
    fig = viz.compare_page(pages[0], preds[0])
    fig.savefig(tmp_path / "page.png")
    assert (tmp_path / "page.png").stat().st_size > 0
