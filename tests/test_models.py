from finlam_prod.data.schema import NO_ARTICLE
from finlam_prod.evaluation.metrics import (
    article_scores,
    classification_scores,
    gold_articles,
    gold_classes,
    gold_order,
    kendall_tau,
    pairwise_article_f1,
    successor_accuracy,
)
from finlam_prod.pipeline.features import (
    PAIR_FEATURE_NAMES,
    ZONE_FEATURE_NAMES,
    estimate_columns,
    pair_features,
    zone_features,
)
from finlam_prod.pipeline.ml import (
    BoundarySegmenter,
    GradientBoostingZoneClassifier,
)
from finlam_prod.pipeline.rules import (
    ColumnOrderer,
    HeuristicClassifier,
    TitleRuleSegmenter,
    XYCutOrderer,
    xy_cut,
)


# ------------------------------------------------------------------ features
def test_feature_vector_lengths(simple_page):
    columns = estimate_columns(simple_page)
    assert len(columns) == 2
    assert len(zone_features(simple_page.zones[0], simple_page, columns)) == len(
        ZONE_FEATURE_NAMES
    )
    a, b = simple_page.zones[1], simple_page.zones[2]
    assert len(pair_features(a, b, simple_page, columns)) == len(PAIR_FEATURE_NAMES)


# ------------------------------------------------------------------- metrics
def test_kendall_tau_extremes():
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0
    assert kendall_tau([0], [0]) == 1.0


def test_successor_accuracy():
    assert successor_accuracy([0, 1, 2], [0, 1, 2]) == 1.0
    assert successor_accuracy([0, 1, 2], [0, 2, 1]) == 0.0


def test_pairwise_article_f1_ignores_no_article():
    gold = [NO_ARTICLE, 1, 1, 2, 2]
    assert pairwise_article_f1(gold, [NO_ARTICLE, 7, 7, 9, 9]) == 1.0
    assert pairwise_article_f1(gold, [NO_ARTICLE, 1, 1, 1, 1]) < 1.0
    assert pairwise_article_f1(gold, [5, 1, 2, 3, 4]) == 0.0


def test_classification_scores_keys():
    scores = classification_scores([5, 6, 6], [5, 6, 5])
    assert set(scores) >= {
        "accuracy",
        "macro_f1",
        "f1_ARTICLE-TITLE",
        "f1_ARTICLE-TEXT",
    }
    assert abs(scores["accuracy"] - 2 / 3) < 1e-9


# --------------------------------------------------------------------- rules
def test_xy_cut_reads_title_before_columns(simple_page):
    ordered = [z.id for z in xy_cut(simple_page.zones)]
    assert ordered == ["hdr", "t1", "l1", "l2", "r1", "r2"]


def test_xy_cut_orderer_is_permutation(simple_page):
    ranks = XYCutOrderer().predict(simple_page)
    assert sorted(ranks) == list(range(len(simple_page.zones)))
    assert ranks == gold_order(simple_page)


def test_column_orderer_is_permutation(simple_page):
    ranks = ColumnOrderer().predict(simple_page)
    assert sorted(ranks) == list(range(len(simple_page.zones)))


def test_heuristic_classifier(simple_page):
    pred = HeuristicClassifier().predict(simple_page)
    assert len(pred) == len(simple_page.zones)
    assert classification_scores(gold_classes(simple_page), pred)["accuracy"] >= 0.5


def test_title_rule_segmenter(simple_page):
    articles = TitleRuleSegmenter().predict(
        simple_page, gold_order(simple_page), gold_classes(simple_page)
    )
    assert articles[0] == NO_ARTICLE
    assert article_scores(gold_articles(simple_page), articles)["pairwise_f1"] == 1.0


# ------------------------------------------------------------------------ ml
def test_gradient_boosting_classifier_learns(train_pages, test_pages, tmp_path):
    model = GradientBoostingZoneClassifier().fit(train_pages)
    gold, pred = [], []
    for page in test_pages:
        gold += gold_classes(page)
        pred += model.predict(page)
    assert classification_scores(gold, pred)["accuracy"] > 0.9
    model.save(tmp_path / "m.joblib")
    reloaded = GradientBoostingZoneClassifier.load(tmp_path / "m.joblib")
    assert reloaded.predict(test_pages[0]) == model.predict(test_pages[0])


def test_boundary_segmenter_learns(train_pages, test_pages):
    model = BoundarySegmenter().fit(train_pages)
    for page in test_pages[:3]:
        articles = model.predict(page, gold_order(page), gold_classes(page))
        assert article_scores(gold_articles(page), articles)["pairwise_f1"] > 0.8


def test_empty_page_predictions():
    from finlam_prod.data.schema import Page

    empty = Page(id="e", issue_id="i", index=1)
    assert HeuristicClassifier().predict(empty) == []
    assert XYCutOrderer().predict(empty) == []
    assert TitleRuleSegmenter().predict(empty, [], []) == []


def test_ordering_scores_decomposition():
    from finlam_prod.evaluation.metrics import ordering_scores

    gold = [0, 1, 2, 3]
    articles = [1, 1, 2, 2]
    swapped_articles = [2, 3, 0, 1]  # articles inversés, ordre interne parfait
    scores = ordering_scores(gold, swapped_articles, articles)
    assert scores["intra_article_tau"] == 1.0
    assert scores["article_seq_tau"] == -1.0
    assert scores["kendall_tau"] < 0
