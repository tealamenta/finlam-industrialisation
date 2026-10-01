"""Métriques d'évaluation des trois tâches.

- classe de zone : exactitude et F1 macro (les classes rares comptent
  autant que ARTICLE-TEXT, ultra-majoritaire) ;
- ordre de lecture : tau de Kendall (concordance des couples) et taux
  de « successions correctes » (une zone est-elle suivie de la bonne ?),
  ce dernier étant proche du BLEU par blocs utilisé par le consortium ;
- séparation d'articles : F1 sur les couples « même article » et
  indice de Rand ajusté, tous deux indépendants de la numérotation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score

from finlam_prod.data.schema import NO_ARTICLE, Page, ZoneClass


# ------------------------------------------------------------ classe de zone
def classification_scores(gold: list[int], pred: list[int]) -> dict[str, float]:
    """Exactitude, F1 macro et F1 par classe (préfixe ``f1_``)."""
    if not gold:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    gold_a, pred_a = np.array(gold), np.array(pred)
    scores = {
        "accuracy": float((gold_a == pred_a).mean()),
        "macro_f1": float(f1_score(gold_a, pred_a, average="macro", zero_division=0)),
    }
    present = sorted(set(gold) | set(pred))
    per_class = f1_score(gold_a, pred_a, labels=present, average=None, zero_division=0)
    for klass, value in zip(present, per_class, strict=True):
        scores[f"f1_{ZoneClass(klass).label}"] = float(value)
    return scores


# ------------------------------------------------------------ ordre de lecture
def kendall_tau(gold: list[int], pred: list[int]) -> float:
    """Tau de Kendall entre deux classements (1 = identiques, -1 = inversés)."""
    n = len(gold)
    if n < 2:
        return 1.0
    concordant = discordant = 0
    for i, j in combinations(range(n), 2):
        sign = (gold[i] - gold[j]) * (pred[i] - pred[j])
        if sign > 0:
            concordant += 1
        elif sign < 0:
            discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def successor_accuracy(gold: list[int], pred: list[int]) -> float:
    """Part des zones dont le successeur prédit est le successeur attendu."""
    n = len(gold)
    if n < 2:
        return 1.0
    gold_seq = sorted(range(n), key=lambda i: gold[i])
    pred_seq = sorted(range(n), key=lambda i: pred[i])
    gold_next = {a: b for a, b in zip(gold_seq, gold_seq[1:], strict=False)}
    pred_next = {a: b for a, b in zip(pred_seq, pred_seq[1:], strict=False)}
    hits = sum(pred_next.get(a) == b for a, b in gold_next.items())
    return hits / (n - 1)


def intra_article_tau(gold: list[int], pred: list[int], articles: list[int]) -> float:
    """Tau de Kendall moyen *à l'intérieur* de chaque article (vérité terrain).

    Mesure l'ordre des zones d'un article indépendamment de la place de
    l'article dans la page. Les zones hors article sont ignorées.
    """
    groups: dict[int, list[int]] = {}
    for i, art in enumerate(articles):
        if art != NO_ARTICLE:
            groups.setdefault(art, []).append(i)
    taus = [
        kendall_tau([gold[i] for i in members], [pred[i] for i in members])
        for members in groups.values()
        if len(members) > 1
    ]
    return float(np.mean(taus)) if taus else 1.0


def article_sequence_tau(gold: list[int], pred: list[int], articles: list[int]) -> float:
    """Tau de Kendall sur la *séquence des articles* (rang de leur première zone).

    Mesure si les articles sont lus dans le bon ordre, quelle que soit
    l'organisation interne de chacun.
    """
    first_gold: dict[int, int] = {}
    first_pred: dict[int, int] = {}
    for i, art in enumerate(articles):
        if art == NO_ARTICLE:
            continue
        first_gold[art] = min(first_gold.get(art, 10**9), gold[i])
        first_pred[art] = min(first_pred.get(art, 10**9), pred[i])
    keys = list(first_gold)
    return kendall_tau([first_gold[k] for k in keys], [first_pred[k] for k in keys])


def ordering_scores(
    gold: list[int], pred: list[int], articles: list[int] | None = None
) -> dict[str, float]:
    """Métriques d'ordre de lecture d'une page, décomposées par article si fourni."""
    scores = {
        "kendall_tau": kendall_tau(gold, pred),
        "successor_acc": successor_accuracy(gold, pred),
    }
    if articles is not None:
        scores["intra_article_tau"] = intra_article_tau(gold, pred, articles)
        scores["article_seq_tau"] = article_sequence_tau(gold, pred, articles)
    return scores


# ------------------------------------------------------- séparation d'articles
def pairwise_article_f1(gold: list[int], pred: list[int]) -> float:
    """F1 des couples de zones « dans le même article ».

    Les zones hors article (NO_ARTICLE) dans la vérité terrain sont
    ignorées, comme dans l'évaluation du consortium FINLAM.
    """
    kept = [i for i, g in enumerate(gold) if g != NO_ARTICLE]
    if len(kept) < 2:
        return 1.0
    tp = fp = fn = 0
    for i, j in combinations(kept, 2):
        same_gold = gold[i] == gold[j]
        same_pred = pred[i] == pred[j] and pred[i] != NO_ARTICLE
        tp += same_gold and same_pred
        fp += same_pred and not same_gold
        fn += same_gold and not same_pred
    if tp == 0:
        return 0.0 if (fp or fn) else 1.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def article_scores(gold: list[int], pred: list[int]) -> dict[str, float]:
    """F1 par couples et ARI de la séparation d'articles d'une page."""
    kept = [i for i, g in enumerate(gold) if g != NO_ARTICLE]
    ari = (
        adjusted_rand_score([gold[i] for i in kept], [pred[i] for i in kept])
        if len(kept) > 1
        else 1.0
    )
    return {"pairwise_f1": pairwise_article_f1(gold, pred), "ari": float(ari)}


def mean_over_pages(per_page: list[dict[str, float]]) -> dict[str, float]:
    """Moyenne (macro par page) d'une liste de dictionnaires de scores."""
    if not per_page:
        return {}
    keys = sorted({k for scores in per_page for k in scores})
    return {k: float(np.nanmean([s[k] for s in per_page if k in s])) for k in keys}


def gold_classes(page: Page) -> list[int]:  # noqa: D103
    return [int(z.zone_class) if z.zone_class is not None else -1 for z in page.zones]


def gold_order(page: Page) -> list[int]:  # noqa: D103
    return [z.order_key for z in page.zones]


def gold_articles(page: Page) -> list[int]:  # noqa: D103
    return [z.article for z in page.zones]
