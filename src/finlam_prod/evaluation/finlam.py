"""Métriques du consortium FINLAM (Mocaër et al., 2026, arXiv 2607.15082), réimplémentées.

L'article Tiramisu évalue la séparation d'articles et l'ordre de lecture
avec des métriques spécifiques ; on les réimplémente ici pour pouvoir
rapporter des chiffres *comparables* aux leurs (pipeline bottom-up : F1
articles 80,4 %, BLEU_block 87,2 % ; Arcanum : F1 59,4 %).

- **AS@t** (article separation) : appariement hongrois des articles
  prédits et annotés sur leur IoU ; un couple apparié avec IoU ≥ t est
  un vrai positif. On en tire précision, rappel, F1 et mIoU (IoU moyen
  des couples appariés). Ici l'IoU est calculé sur les surfaces des
  zones (les deux partitions portent sur les mêmes zones), ce qui
  équivaut à l'IoU des masques quand les zones ne se recouvrent pas.
- **Jaccard_article** : rapport min/max des nombres d'articles annotés
  et prédits (l'article le nomme ainsi ; c'est une mesure du bon
  *dénombrement* des articles, pas de leur contenu).
- **BLEU_block** : BLEU (1 à 4-grammes, lissage epsilon, pénalité de brièveté) entre la
  séquence des zones dans l'ordre de lecture annoté et dans l'ordre
  prédit, les zones étant identifiées par leur indice ; c'est la métrique
  d'ordre de lecture de LayoutReader.

Implémentations sans dépendance externe hormis scipy (appariement).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from finlam_prod.data.schema import NO_ARTICLE, Page


def _groups(labels: list[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab != NO_ARTICLE and lab is not None:
            groups.setdefault(lab, []).append(i)
    return groups


def article_separation(
    gold: list[int], pred: list[int], areas: list[float], threshold: float = 0.5
) -> dict[str, float]:
    """AS@threshold : précision, rappel, F1 et mIoU des articles appariés.

    Args:
        gold: identifiant d'article annoté par zone (NO_ARTICLE hors article).
        pred: identifiant d'article prédit par zone.
        areas: surface de chaque zone (même unité pour toutes).
        threshold: IoU minimal pour compter un appariement comme correct.

    """
    g, p = _groups(gold), _groups(pred)
    if not g and not p:
        return {"as_precision": 1.0, "as_recall": 1.0, "as_f1": 1.0, "as_miou": 1.0}
    if not g or not p:
        return {"as_precision": 0.0, "as_recall": 0.0, "as_f1": 0.0, "as_miou": 0.0}
    g_keys, p_keys = list(g), list(p)
    iou = np.zeros((len(g_keys), len(p_keys)))
    for a, gk in enumerate(g_keys):
        gset = set(g[gk])
        g_area = sum(areas[i] for i in gset)
        for b, pk in enumerate(p_keys):
            pset = set(p[pk])
            inter = sum(areas[i] for i in gset & pset)
            union = g_area + sum(areas[i] for i in pset) - inter
            iou[a, b] = inter / union if union > 0 else 0.0
    rows, cols = linear_sum_assignment(-iou)
    matched = [iou[r, c] for r, c in zip(rows, cols, strict=True) if iou[r, c] >= threshold]
    tp = len(matched)
    precision = tp / len(p_keys)
    recall = tp / len(g_keys)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "as_precision": precision,
        "as_recall": recall,
        "as_f1": f1,
        "as_miou": float(np.mean(matched)) if matched else 0.0,
    }


def jaccard_count(gold: list[int], pred: list[int]) -> float:
    """Jaccard_article de l'article : min/max des nombres d'articles annotés et prédits."""
    n_g, n_p = len(_groups(gold)), len(_groups(pred))
    if n_g == 0 and n_p == 0:
        return 1.0
    return min(n_g, n_p) / max(n_g, n_p)


def bleu_sequence(reference: list[int], hypothesis: list[int], max_n: int = 4) -> float:
    """BLEU (n-grammes 1..max_n, lissage epsilon, pénalité de brièveté) entre deux séquences.

    Utilisé avec des séquences d'indices de zones : ``reference`` est
    l'ordre annoté, ``hypothesis`` l'ordre prédit.
    """
    if not hypothesis or not reference:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(reference[i : i + n]) for i in range(len(reference) - n + 1))
        hyp_ngrams = Counter(tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1))
        total = sum(hyp_ngrams.values())
        if total == 0:  # séquence plus courte que n : on s'arrête aux ordres disponibles
            break
        overlap = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
        if overlap == 0 and n == 1:
            return 0.0
        # lissage (Chen & Cherry, méthode 1) : un n-gramme absent compte epsilon, pas zéro
        log_precisions.append(math.log(max(overlap, 0.1) / total))
    brevity = (
        1.0
        if len(hypothesis) >= len(reference)
        else math.exp(1 - len(reference) / len(hypothesis))
    )
    return brevity * math.exp(sum(log_precisions) / len(log_precisions))


def bleu_block(gold_order: list[int], pred_order: list[int]) -> float:
    """BLEU_block : BLEU entre la séquence des zones triées par rang annoté et par rang prédit."""
    n = len(gold_order)
    reference = sorted(range(n), key=lambda i: gold_order[i])
    hypothesis = sorted(range(n), key=lambda i: pred_order[i])
    return bleu_sequence(reference, hypothesis)


def finlam_page_scores(
    page: Page,
    pred_articles: list[int],
    pred_order: list[int] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Toutes les métriques FINLAM d'une page, à partir des prédictions."""
    gold_articles = [int(z.article_id) for z in page.zones]
    areas = [max(z.area, 1e-9) for z in page.zones]
    scores = article_separation(gold_articles, pred_articles, areas, threshold)
    scores["jaccard_article"] = jaccard_count(gold_articles, pred_articles)
    if pred_order is not None:
        scores["bleu_block"] = bleu_block([int(z.order) for z in page.zones], pred_order)
    return scores


def bootstrap_ci(
    per_page: list[dict[str, float]],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Moyenne et intervalle de confiance par métrique (bootstrap sur les pages).

    L'article Tiramisu rapporte des estimations ponctuelles ; sur 48 ou
    433 pages, l'incertitude n'est pas négligeable et doit accompagner
    tout chiffre cité.
    """
    rng = np.random.default_rng(seed)
    keys = sorted({k for s in per_page for k in s})
    matrix = np.array([[s.get(k, np.nan) for k in keys] for s in per_page], dtype=float)
    n = len(per_page)
    means = np.nanmean(matrix, axis=0)
    samples = np.empty((n_boot, len(keys)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[b] = np.nanmean(matrix[idx], axis=0)
    low = np.nanpercentile(samples, 100 * alpha / 2, axis=0)
    high = np.nanpercentile(samples, 100 * (1 - alpha / 2), axis=0)
    return {k: (float(means[j]), float(low[j]), float(high[j])) for j, k in enumerate(keys)}
