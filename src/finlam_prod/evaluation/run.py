"""Évaluation d'un pipeline (prototype ou doublure) avec les métriques du consortium.

Deux régimes :

- ``evaluate_pipeline`` : on applique un :class:`~finlam_prod.pipeline.run.Pipeline`
  aux pages annotées et on compare ;
- ``evaluate_predictions`` : on compare des pages *déjà prédites* (sorties
  du prototype relues depuis le contrat JSON) aux pages annotées, par
  identifiant de page. C'est le régime de production : le prototype
  tourne ailleurs, on n'évalue que ses sorties.

Les résultats sont rapportés au format de HIPE-OCRepair (ICDAR 2026) :
moyenne et intervalle de confiance à 95 % par bootstrap sur les pages,
ainsi que le coût par page.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from finlam_prod.data.schema import Page
from finlam_prod.evaluation.finlam import bootstrap_ci, finlam_page_scores
from finlam_prod.evaluation.metrics import (
    article_scores,
    classification_scores,
    gold_articles,
    gold_classes,
    gold_order,
    ordering_scores,
)
from finlam_prod.pipeline.run import Pipeline


@dataclass
class EvaluationReport:
    """Résultat d'une évaluation : table par page, résumé avec IC95 %, coût."""

    per_page: pd.DataFrame
    summary: dict[str, tuple[float, float, float]]
    seconds_per_page: float
    n_pages: int
    n_boot: int = 1000

    def table(self) -> pd.DataFrame:
        """Table « métrique, moyenne, IC bas, IC haut » prête à publier."""
        rows = [
            {"metric": k, "mean": m, "ci_low": lo, "ci_high": hi}
            for k, (m, lo, hi) in self.summary.items()
        ]
        rows.append(
            {
                "metric": "seconds_per_page",
                "mean": self.seconds_per_page,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
        )
        return pd.DataFrame(rows)

    def markdown(self) -> str:
        """Tableau Markdown (moyenne ± IC95 %) pour le README ou le rapport."""
        lines = ["| métrique | moyenne | IC95 % |", "|---|---|---|"]
        for k, (m, lo, hi) in self.summary.items():
            lines.append(f"| {k} | {m:.3f} | [{lo:.3f} ; {hi:.3f}] |")
        lines.append(f"| seconds_per_page | {self.seconds_per_page:.3f} | |")
        lines.append(
            f"\n{self.n_pages} pages ; IC95 % par bootstrap ({self.n_boot} tirages de pages)."
        )
        return "\n".join(lines)


def _page_scores(gold: Page, pred: Page) -> dict[str, float]:
    classes = [int(z.zone_class) if z.zone_class is not None else -1 for z in pred.zones]
    order = [int(z.order) for z in pred.zones]
    articles = [int(z.article_id) for z in pred.zones]
    scores: dict[str, float] = {}
    if all(z.zone_class is not None for z in gold.zones):
        cls = classification_scores(gold_classes(gold), classes)
        scores["zone_accuracy"] = cls["accuracy"]
        scores["zone_macro_f1"] = cls["macro_f1"]
    scores.update(finlam_page_scores(gold, articles, order))
    scores["pairwise_f1"] = article_scores(gold_articles(gold), articles)["pairwise_f1"]
    scores.update(
        {
            f"order_{k}": v
            for k, v in ordering_scores(gold_order(gold), order, gold_articles(gold)).items()
        }
    )
    return scores


def evaluate_predictions(
    gold_pages: list[Page],
    predicted_pages: list[Page],
    n_boot: int = 1000,
    seconds_per_page: float = float("nan"),
) -> EvaluationReport:
    """Compare des pages prédites (même ``id`` et mêmes zones) aux pages annotées."""
    gold_by_id = {p.id: p for p in gold_pages}
    rows = []
    for pred in predicted_pages:
        gold = gold_by_id.get(pred.id)
        if gold is None:
            continue
        if len(gold.zones) != len(pred.zones):
            raise ValueError(
                f"Page {pred.id} : {len(pred.zones)} zones prédites, {len(gold.zones)} annotées"
            )
        rows.append({"page_id": pred.id, **_page_scores(gold, pred)})
    per_page = pd.DataFrame(rows)
    scores = [{k: v for k, v in r.items() if k != "page_id"} for r in rows]
    return EvaluationReport(
        per_page, bootstrap_ci(scores, n_boot=n_boot), seconds_per_page, len(rows), n_boot
    )


def evaluate_pipeline(
    pipeline: Pipeline, pages: list[Page], n_boot: int = 1000
) -> EvaluationReport:
    """Applique un pipeline aux pages annotées et évalue ; mesure le coût par page."""
    start = time.perf_counter()
    predicted = [pipeline.run(p) for p in pages]
    seconds = (time.perf_counter() - start) / max(len(pages), 1)
    return evaluate_predictions(pages, predicted, n_boot=n_boot, seconds_per_page=seconds)
