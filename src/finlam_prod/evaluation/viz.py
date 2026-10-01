"""Figures d'analyse des résultats (notebooks et rapport).

Trois vues, toutes construites à partir des objets du paquet et des tables
``*_pages.csv`` écrites par ``finlam-prod evaluate`` :

- distribution d'une métrique par page pour plusieurs systèmes, avec les
  points de référence de l'article ;
- matrice de confusion des classes de zones ;
- une page dessinée, polygones colorés par article (vérité terrain et
  prédiction côte à côte), sur l'image si elle est disponible.

Matplotlib est importé à l'appel : le paquet n'en dépend pas.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finlam_prod.data.schema import NO_ARTICLE, Page, ZoneClass

REFERENCE_SCORES = {  # Mocaër et al. 2026, La Liberté test
    "as_f1": {"pipeline FINLAM": 0.804, "Arcanum": 0.594, "Tiramisu": 0.707},
    "bleu_block": {"pipeline FINLAM": 0.872},
}


def load_per_page(results_dir: str | Path, tag: str) -> dict[str, pd.DataFrame]:
    """Tables par page des systèmes évalués sur un jeu (``eval_<tag>_<système>_pages.csv``)."""
    tables = {}
    for path in sorted(Path(results_dir).glob(f"eval_{tag}_*_pages.csv")):
        system = path.stem[len(f"eval_{tag}_") : -len("_pages")]
        tables[system] = pd.read_csv(path)
    return tables


def plot_metric_distribution(
    tables: dict[str, pd.DataFrame], metric: str, ax: Any = None, bins: int = 25
) -> Any:
    """Histogrammes superposés d'une métrique par page ; références de l'article en pointillés."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    edges = np.linspace(0, 1, bins + 1)
    for system, table in tables.items():
        values = table[metric].dropna()
        label = f"{system} (moy. {values.mean():.3f}, n={len(values)})"
        ax.hist(values, bins=edges, alpha=0.55, label=label)
    for name, score in REFERENCE_SCORES.get(metric, {}).items():
        ax.axvline(score, linestyle="--", linewidth=1, color="black")
        top = ax.get_ylim()[1] * 0.95
        ax.text(score, top, f" {name} {score:.3f}", rotation=90, va="top", fontsize=8)
    ax.set_xlabel(metric)
    ax.set_ylabel("pages")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_metric_by(
    table: pd.DataFrame, pages: list[Page], metric: str, by: str, ax: Any = None
) -> Any:
    """Métrique par page en fonction d'une propriété de la page (articles, zones, colonnes)."""
    import matplotlib.pyplot as plt

    from finlam_prod.pipeline.features import estimate_columns

    ax = ax or plt.gca()
    props = {}
    for page in pages:
        props[page.id] = {
            "articles": len({z.article_id for z in page.zones if z.article_id != NO_ARTICLE}),
            "zones": len(page.zones),
            "colonnes": len(estimate_columns(page)),
        }
    frame = table.set_index("page_id").join(pd.DataFrame(props).T, how="inner")
    ax.scatter(frame[by], frame[metric], s=12, alpha=0.6)
    if frame[by].nunique() > 1:
        slope, intercept = np.polyfit(frame[by], frame[metric], 1)
        xs = np.linspace(frame[by].min(), frame[by].max(), 2)
        corr = frame[[by, metric]].corr().iloc[0, 1]
        ax.plot(xs, slope * xs + intercept, color="black", linewidth=1, label=f"r = {corr:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel(f"{by} par page")
    ax.set_ylabel(metric)
    return ax


def confusion_table(gold_pages: list[Page], pred_pages: list[Page]) -> pd.DataFrame:
    """Matrice de confusion des classes de zones (lignes : vérité terrain, colonnes : prédit)."""
    names = [c.name for c in ZoneClass]
    counts: Counter[tuple[str, str]] = Counter()
    for gold, pred in zip(gold_pages, pred_pages, strict=True):
        for g, p in zip(gold.zones, pred.zones, strict=True):
            if g.zone_class is None or p.zone_class is None:
                continue
            counts[(ZoneClass(g.zone_class).name, ZoneClass(p.zone_class).name)] += 1
    matrix = pd.DataFrame(
        [[counts[(row, col)] for col in names] for row in names],
        index=names,
        columns=names,
        dtype=int,
    )
    present = matrix.index[(matrix.sum(axis=1) > 0) | (matrix.sum(axis=0) > 0)]
    return matrix.loc[present, present]


def plot_confusion(matrix: pd.DataFrame, ax: Any = None, normalise: bool = True) -> Any:
    """Affiche la matrice de confusion (normalisée par ligne si demandé)."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    shown = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) if normalise else matrix
    ax.imshow(shown.values, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(len(shown.columns)), shown.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(shown.index)), shown.index, fontsize=7)
    for i in range(len(shown.index)):
        for j in range(len(shown.columns)):
            value = shown.values[i, j]
            if value > 0:
                text = f"{value:.2f}" if normalise else str(int(value))
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > 0.5 else "black",
                )
    ax.set_xlabel("prédit")
    ax.set_ylabel("vérité terrain")
    return ax


def draw_articles(
    page: Page, ax: Any = None, image_path: str | Path | None = None, title: str | None = None
) -> Any:
    """Dessine les zones d'une page colorées par article (gris : hors article)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    ax = ax or plt.gca()
    path = image_path or page.image_path
    aspect = 1.4  # hauteur / largeur d'une page de journal, à défaut d'image
    if path and Path(path).exists():
        from PIL import Image

        image = Image.open(path)
        width, height = image.size
        aspect = height / width
        ax.imshow(image, cmap="gray", extent=(0, 100, 100, 0))
    else:
        ax.set_facecolor("white")
    ids = sorted({z.article for z in page.zones if z.article != NO_ARTICLE})
    cmap = plt.get_cmap("tab20")
    colours = {aid: cmap(i % 20) for i, aid in enumerate(ids)}
    for zone in page.zones:
        colour = colours.get(zone.article, (0.5, 0.5, 0.5, 1.0))
        ax.add_patch(
            Polygon(
                zone.polygon,
                closed=True,
                facecolor=(*colour[:3], 0.35),
                edgecolor=colour,
                linewidth=0.6,
            )
        )
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect(aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"{page.id[:8]} — {len(ids)} articles", fontsize=9)
    return ax


def compare_page(
    gold: Page, pred: Page, image_path: str | Path | None = None, score: float | None = None
) -> Any:
    """Figure à deux panneaux : vérité terrain et prédiction d'une même page."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 7))
    draw_articles(gold, axes[0], image_path, title="vérité terrain")
    suffix = f" (AS F1 {score:.2f})" if score is not None else ""
    draw_articles(pred, axes[1], image_path, title=f"prédiction{suffix}")
    fig.suptitle(f"page {gold.id[:8]}", fontsize=10)
    fig.tight_layout()
    return fig
