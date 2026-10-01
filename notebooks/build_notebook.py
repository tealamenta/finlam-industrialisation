"""Génère notebooks/01_analyse_resultats_la_liberte.ipynb (source lisible, diffable).

    python notebooks/build_notebook.py && jupyter nbconvert --execute --to notebook --inplace \
        notebooks/01_analyse_resultats_la_liberte.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
py = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Analyse des résultats — *La Liberté*, split test

Ce carnet **lit** ce que la chaîne a produit (`finlam-prod evaluate` → `results/*_pages.csv`,
`finlam-prod train` → `models/la_liberte/`) et le met en images. Il n'entraîne rien et ne recalcule
aucune métrique : tout vient de `finlam_prod.evaluation`. Question à laquelle il répond : **où et
pourquoi la doublure se trompe-t-elle**, et qu'est-ce que cela impose au contrôle qualité ?

Prérequis (voir README) :

```bash
finlam-prod download --dataset la-liberte --split test                 # annotations, 433 pages
finlam-prod download --dataset la-liberte --split test --images --limit 40   # 40 images pour la partie 3
finlam-prod train --dataset la-liberte --split validation --models-dir models/la_liberte
finlam-prod evaluate --dataset la-liberte --split test                  # règles
finlam-prod evaluate --dataset la-liberte --split test --models-dir models/la_liberte   # doublure
```
""")

py("""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from finlam_prod.data.schema import load_pages
from finlam_prod.evaluation import viz
from finlam_prod.evaluation.finlam import bootstrap_ci
from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier
from finlam_prod.pipeline.run import Pipeline

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CACHE, RESULTS, MODELS = ROOT / "data/cache", ROOT / "results", ROOT / "models/la_liberte"
plt.rcParams["figure.dpi"] = 110

full = CACHE / "la_liberte_test.jsonl"
pages = load_pages(full if full.exists() else CACHE / "la_liberte_test_n40.jsonl")
image_of = {p.id: CACHE / "images_la_liberte" / f"{p.id}.jpg" for p in pages}
print(f"{len(pages)} pages, {sum(image_of[p.id].exists() for p in pages)} avec image")
""")

md("""
## 1. Les métriques, avec leurs intervalles

Une ligne par page dans `results/eval_la_liberte_test_<système>_pages.csv`. La moyenne et l'IC95 %
(bootstrap sur les pages, comme HIPE-OCRepair 2026) sont recalculés ici par `bootstrap_ci` pour
montrer que la table du README se reproduit.
""")

py("""
tables = viz.load_per_page(RESULTS, "la_liberte_test")
metrics = ["as_f1", "as_miou", "jaccard_article", "bleu_block", "pairwise_f1", "zone_accuracy"]
summary = {}
for system, table in tables.items():
    ci = bootstrap_ci(table[metrics].to_dict("records"), n_boot=500, seed=0)
    summary[system] = {m: "{:.3f} [{:.3f} ; {:.3f}]".format(*ci[m]) for m in metrics}
pd.DataFrame(summary)
""")

py("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
viz.plot_metric_distribution(tables, "as_f1", ax=axes[0])
viz.plot_metric_distribution(tables, "bleu_block", ax=axes[1])
fig.suptitle("Distribution par page : séparation d'articles (AS@0.5 F1) et ordre (BLEU_block)")
fig.tight_layout()
""")

md("""
Lecture : la doublure déplace toute la distribution d'AS F1 vers la droite mais reste loin des
0,80 du pipeline du consortium ; sur l'ordre les deux systèmes sont confondus (même ordonnanceur
XY guidé par les colonnes), la référence à 0,87 est ce que LayoutReader apporte.
""")

md("""
## 2. Où sont les erreurs

Deux angles : la confusion entre classes de zones (le boosting), et la dépendance du F1 à la densité
de la page (le regroupeur).
""")

py("""
pipeline = Pipeline(
    GradientBoostingZoneClassifier.load(MODELS / "zone_classifier.joblib"),
    segmenter=BoundarySegmenter.load(MODELS / "segmenter.joblib"),
)
sample = pages[:120]  # suffisant pour la confusion ; ~10 s
predicted = [pipeline.run(p) for p in sample]
confusion = viz.confusion_table(sample, predicted)
fig, ax = plt.subplots(figsize=(6, 5))
viz.plot_confusion(confusion, ax=ax)
ax.set_title("Classes de zones, normalisées par ligne (vérité terrain)")
confusion
""")

py("""
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, by in zip(axes, ["articles", "zones", "colonnes"]):
    viz.plot_metric_by(tables["doublure"], pages, "as_f1", by, ax=ax)
fig.suptitle("Doublure : AS@0.5 F1 par page selon la densité de la page")
fig.tight_layout()
""")

py("""
worst = tables["doublure"].nsmallest(10, "as_f1")[["page_id", "as_f1", "jaccard_article", "zone_accuracy"]]
worst
""")

md("""
## 3. Regarder les pages

Vérité terrain à gauche, prédiction à droite, une couleur par article (gris : hors article).
Les trois pages : la meilleure, la médiane et la pire parmi celles dont on a l'image.
""")

py("""
scores = tables["doublure"].set_index("page_id")["as_f1"]
with_image = [p for p in pages if image_of[p.id].exists() and p.id in scores.index]
ranked = sorted(with_image, key=lambda p: scores[p.id])
picks = {"pire": ranked[0], "médiane": ranked[len(ranked) // 2], "meilleure": ranked[-1]}
for label, page in picks.items():
    fig = viz.compare_page(page, pipeline.run(page), image_of[page.id], scores[page.id])
    fig.suptitle(f"{label} — page {page.id[:8]}", fontsize=10)
""")

md("""
## 4. Ce qu'on en tire (433 pages)

1. **L'erreur dominante est la sur-segmentation, et c'est en partie une convention d'annotation.**
   Précision AS 0,40 < rappel 0,47 : la doublure prédit plus d'articles qu'il n'y en a. Les pires
   pages (AS F1 = 0 avec exactitude de classes 0,86-1,00) sont celles où la vérité terrain regroupe
   une **rubrique entière** en un seul article — « Échos de Paris » sur deux colonnes, le « Guide
   Paris-Quartier » (une page de petites annonces = un article), un feuilleton — alors que la doublure
   ouvre un article à chaque titre. Sur ces pages la classification est juste et le regroupement est
   faux : le modèle ne manque pas d'yeux, il lui manque la règle. **Question au consortium** : qu'est-ce
   qui ferme un article — un titre, ou un changement de rubrique (`zone_section_ids`) ?
2. **Les confusions restantes sont exactement celles qui décident des frontières.** TEXT est reconnu à
   98 %, mais TITLE à 79 % (14 % pris pour du texte), SUBTITLE à 38 % (21 % pris pour un titre,
   30 % pour du texte) et INSIDEHEADING à 46 %. Un sous-titre pris pour un titre ouvre un faux article ;
   un titre pris pour du texte en fusionne deux. Les 94 % d'exactitude sont portés par TEXT (84 % des
   zones) : la métrique à suivre est le F1 macro (0,67), pas l'exactitude.
3. **La densité qui compte, c'est le nombre de lignes, pas d'articles.** AS F1 par page : r = −0,31
   avec le nombre de zones, r = 0,07 avec le nombre d'articles, r = +0,26 avec le nombre de colonnes
   (les pages à 1-3 colonnes sont des pages spéciales : publicités, guides, pleine page). Règle de QC
   dérivée, sans vérité terrain : **page au-delà de ~280 zones ou à moins de 4 colonnes → relecture**,
   seuils à calibrer sur le 90ᵉ centile de ce jeu.
4. **Ce que ça change pour l'industrialisation.** Rien de tout cela ne se corrige en améliorant la
   doublure : la sur-segmentation demande l'information de rubrique et de continuation que seul un
   modèle avec le texte et l'image apporte — c'est précisément ce que le pipeline du consortium
   (LayoutReader + regroupeur) gagne, +38 points. En revanche la règle de QC du point 3 et la
   convention du point 1 s'appliquent **à ses sorties aussi**, et c'est là que ce carnet sert.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = Path(__file__).with_name("01_analyse_resultats_la_liberte.ipynb")
nbf.write(nb, out)
print(f"écrit {out}")
