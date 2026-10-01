"""Génère notebooks/00_cadre_evaluation.ipynb : reproduire le cadre d'évaluation de l'article.

    python notebooks/build_notebook_00.py && jupyter nbconvert --execute --to notebook --inplace \
        notebooks/00_cadre_evaluation.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
py = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Reproduire le cadre d'évaluation de l'article FINLAM

Mocaër et al., *Tiramisu…*, HIP'26 @ ICDAR 2026 (arXiv 2607.15082) rapportent, sur le split test de
*La Liberté*, un F1 de séparation d'articles de **80,4 %** pour leur pipeline modulaire, 70,7 % pour
Tiramisu (bout-en-bout) et 59,4 % pour Arcanum. Leurs modèles et leur dépôt d'évaluation ne sont pas
publics : **on ne reproduit pas leurs chiffres, on reproduit leur cadre** — même jeu, même split,
mêmes métriques, même granularité — et on vérifie que ce qu'on mesure se comporte comme ce qu'ils
décrivent. Le jour où leurs sorties sont disponibles, `finlam-prod evaluate --predictions` doit
redonner leurs nombres ; c'est le test final de ce carnet.

Ce carnet tourne sans modèle et sans image : uniquement `data/cache/la_liberte_test.jsonl`
(`finlam-prod download --dataset la-liberte --split test`).
""")

py("""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from finlam_prod.data.schema import NO_ARTICLE, ZoneClass, load_pages
from finlam_prod.evaluation.finlam import (
    article_separation,
    bleu_block,
    bootstrap_ci,
    finlam_page_scores,
    jaccard_count,
)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CACHE = ROOT / "data/cache"
full = CACHE / "la_liberte_test.jsonl"
pages = load_pages(full if full.exists() else CACHE / "la_liberte_test_n40.jsonl")
print(len(pages), "pages")
""")

md("""
## 1. Le jeu : même split, même granularité

L'article annonce 8 836 pages de *La Liberté* (1925-1928), splits train 7 957 / val 446 / test 433.
On vérifie le test, et on regarde ce que sont les « zones » : leur hauteur médiane dit si ce sont
des blocs ou des lignes.
""")

py("""
heights = np.array([z.height for p in pages for z in p.zones])
widths = np.array([z.width for p in pages for z in p.zones])
n_articles = [len({z.article_id for z in p.zones if z.article_id != NO_ARTICLE}) for p in pages]
pd.Series({
    "pages": len(pages),
    "zones": int(sum(len(p.zones) for p in pages)),
    "zones / page (moy.)": round(np.mean([len(p.zones) for p in pages]), 1),
    "articles / page (moy.)": round(np.mean(n_articles), 1),
    "hauteur médiane d'une zone (% page)": round(float(np.median(heights)), 2),
    "largeur médiane d'une zone (% page)": round(float(np.median(widths)), 2),
}, name="La Liberté, test")
""")

md("""
Une zone fait ~1 % de la hauteur de page : ce sont des **lignes de texte**, regroupées en articles.
Les métriques d'articles de l'article sont donc calculées sur des lignes ; un F1 de 80 % s'y lit
« 80 % des articles retrouvés à IoU ≥ 0,5 sur l'union de leurs lignes ».
""")

py("""
classes = json.loads((CACHE / "la_liberte_classes.json").read_text())
raw = Counter(z.raw_class for p in pages for z in p.zones)
mapping = pd.DataFrame(
    [(name, classes["mapping"].get(name, "—"), raw.get(name, 0)) for name in classes["class_names"]],
    columns=["classe La Liberté (16)", "classe du paquet (13, taxonomie multi-titres)", "zones dans le test"],
)
mapping
""")

md("""
## 2. Les trois métriques, telles que l'article les définit

- **AS@0.5** (*article separation*) : on apparie articles prédits et annotés par l'algorithme
  hongrois sur leur IoU ; un couple à IoU ≥ 0,5 est un vrai positif ; précision, rappel, F1, mIoU.
- **Jaccard article** : rapport min/max des nombres d'articles annotés et prédits (dénombrement).
- **BLEU_block** : BLEU 1-4-grammes entre la séquence des zones dans l'ordre annoté et dans l'ordre
  prédit (métrique d'ordre de LayoutReader).

Implémentées dans `finlam_prod.evaluation.finlam`. Un exemple à la main sur cinq zones :
""")

py("""
areas = [1.0, 1.0, 1.0, 1.0, 1.0]
gold = [0, 0, 0, 1, 1]           # deux articles : zones 0-2 et zones 3-4
exact = [7, 7, 7, 9, 9]          # mêmes groupes, autres identifiants
split_one = [0, 0, 5, 1, 1]      # l'article 0 coupé en deux
merged = [0, 0, 0, 0, 0]         # tout fusionné
for name, pred in [("identique", exact), ("un article coupé", split_one), ("tout fusionné", merged)]:
    s = article_separation(gold, pred, areas, threshold=0.5)
    print(f"{name:18s} AS F1 {s['as_f1']:.2f}  mIoU {s['as_miou']:.2f}  jaccard {jaccard_count(gold, pred):.2f}")
print("BLEU ordre identique :", round(bleu_block([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), 2))
print("BLEU deux blocs échangés :", round(bleu_block([0, 1, 2, 3, 4], [0, 1, 3, 2, 4]), 2))
print("BLEU ordre inversé :", round(bleu_block([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]), 2))
""")

md("""
## 3. Contrôles : la vérité terrain contre elle-même, puis dégradée

Si l'implémentation est correcte, la vérité terrain évaluée contre elle-même vaut 1,0 partout, et
chaque dégradation contrôlée fait baisser la métrique qui doit baisser, et seulement elle.
""")

py("""
rng = np.random.default_rng(0)


def perturb(page, kind):
    articles = [int(z.article_id) for z in page.zones]
    order = [int(z.order) for z in page.zones]
    ids = sorted({a for a in articles if a != NO_ARTICLE})
    if kind == "identique":
        return articles, order
    if kind == "fusion de 2 articles" and len(ids) >= 2:
        a, b = rng.choice(ids, 2, replace=False)
        return [a if x == b else x for x in articles], order
    if kind == "1 article coupé en 2" and ids:
        a = rng.choice(ids)
        members = [i for i, x in enumerate(articles) if x == a]
        half = set(members[len(members) // 2 :])
        new = max(ids) + 1
        return [new if i in half else x for i, x in enumerate(articles)], order
    if kind == "2 zones échangées (ordre)" and len(order) >= 2:
        i, j = rng.choice(len(order), 2, replace=False)
        order = list(order)
        order[i], order[j] = order[j], order[i]
        return articles, order
    if kind == "ordre aléatoire":
        return articles, [int(o) for o in rng.permutation(order)]
    return articles, order  # page trop petite pour cette perturbation : inchangée


rows = {}
for kind in ["identique", "fusion de 2 articles", "1 article coupé en 2", "2 zones échangées (ordre)", "ordre aléatoire"]:
    per_page = [finlam_page_scores(p, *perturb(p, kind)) for p in pages]
    ci = bootstrap_ci(per_page, n_boot=300, seed=0)
    rows[kind] = {m: f"{ci[m][0]:.3f}" for m in ["as_f1", "as_miou", "jaccard_article", "bleu_block"]}
pd.DataFrame(rows).T
""")

md("""
Lecture attendue : ligne « identique » à 1,000 ; les deux dégradations d'articles font baisser AS F1
et Jaccard sans toucher BLEU ; les deux dégradations d'ordre font baisser BLEU sans toucher AS.
Une fusion de deux articles sur ~23 coûte environ 2 × (1/23) de F1 : l'AS est une métrique par
article, pas par zone — les gros articles et les petits pèsent pareil.
""")

md("""
## 4. Les références de l'article

Recopiées de la table des résultats (même split). L'article ne donne pas d'intervalle : ce sont des
estimations ponctuelles. Sur 433 pages, nos propres IC95 % font ± 1,5 point (carnet 01) ; deux
systèmes à moins de 3 points d'écart ne sont pas départagés par ce jeu.
""")

py("""
pd.DataFrame(
    {
        "AS@0.5 F1": [0.804, 0.707, 0.594],
        "BLEU_block": [0.872, None, None],
        "s / page": ["0.83 GPU, 1.66 CPU", "≈57 CPU", "—"],
    },
    index=["pipeline modulaire (YOLO → séparateurs → LayoutReader → regroupeur)", "Tiramisu (bout-en-bout)", "Arcanum"],
)
""")

md("""
## 5. Ce qu'il reste pour une reproduction complète

1. Les **sorties du prototype** sur ce split (un JSON par page au contrat `pipeline/run.py`) :
   `finlam-prod evaluate --dataset la-liberte --split test --predictions dossier/` doit alors
   redonner 0,804 à l'incertitude près. Si ce n'est pas le cas, l'écart se lit dans ce carnet :
   granularité (ligne / bloc), seuil d'IoU, calcul de l'IoU (surface des zones ici, masque pixel
   chez eux si les zones se recouvrent), traitement des zones hors article.
2. Le **dépôt d'évaluation** du consortium (`gitlab.teklia.com/adr/newspaper/evaluation`) pour
   comparer les implémentations sur les mêmes cas de la section 3.
3. Une réponse à la question : **un sous-titre de rubrique ouvre-t-il un article ?** (voir carnet 01,
   sur-segmentation des rubriques).
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out = Path(__file__).with_name("00_cadre_evaluation.ipynb")
nbf.write(nb, out)
print(f"écrit {out}")
