# finlam-prod

**Industrialisation du pipeline FINLAM de structuration de la presse numérisée : contrat d'échange,
conversion ALTO/METS, contrôle qualité, évaluation, service.**

Le consortium FINLAM (LITIS, TEKLIA, BnF) a produit un prototype qui segmente une page de journal en
blocs, les ordonne et les regroupe en articles ([Mocaër et al., 2026](https://arxiv.org/abs/2607.15082)).
L'article conclut que le pipeline modulaire (YOLO → séparateurs → LayoutReader → regroupeur) est celui
à déployer, et annonce sa mise en œuvre par la BnF sur plus de 8 millions de documents d'ici fin 2026.
Ce dépôt construit **ce qui vient après le prototype** : tout ce qui manque entre une sortie de modèle
et un fascicule archivable dans Gallica.

| Brique | Module | Ce qu'elle fait |
|---|---|---|
| Contrat d'échange | `pipeline/run.py` | Format JSON de sortie du pipeline (une entrée par page, zones avec classe, ordre, article), lecture/écriture, versionné |
| Conversion | `convert/alto.py`, `convert/mets.py` | JSON → ALTO 4.4 par page (classes en `LayoutTag`, articles en `StructureTag`, `ReadingOrder`) et METS par fascicule (carte de structure logique par article) ; **validés contre les XSD embarqués**, hors ligne |
| Lecture de l'existant | `data/mets_alto.py` | Relit un METS/ALTO « mode article » (le nôtre, ou celui d'Europeana Newspapers déjà dans Gallica) en structure comparable |
| Contrôle qualité | `qc/checks.py` | Détection d'incohérences sans vérité terrain : géométrie, chevauchements, texte/classe, ordre, articles, confiance, **pages à tableaux** (Neudecker & Antonacopoulos 2016), **qualité texte estimée** (Ben Salah et al. 2015) |
| Évaluation | `evaluation/finlam.py`, `evaluation/run.py` | **Les métriques du consortium** (AS@0.5 avec appariement hongrois, Jaccard article, BLEU_block) plus F1 par couples et tau de Kendall décomposé ; **IC95 % par bootstrap**, coût par page ; traçage MLflow optionnel |
| Données | `data/finlam.py`, `data/la_liberte.py`, `data/gallica.py` | Jeux FINLAM multi-titres (721 p.) et **La Liberté** (8 836 p., jeu des résultats de l'article) ; client Gallica (IIIF + ALTO) à débit limité |
| Doublure | `pipeline/rules.py`, `pipeline/ml.py` | Règles + gradient boosting produisant des sorties au bon format **en attendant celles du prototype** ; à débrancher, pas à améliorer |
| Service | `service/api.py` | FastAPI : `POST /issues` (JSON → zip ALTO+METS+QC), `/issues/qc`, `/issues/predict` ; OpenAPI sur `/docs` |
| Orchestration | `dags/finlam_pipeline.py` | DAG Airflow : fetch → predict → convert → qc → archive ou relecture humaine |
| Documentation | `docs/` | DAT, installation, constats sur les données, protocole de jeu gold, état de l'art, plan du rapport ANR |

## Installation et prise en main

```bash
python -m venv .venv && source .venv/bin/activate          # ou uv venv
pip install -e ".[data,service,dev]"                        # + [tracking] pour MLflow, [airflow] pour le DAG
pytest                                                      # 45 tests, sans réseau

finlam-prod download --dataset la-liberte --split test      # le split de l'article (433 pages)
finlam-prod evaluate --dataset la-liberte --split test      # règles seules, métriques du consortium + IC95 %
finlam-prod train --dataset la-liberte && finlam-prod evaluate --dataset la-liberte --models-dir models
finlam-prod run bpt6k7155522 --pages 1,2,3 --models-dir models   # fascicule Gallica → JSON, ALTO, METS, QC
finlam-prod convert results/run/bpt6k7155522/pipeline.json  # le contrat JSON seul → ALTO/METS/QC
finlam-prod qc results/run/bpt6k7155522/mets.xml            # QC d'un METS/ALTO existant
finlam-prod compare ancien/mets.xml nouveau/mets.xml        # OLR existant vs nouveau, mêmes métriques
finlam-prod serve                                           # http://127.0.0.1:8000/docs
```

Quand les sorties du prototype sont disponibles : `finlam-prod evaluate --predictions dossier_json/` les
évalue à la place de la doublure, et `finlam-prod convert` les archive. Rien d'autre ne change.

## Résultats

Métriques de l'article (AS@0.5, Jaccard, BLEU_block) rapportées avec IC95 % (bootstrap sur les pages),
au format de HIPE-OCRepair (ICDAR 2026). **Jeu FINLAM multi-titres, split test (48 pages)** :

| métrique | règles seules | doublure (règles + boosting) |
|---|---|---|
| AS@0.5 F1 | 0.188 [0.152 ; 0.232] | 0.280 [0.221 ; 0.341] |
| AS@0.5 mIoU | 0.708 | 0.695 |
| Jaccard article | 0.413 | 0.468 |
| BLEU_block (ordre) | 0.526 [0.484 ; 0.568] | 0.526 |
| F1 par couples | 0.392 | 0.540 |
| exactitude classes / F1 macro | 0.601 / 0.345 | 0.827 / 0.353 |
| coût par page (CPU, 1 cœur) | 0.062 s | 0.157 s |

Ces chiffres **ne sont pas comparables** à ceux de l'article (F1 articles 80,4 % du pipeline, 59,4 %
d'Arcanum) : jeu différent (multi-titres, plus dur que La Liberté) et doublure sans modèle d'image.
Ils servent à trois choses : vérifier que la chaîne d'évaluation produit les bonnes métriques, montrer
la largeur des intervalles sur 48 pages (± 6 points), et donner un plancher. La comparaison honnête
se fait sur le split test de La Liberté (`--dataset la-liberte`), avec les sorties du prototype.

Ce que l'analyse des données a montré, et qui conditionne toute lecture de ces chiffres
(`docs/donnees_finlam.md`) : dans le jeu multi-titres, la séquence des articles de `zone_orders` est un
tri lexicographique d'identifiants (artefact d'export), 40 % des articles annotés n'ont pas de titre,
et les polygones se touchent bord à bord. L'article lui-même signale une vérité terrain dérivée des
METS/ALTO existants et un gold standard « en cours » ; `docs/protocole_gold.md` propose comment le
constituer.

## Choix techniques

- **Le contrat JSON est l'interface** entre le prototype (modèles, GPU) et l'archivage (conversion,
  QC, dépôt, CPU). Les deux moitiés se déploient et se testent séparément.
- **Coordonnées en pourcentage de page** dans tout le paquet ; pixels seulement à l'export.
- **Validation XSD hors ligne** : ALTO 4.4, METS 1.12 et xlink embarqués (réseau fermé possible).
- **QC sans vérité terrain** : en production il n'y a pas d'annotation ; on contrôle la cohérence, on
  route vers relecture au-dessus de seuils (`QCConfig`), et deux règles viennent de la littérature.
- **Évaluation en chaîne, jamais isolée** : l'ordre annoté encode déjà les frontières d'articles ;
  seule la chaîne complète mesure ce qu'un usager de Gallica verra.
- **Dépendances lourdes optionnelles** : le service et la conversion tournent sans torch ni GPU.

## Limites connues

Positions des lignes et des mots dans l'ALTO estimées par répartition dans le bloc (un ALTO de
production reprend celles de l'OCR) ; profils ALTO/METS de la BnF (`alto_bnf-v2_0`, METS BnF) non
encore appliqués ; sorties réelles du prototype non branchées ; comparaison avec l'OLR d'Europeana
Newspapers outillée (`compare`) mais non exécutée faute d'accès au jeu FTP depuis ici.

## Licence et données

Code MIT. Jeux FINLAM : Teklia / LITIS / BnF, MIT. Documents Gallica : conditions d'utilisation BnF.
