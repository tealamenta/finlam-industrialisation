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
| Service | `service/api.py` | FastAPI : `POST /issues` (JSON → zip ALTO+METS+QC), `/issues/qc`, `/issues/predict` ; OpenAPI sur `/docs` ; clé d'API optionnelle, limites de taille, `/metrics` Prometheus, journal des requêtes |
| Configuration et journaux | `config.py`, `logging_utils.py` | Une seule source (`FINLAM_*`, `.env`) pour la CLI, le service et le DAG ; journaux JSON une ligne par événement en exploitation |
| Erreurs | `errors.py` | Exceptions typées par couche (`ContractError`, `DatasetError`, `GallicaError`, `ConversionError`) ; contrat JSON validé champ par champ ; CLI → une ligne et code 1, service → 422/500 |
| Versionnage des modèles | `pipeline/registry.py` | `metadata.json` à l'entraînement (version, données, empreintes SHA-256, commit, versions) ; vérification au chargement ; modèles tracés dans MLflow |
| Orchestration | `dags/finlam_pipeline.py` | DAG Airflow : fetch → predict → convert → qc → archive ou relecture humaine |
| Analyse | `evaluation/viz.py`, `notebooks/` | Figures d'analyse d'erreurs (distributions avec IC, confusion, pages dessinées vérité terrain / prédiction) ; deux carnets : `00_cadre_evaluation` (reproduire le **cadre** de l'article : jeu, split, métriques, contrôles ; pas ses chiffres, ses modèles ne sont pas publics) et `01_analyse_resultats_la_liberte` (notre apport : IC95 %, distributions, confusion, pages dessinées) |
| Documentation | `docs/` | DAT, installation, exploitation (configuration, sécurité, observabilité), constats sur les données, protocole de jeu gold, état de l'art, plan du rapport ANR |

## Installation et prise en main

```bash
python -m venv .venv && source .venv/bin/activate          # ou uv venv
pip install -e ".[data,service,dev]"                        # + [tracking] pour MLflow, [airflow] pour le DAG
pytest                                                      # 69 tests, sans réseau ; mypy et ruff en CI

finlam-prod download --dataset la-liberte --split test      # le split de l'article (433 pages)
finlam-prod evaluate --dataset la-liberte --split test      # règles seules, métriques du consortium + IC95 %
finlam-prod download --dataset la-liberte --split validation   # 446 pages, suffit à la doublure
finlam-prod train --dataset la-liberte --split validation --models-dir models/la_liberte
finlam-prod evaluate --dataset la-liberte --split test --models-dir models/la_liberte
finlam-prod run bpt6k7155522 --pages 1,2,3 --models-dir models   # fascicule Gallica → JSON, ALTO, METS, QC
finlam-prod convert results/run/bpt6k7155522/pipeline.json  # le contrat JSON seul → ALTO/METS/QC
finlam-prod qc results/run/bpt6k7155522/mets.xml            # QC d'un METS/ALTO existant
finlam-prod compare ancien/mets.xml nouveau/mets.xml        # OLR existant vs nouveau, mêmes métriques
finlam-prod serve                                           # http://127.0.0.1:8000/docs
jupyter lab notebooks/                                      # 00 cadre de l'article, 01 analyse d'erreurs (extra [notebooks])
```

Quand les sorties du prototype sont disponibles : `finlam-prod evaluate --predictions dossier_json/` les
évalue à la place de la doublure, et `finlam-prod convert` les archive. Rien d'autre ne change.

## Résultats

Métriques de l'article (AS@0.5, Jaccard, BLEU_block) rapportées avec IC95 % (bootstrap sur les pages),
au format de HIPE-OCRepair (ICDAR 2026).

**La Liberté, split test de l'article (433 pages, 88 951 zones)** — les chiffres comparables :

| métrique | règles seules | doublure (règles + boosting, entraînée sur le split *val*, 446 p.) | pipeline FINLAM (article) | Arcanum (article) |
|---|---|---|---|---|
| AS@0.5 F1 | 0.264 [0.250 ; 0.276] | **0.424 [0.409 ; 0.438]** | 0.804 | 0.594 |
| AS@0.5 mIoU | 0.742 | 0.808 | — | — |
| Jaccard article | 0.716 [0.696 ; 0.734] | 0.784 [0.768 ; 0.799] | — | — |
| BLEU_block (ordre) | 0.625 [0.612 ; 0.638] | 0.625 | 0.872 | — |
| F1 par couples | 0.471 | 0.585 | — | — |
| exactitude classes / F1 macro | 0.626 / 0.395 | 0.938 / 0.674 | — | — |
| coût par page (CPU, 1 cœur) | 0.025 s | 0.089 s | 0.83 s GPU / 1.66 s CPU | — |

Lecture : la doublure, sans modèle d'image et entraînée sur 446 pages, fait la moitié du pipeline du
consortium sur la séparation d'articles et vaut le plancher attendu ; l'ordre de lecture est le même
dans les deux lignes (même découpage XY guidé par les colonnes, le boosting ne touche pas à l'ordre).
Les intervalles sur 433 pages font ± 1,5 point : c'est l'ordre de grandeur à viser pour un jeu gold
(`docs/protocole_gold.md`). Chiffres du consortium : Mocaër et al. 2026, table des résultats, même
split ; ils restent à reproduire avec leurs sorties (`--predictions`).

**Jeu FINLAM multi-titres, split test (48 pages)** — non comparable à l'article, utile pour la
largeur des intervalles :

| métrique | règles seules | doublure |
|---|---|---|
| AS@0.5 F1 | 0.188 [0.152 ; 0.232] | 0.280 [0.221 ; 0.341] |
| Jaccard article | 0.413 | 0.468 |
| BLEU_block (ordre) | 0.526 [0.484 ; 0.568] | 0.526 |
| coût par page (CPU) | 0.062 s | 0.157 s |

Ce que l'analyse des données a montré, et qui conditionne toute lecture de ces chiffres
(`docs/donnees_finlam.md`) : les zones de *La Liberté* sont des **lignes** (205 par page, 23 articles
par page), celles du multi-titres des blocs ; les coordonnées sont en fraction dans l'un, en
pourcentage dans l'autre ; dans le multi-titres, la séquence des articles de `zone_orders` est un tri
lexicographique d'identifiants, 40 % des articles n'ont pas de titre, et les polygones se touchent
bord à bord. L'article signale une vérité terrain dérivée des METS/ALTO existants et un gold
standard « en cours » ; `docs/protocole_gold.md` propose comment le constituer.

## Exploitation

Tout se règle par variables d'environnement `FINLAM_*` (ou un fichier `.env`) : chemins, clé d'API
(`FINLAM_API_KEY`, en-tête `X-API-Key`), limites de taille des requêtes, niveau et format des
journaux (`FINLAM_LOG_JSON=1`), expérience MLflow. Le service expose `/health` (version, modèle,
limites) et `/metrics` (Prometheus : requêtes par route et statut, temps cumulé, pages traitées,
erreurs QC et XSD). Les modèles entraînés portent un `metadata.json` (version, jeu, empreintes,
commit) vérifié au chargement. Détail : `docs/exploitation.md`.

Qualité du code : PEP 8 et imports par ruff, typage vérifié par mypy, 69 tests dont un bout en bout
(fascicule synthétique → doublure → contrat → ALTO/METS validés → relecture → évaluation) ; tout
tourne en CI sur Python 3.10-3.12 et en `pre-commit`.

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

vLLM et Ollama, cités dans l'offre, ne sont pas utilisés : il n'y a pas de modèle de langage dans
cette chaîne ; ils entreraient en jeu pour une post-correction OCR par LLM, hors périmètre ici.
Positions des lignes et des mots dans l'ALTO estimées par répartition dans le bloc (un ALTO de
production reprend celles de l'OCR) ; profils ALTO/METS de la BnF (`alto_bnf-v2_0`, METS BnF) non
encore appliqués ; sorties réelles du prototype non branchées ; comparaison avec l'OLR d'Europeana
Newspapers outillée (`compare`) mais non exécutée faute d'accès au jeu FTP depuis ici.

## Licence et données

Code MIT. Jeux FINLAM : Teklia / LITIS / BnF, MIT. Documents Gallica : conditions d'utilisation BnF.
