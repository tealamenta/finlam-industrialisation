# Dossier d'architecture technique (DAT) — finlam-prod

Version 0.1. Décrit l'architecture implémentée pour industrialiser la sortie du pipeline FINLAM,
et ce qui reste à faire pour une intégration dans le SI de la BnF.

## 1. Périmètre

**Entrée** : la sortie du prototype FINLAM pour un fascicule (blocs, classes, ordre de lecture,
articles), sous forme du contrat JSON ; ou, à défaut, un fascicule Gallica (images IIIF + ALTO) traité
par la doublure.
**Sortie** : un fascicule archivable (ALTO 4.4 par page, METS avec carte de structure logique par
article, validés XSD) et un rapport de contrôle qualité qui décide de l'archivage ou de la relecture.
**Hors périmètre** : les modèles du prototype (détection, ordre, regroupement) et l'OCR.

## 2. Flux de fichiers

```
                                   ┌──────────────── prototype FINLAM (GPU) ────────────────┐
 Gallica (IIIF + ALTO) ───────────▶│  YOLO → séparateurs → LayoutReader → regroupeur         │
 ou lot de fascicules              └──────────────────────────┬──────────────────────────────┘
                                                              │ contrat JSON (une entrée par page)
                                            ┌─────────────────▼─────────────────┐
                                            │ finlam-prod (CPU, sans torch)     │
                                            │  convert : JSON → ALTO 4.4 + METS │
                                            │  validation XSD (hors ligne)      │
                                            │  qc : cohérence, seuils, sources  │
                                            └───────┬───────────────┬───────────┘
                                                    │ QC ok         │ QC bloquant
                                                ┌───▼────┐     ┌────▼───────┐
                                                │archive │     │ relecture  │
                                                │(SAN,   │     │ humaine    │
                                                │Gallica)│     │ (Callico…) │
                                                └────────┘     └────────────┘
 Évaluation (hors flux) : contrat JSON + annotations → métriques du consortium + IC95 % → MLflow
```

Le **contrat JSON** (`pipeline/run.py`) est l'interface : versionné (`pipeline.version`), lisible sans
les modèles, il découple le prototype (LITIS/TEKLIA, GPU) de l'archivage (BnF, CPU). Chaque moitié se
déploie, se teste et se fait évoluer séparément.

## 3. Composants

| Composant | Rôle | Dépendances | Remplaçable par |
|---|---|---|---|
| `data.schema` | Modèle de données (`Zone`, `Page`, `Issue`), JSONL | stdlib | — |
| `data.finlam`, `data.la_liberte` | Jeux d'évaluation, cache local | `datasets` (optionnel) | export Arkindex |
| `data.gallica` | Client IIIF/ALTO à débit limité, parseur ALTO BnF | `requests`, `lxml` | accès direct au SAN |
| `data.mets_alto` | Lecture METS/ALTO « mode article » (existant ou produit) | `lxml` | — |
| `pipeline.run` | Contrat JSON, orchestration doublure | — | — |
| `pipeline.rules`, `pipeline.ml` | Doublure (règles, boosting) | `scikit-learn` | **le prototype FINLAM** |
| `convert.alto`, `convert.mets` | Conversion + validation XSD | `lxml` | profils ALTO/METS BnF |
| `qc.checks` | Contrôles, seuils, règles issues de la littérature | `pandas` | — |
| `evaluation.*` | Métriques du consortium, bootstrap, MLflow | `scipy`, `mlflow` (opt.) | dépôt d'évaluation TEKLIA |
| `service.api` | FastAPI : conversion, QC, doublure ; clé d'API, limites, `/metrics` | `fastapi` | — |
| `config`, `logging_utils`, `errors` | Configuration `FINLAM_*`, journaux JSON, exceptions typées | stdlib | — |
| `pipeline.registry` | Versionnage des modèles (`metadata.json`, empreintes) | stdlib | registre MLflow |
| `dags/finlam_pipeline.py` | Orchestration par fascicule | Airflow | file de traitement interne |

## 4. Choix structurants

1. **Contrat JSON comme frontière** entre recherche et production ; adaptateur unique à écrire quand
   le format réel du prototype sera connu (`page_from_json`).
2. **Coordonnées relatives** dans tout le traitement, pixels à l'export : indépendance à la résolution.
3. **Validation XSD embarquée** : fonctionne en réseau fermé.
4. **QC sans vérité terrain** avec seuils par strate et routage vers relecture ; règles documentées
   par leur source (tableaux : Neudecker & Antonacopoulos 2016 ; qualité texte : Ben Salah et al. 2015).
5. **Évaluation au format du consortium** (AS@0.5, Jaccard, BLEU_block) et de HIPE-OCRepair
   (IC95 %), pour que le rapport ANR parle la langue des partenaires.
6. **Service et DAG sans modèle** : l'archivage est léger et scalable horizontalement ; le prototype
   est appelé par URL (`FINLAM_PREDICT_URL`) ou remplacé par la doublure pour les tests.

## 5. Qualité logicielle

PEP 8 via ruff (E/W/F/I/N/UP/B/SIM/D), docstrings sur l'API publique, 69 tests pytest sans réseau
(fixtures synthétiques, aller-retour METS/ALTO, service via TestClient, erreurs, bout en bout, structure du DAG), typage vérifié par mypy, pre-commit, CI sur Python
3.10-3.12, journalisation par module, `random_state` fixé, Dockerfile sans torch.

**Gestion des erreurs** (`errors.py`). Une hiérarchie courte distingue les erreurs *attendues* des
bugs : `FinlamError` (base), `ContractError` (contrat JSON invalide : clé manquante, type inattendu,
classe inconnue, polygone dégénéré — avec la page et le champ fautifs), `DatasetError` (jeu absent ou
inexploitable), `GallicaError` (ARK, pagination, ALTO absent, échecs réseau après reprises),
`ConversionError` (écriture ALTO/METS). Chaque couche sait quoi attraper : la CLI affiche une ligne et
sort avec le code 1 ; le service renvoie 422 pour un contrat invalide et 500 pour le reste, avec le
message ; le DAG laisse remonter pour qu'Airflow relance ou marque la tâche en échec. Les erreurs de
programmation ne sont jamais converties : elles gardent leur trace. Règle : on attrape près de la
cause ce qu'on sait traduire en message utile, et rien d'autre.

## 6. Ce qui reste pour la production BnF

| Sujet | État | À faire |
|---|---|---|
| Format réel du prototype | contrat JSON générique | adaptateur depuis l'export du pipeline (JSON dérivé de METS/ALTO selon l'article) |
| Profils BnF | ALTO 4.4 / METS standard | `alto_bnf-v2_0`, METS BnF, positions fines des lignes/mots reprises de l'OCR |
| Compatibilité avec l'existant | lecteur METS/ALTO + `compare` | exécuter sur fascicules communs avec l'OLR Europeana Newspapers |
| Jeu gold | protocole écrit | constitution avec le consortium (`docs/protocole_gold.md`) |
| Exploitation | config centralisée, clé d'API, `/metrics`, journaux JSON, modèles versionnés (`docs/exploitation.md`) | file de fascicules, TLS et débit au reverse proxy, image signée, utilisateur non-root, secrets |
| Modèles | doublure | brancher les modèles du prototype (workers Arkindex ou service dédié) |
