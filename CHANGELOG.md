# Journal des versions

## 0.3.0 — 2026-10-01

Prétraitement d'image.

- `preprocess/` (extra `image`, OpenCV) : journaux pliés et scannés — détection du pli dans le fond
  de page, normalisation de l'éclairage, redressement de chaque moitié ; rapport sérialisable pour
  le QC et les journaux ; commande `finlam-prod preprocess` ; pli simulé à vérité connue sur page
  synthétique ou réelle (`preprocess/synthetic.py`) ; carnet `02_pretraitement_pages_pliees`
  (40 vraies pages à pli simulé : 40/40 retrouvés ; 433 scans d'origine : 0 fausse alerte ; pires
  cas réels du split test via `scripts/find_worst_pages.py`) ; figure `docs/img/pli_avant_apres.png` ;
  13 tests.
- Dégauchissement appris (`preprocess/dewarp.py`, extra `dewarp`) : UVDoc (MIT, réseau repris dans
  `preprocess/uvdoc_net.py`, poids téléchargés une fois et vérifiés par SHA-256) pour les photos de
  pages pliées en 3D ; commande `finlam-prod dewarp` ; 2 tests (sautés sans torch).
- Tâche `preprocess` dans la chaîne (`preprocess/issue.py`) : DAG `fetch → preprocess → predict`,
  `finlam-prod run`, `FINLAM_PREPROCESS` / `FINLAM_DEWARP`, rapport `preprocess.json` repris dans le
  résumé QC (`folded_pages`, `skewed_pages`) ; 3 tests (87 au total).
- Mesure en aval (carnet `02`, section 8, extra `ocr`) : rappel de mots Tesseract contre la
  transcription de référence, quatre conditions (origine, origine normalisée, pliée, corrigée).
- `PreprocessError` dans la hiérarchie d'exceptions.
- CI : mypy suit la version de l'interpréteur (stubs numpy ≥ 2.4 en syntaxe 3.12) ; matrice
  d'évaluation construite sans affectation `.loc` (stubs pandas récents).

## 0.2.0 — 2026-10-01

Durcissement pour la pré-production.

- Exceptions typées (`errors.py`) : `ContractError`, `DatasetError`, `GallicaError`,
  `ConversionError` ; contrat JSON validé champ par champ ; CLI → une ligne et code 1 ;
  service → 422 / 500 ; erreurs Gallica 4xx explicites, reprises sur réseau et 5xx.
- Configuration centralisée `FINLAM_*` et `.env` (`config.py`), partagée par la CLI, le service et
  le DAG ; journaux JSON optionnels (`logging_utils.py`).
- Service : clé d'API (`X-API-Key`), limites de taille (413), `/metrics` Prometheus, journal des
  requêtes avec durée, `/health` enrichi.
- Versionnage des modèles (`pipeline/registry.py`) : `metadata.json`, empreintes SHA-256 vérifiées au
  chargement, alerte sur changement de version de scikit-learn ; modèles tracés dans MLflow.
- Typage vérifié par mypy (33 fichiers), `pre-commit`, test d'intégration bout en bout, test de
  structure du DAG ; 69 tests.
- Docs : `docs/exploitation.md` (configuration, sécurité, observabilité, versions, dimensionnement).

## 0.1.0 — 2026-09-25

Première version : contrat JSON, conversion ALTO 4.4 / METS validée XSD, lecteur METS/ALTO mode
article, contrôle qualité, métriques du consortium avec IC95 %, doublure règles + boosting, service
FastAPI, DAG Airflow, carnets d'analyse, DAT et documentation.
