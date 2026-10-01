# Journal des versions

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
