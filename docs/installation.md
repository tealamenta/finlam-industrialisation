# Manuel d'installation

## Prérequis

- Python ≥ 3.10 (testé 3.10–3.12), `pip` ou `uv`.
- Accès réseau à `huggingface.co` (jeu FINLAM) et `gallica.bnf.fr` (fascicules) pour les commandes
  `download` et `run <ark>` ; tout le reste fonctionne hors ligne.
- Optionnel : GPU CUDA ou Apple Silicon (`mps`) pour l'affinage CamemBERT.

## Installation développeur

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[data,service,dev]"
pytest && ruff check src tests
```

Avec `uv` : `uv venv && source .venv/bin/activate && uv pip install -e ".[data,service,dev]"`.

## Extras

| extra | contenu | quand |
|---|---|---|
| `data` | `datasets`, `huggingface-hub` | `finlam-prod download` |
| `finetune` | torch, transformers, accelerate | `finlam-prod finetune`, notebook 03 |
| `dev` | pytest, ruff, mypy | développement |

## Conteneur

```bash
docker build -t finlam-prod .
docker run --rm -v $PWD/models:/app/models -v $PWD/results:/app/results finlam-prod run bpt6k7155522 --pages 1,2 --models-dir models
```

## Données et modèles

- `finlam-prod download` écrit `data/cache/finlam_{train,val,test}.jsonl` et `data/cache/images/`.
- `finlam-prod train` écrit `models/{zone_classifier,orderer,segmenter}.joblib`.
- `finlam-prod finetune` écrit `models/camembert_zone/final/`.

## Variables et limites

- Le client Gallica attend 3 s entre deux requêtes (`GallicaClient(min_interval=…)`) et réessaie 5 fois
  sur HTTP 429/5xx avec attente exponentielle.
- La validation XSD est hors ligne : schémas dans `src/finlam_prod/schemas/`.
