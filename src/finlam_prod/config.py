"""Configuration centralisée, lue dans l'environnement.

Une seule source de vérité pour les chemins, les seuils d'exploitation et les
secrets, afin que la CLI, le service et le DAG Airflow lisent les mêmes
valeurs. Toutes les variables sont préfixées ``FINLAM_`` ; les valeurs par
défaut conviennent à un poste de développement.

- ``FINLAM_CACHE_DIR`` : cache des jeux de données (JSON Lines) — ``data/cache``
- ``FINLAM_MODELS_DIR`` : dossier de la doublure entraînée (vide : règles seules)
- ``FINLAM_RESULTS_DIR`` : sorties d'évaluation et de ``run`` — ``results``
- ``FINLAM_WORK_DIR`` : répertoire de travail du DAG — ``/tmp/finlam_prod``
- ``FINLAM_PREDICT_URL`` : service de prédiction du prototype (vide : doublure)
- ``FINLAM_API_KEY`` : clé attendue dans l'en-tête ``X-API-Key`` (vide : pas d'auth)
- ``FINLAM_MAX_PAGES`` : pages max par requête au service — ``200``
- ``FINLAM_MAX_ZONES_PER_PAGE`` : zones max par page acceptées — ``5000``
- ``FINLAM_LOG_LEVEL`` : ``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` — ``INFO``
- ``FINLAM_LOG_JSON`` : ``1`` pour des journaux JSON, une ligne par événement — ``0``
- ``FINLAM_MLFLOW_EXPERIMENT`` : nom de l'expérience MLflow — ``finlam-prod``

Un fichier ``.env`` à la racine est lu s'il existe (sans dépendance externe :
lignes ``CLE=valeur``, ``#`` pour les commentaires) ; l'environnement réel a
priorité.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

PREFIX = "FINLAM_"


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


@dataclass(frozen=True)
class Settings:
    """Valeurs de configuration ; construire avec :func:`Settings.from_env`."""

    cache_dir: Path = Path("data/cache")
    models_dir: Path | None = None
    results_dir: Path = Path("results")
    work_dir: Path = Path("/tmp/finlam_prod")
    predict_url: str = ""
    api_key: str = ""
    max_pages: int = 200
    max_zones_per_page: int = 5000
    log_level: str = "INFO"
    log_json: bool = False
    mlflow_experiment: str = "finlam-prod"
    source: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_env(
        cls, env: dict[str, str] | None = None, dotenv: Path | None = Path(".env")
    ) -> Settings:
        """Lit ``.env`` puis l'environnement (qui a priorité) ; valide les types."""
        merged: dict[str, str] = {}
        if dotenv is not None:
            merged.update(_read_dotenv(dotenv))
        merged.update(env if env is not None else os.environ)
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            if f.name == "source":
                continue
            key = PREFIX + f.name.upper()
            if key not in merged or merged[key] == "":
                continue
            raw = merged[key]
            if f.name in ("cache_dir", "models_dir", "results_dir", "work_dir"):
                kwargs[f.name] = Path(raw)
            elif f.name in ("max_pages", "max_zones_per_page"):
                try:
                    kwargs[f.name] = int(raw)
                except ValueError as exc:
                    raise ValueError(f"{key} doit être un entier, reçu {raw!r}") from exc
            elif f.name == "log_json":
                kwargs[f.name] = raw.lower() in ("1", "true", "yes", "on")
            elif f.name == "log_level":
                level = raw.upper()
                if level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                    raise ValueError(f"{key} invalide : {raw!r}")
                kwargs[f.name] = level
            else:
                kwargs[f.name] = raw
        kwargs["source"] = {
            k: v for k, v in merged.items() if k.startswith(PREFIX) and "KEY" not in k
        }
        return cls(**kwargs)  # type: ignore[arg-type]

    def describe(self) -> dict[str, str]:
        """Valeurs effectives, secrets masqués (pour les journaux et ``/health``)."""
        out = {}
        for f in fields(self):
            if f.name == "source":
                continue
            value = getattr(self, f.name)
            if f.name == "api_key":
                value = "***" if value else ""
            out[f.name] = str(value) if value is not None else ""
        return out


_SETTINGS: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Instance partagée, construite au premier appel."""
    global _SETTINGS
    if _SETTINGS is None or reload:
        _SETTINGS = Settings.from_env()
    return _SETTINGS
