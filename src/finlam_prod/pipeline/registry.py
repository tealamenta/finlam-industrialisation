"""Versionnage des modèles de la doublure : un ``metadata.json`` à côté des fichiers.

Un fichier ``joblib`` seul ne dit ni avec quoi il a été entraîné, ni avec quelle
version du code, ni s'il a été altéré. ``write_metadata`` enregistre, au moment
de l'entraînement : version du paquet, commit git s'il existe, jeu et split,
nombre de pages, date, versions de scikit-learn et de Python, et l'empreinte
SHA-256 de chaque fichier. ``load_metadata`` relit et vérifie les empreintes ;
``check_compatibility`` avertit si la version de scikit-learn a changé (un
modèle sérialisé n'est garanti que pour celle qui l'a produit).
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from finlam_prod import __version__
from finlam_prod.errors import DatasetError

LOGGER = logging.getLogger(__name__)
METADATA_FILE = "metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


@dataclass
class ModelMetadata:
    """Carte d'identité d'un dossier de modèles."""

    package_version: str
    created_at: str
    dataset: str
    split: str
    n_pages: int
    files: dict[str, str]  # nom → sha256
    sklearn_version: str
    python_version: str
    git_commit: str | None = None
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        """Identifiant court et stable : paquet + date + 8 hex de l'empreinte du classifieur."""
        first = next(iter(sorted(self.files.values())), "00000000")
        return f"{self.package_version}-{self.created_at[:10]}-{first[:8]}"


def write_metadata(
    models_dir: str | Path,
    dataset: str,
    split: str,
    n_pages: int,
    notes: str = "",
    extra: dict[str, object] | None = None,
) -> ModelMetadata:
    """Écrit ``metadata.json`` pour les ``*.joblib`` présents dans ``models_dir``."""
    import sklearn

    models_dir = Path(models_dir)
    files = {p.name: _sha256(p) for p in sorted(models_dir.glob("*.joblib"))}
    if not files:
        raise DatasetError(f"aucun fichier .joblib dans {models_dir}")
    meta = ModelMetadata(
        package_version=__version__,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=dataset,
        split=split,
        n_pages=n_pages,
        files=files,
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
        git_commit=_git_commit(),
        notes=notes,
        extra=extra or {},
    )
    (models_dir / METADATA_FILE).write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def load_metadata(models_dir: str | Path, verify: bool = True) -> ModelMetadata | None:
    """Relit ``metadata.json`` ; ``None`` s'il n'existe pas ; vérifie les empreintes."""
    path = Path(models_dir) / METADATA_FILE
    if not path.is_file():
        LOGGER.warning("%s sans %s : modèles non versionnés", models_dir, METADATA_FILE)
        return None
    meta = ModelMetadata(**json.loads(path.read_text(encoding="utf-8")))
    if verify:
        for name, expected in meta.files.items():
            target = Path(models_dir) / name
            if not target.is_file():
                raise DatasetError(f"{name} manquant dans {models_dir} (déclaré dans metadata)")
            actual = _sha256(target)
            if actual != expected:
                raise DatasetError(
                    f"{name} altéré : sha256 {actual[:12]}… ≠ metadata {expected[:12]}…"
                )
    return meta


def check_compatibility(meta: ModelMetadata) -> list[str]:
    """Avertissements si l'environnement diffère de celui de l'entraînement."""
    import sklearn

    warnings = []
    if meta.sklearn_version != sklearn.__version__:
        warnings.append(
            f"scikit-learn {sklearn.__version__} ≠ {meta.sklearn_version} à l'entraînement : "
            "réentraîner avant mise en production"
        )
    if meta.package_version != __version__:
        warnings.append(f"paquet {__version__} ≠ {meta.package_version} à l'entraînement")
    for w in warnings:
        LOGGER.warning(w)
    return warnings
