"""Versionnage des modèles : metadata, empreintes, compatibilité."""

import json

import pytest

from finlam_prod.errors import DatasetError
from finlam_prod.pipeline.ml import BoundarySegmenter, GradientBoostingZoneClassifier
from finlam_prod.pipeline.registry import (
    METADATA_FILE,
    check_compatibility,
    load_metadata,
    write_metadata,
)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory, train_pages):
    target = tmp_path_factory.mktemp("models")
    GradientBoostingZoneClassifier().fit(train_pages).save(target / "zone_classifier.joblib")
    BoundarySegmenter().fit(train_pages).save(target / "segmenter.joblib")
    write_metadata(target, dataset="synthetic", split="train", n_pages=len(train_pages))
    return target


def test_metadata_written_and_reloaded(models_dir):
    meta = load_metadata(models_dir)
    assert meta is not None
    assert set(meta.files) == {"zone_classifier.joblib", "segmenter.joblib"}
    assert meta.dataset == "synthetic" and meta.n_pages > 0
    assert meta.model_version.startswith(meta.package_version)
    raw = json.loads((models_dir / METADATA_FILE).read_text(encoding="utf-8"))
    assert raw["sklearn_version"] and raw["python_version"]


def test_tampered_model_is_detected(models_dir):
    target = models_dir / "segmenter.joblib"
    original = target.read_bytes()
    target.write_bytes(original + b"\0")
    try:
        with pytest.raises(DatasetError, match="altéré"):
            load_metadata(models_dir)
    finally:
        target.write_bytes(original)
    assert load_metadata(models_dir) is not None


def test_missing_metadata_returns_none(tmp_path):
    assert load_metadata(tmp_path) is None


def test_compatibility_warns_on_version_drift(models_dir):
    meta = load_metadata(models_dir)
    assert check_compatibility(meta) == []
    meta.sklearn_version = "0.0.0"
    assert any("scikit-learn" in w for w in check_compatibility(meta))
