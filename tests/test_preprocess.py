"""Prétraitement des pages pliées : détection du pli, éclairage, redressement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from finlam_prod.preprocess import (  # noqa: E402
    deskew_angle,
    detect_fold,
    normalise_illumination,
    preprocess_page,
)
from finlam_prod.preprocess.fold import rotate  # noqa: E402
from finlam_prod.preprocess.synthetic import folded_page  # noqa: E402


@pytest.fixture(scope="module")
def page():
    return folded_page(seed=1)


def test_fold_detected_at_the_right_place(page):
    report = detect_fold(page.image)
    assert report.detected
    assert report.dark
    assert abs(report.x - page.fold_x) <= 2
    assert report.strength >= 4


def test_no_fold_on_a_flat_page():
    flat = folded_page(angles=(0.0, 0.0), shadow_depth=0, seed=2)
    assert not detect_fold(flat.image).detected


def test_faint_fold_still_detected():
    faint = folded_page(shadow_depth=25, seed=3)
    assert detect_fold(faint.image).detected


def test_illumination_normalisation_removes_the_shadow(page):
    before = detect_fold(page.image)
    after = detect_fold(normalise_illumination(page.image))
    assert abs(after.depth) < abs(before.depth) / 5
    assert not after.detected


def test_deskew_angle_recovers_a_known_tilt():
    clean = folded_page(angles=(0.0, 0.0), shadow_depth=0, seed=4).clean
    for tilt in (-2.0, -0.5, 1.2, 2.7):
        tilted = rotate(clean, tilt)
        assert abs(deskew_angle(tilted) + tilt) <= 0.25, tilt


def test_deskew_angle_is_zero_without_text():
    blank = np.full((400, 300), 240, dtype=np.uint8)
    assert deskew_angle(blank) == 0.0


@pytest.mark.parametrize("angles", [(1.5, -1.0), (-2.0, 0.5), (0.0, 0.0)])
def test_preprocess_page_corrects_each_half(angles):
    page = folded_page(angles=angles, seed=5)
    result = preprocess_page(page.image)
    assert result.fold.detected
    assert len(result.angles) == 2
    assert abs(result.angles[0] + angles[0]) <= 0.25
    assert abs(result.angles[1] + angles[1]) <= 0.25
    assert result.image.shape == page.image.shape
    assert not detect_fold(result.image).detected
    # sérialisable tel quel pour le QC et les journaux
    json.dumps(result.to_dict())


def test_preprocess_page_accepts_colour_input(page):
    colour = cv2.cvtColor(page.image, cv2.COLOR_GRAY2BGR)
    result = preprocess_page(colour)
    assert result.image.ndim == 2
    assert result.fold.detected


def test_cli_preprocess_writes_an_image_and_a_report(tmp_path, page):
    from typer.testing import CliRunner

    from finlam_prod.cli import app

    source = tmp_path / "page.png"
    cv2.imwrite(str(source), page.image)
    result = CliRunner().invoke(app, ["preprocess", str(source)])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)  # les journaux partent sur stderr
    assert report["fold"]["detected"]
    assert (tmp_path / "page_pre.png").is_file()


def test_cli_preprocess_unreadable_image_is_a_clean_error(tmp_path):
    from typer.testing import CliRunner

    from finlam_prod.cli import app
    from finlam_prod.errors import PreprocessError

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"pas une image")
    result = CliRunner().invoke(app, ["preprocess", str(bad)])
    assert result.exit_code != 0
    assert isinstance(result.exception, PreprocessError)  # main() en fait une ligne + code 1


def test_bright_band_is_layout_not_fold():
    # une colonne vide claire (pleine page de publicité) ne doit pas passer pour un pli
    page = folded_page(angles=(0.0, 0.0), shadow_depth=-60, seed=6)
    assert not detect_fold(page.image).detected
    assert detect_fold(page.image, allow_bright=True).depth > 0


def test_dewarp_network_shapes():
    torch = pytest.importorskip("torch")
    from finlam_prod.preprocess.dewarp import GRID_SIZE, INPUT_SIZE, unwarp
    from finlam_prod.preprocess.uvdoc_net import UVDocnet

    model = UVDocnet(num_filter=32, kernel_size=5).eval()
    with torch.no_grad():
        grid, _ = model(torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0]))
    assert tuple(grid.shape) == (1, 2, GRID_SIZE[0], GRID_SIZE[1])
    # grille identité : l'image ressort inchangée
    ys, xs = np.meshgrid(
        np.linspace(-1, 1, GRID_SIZE[0]), np.linspace(-1, 1, GRID_SIZE[1]), indexing="ij"
    )
    identity = np.stack([xs, ys]).astype(np.float32)
    image = np.random.default_rng(0).integers(0, 255, (120, 90, 3), dtype=np.uint8)
    out = unwarp(image, identity)
    assert out.shape == image.shape
    assert np.abs(out.astype(int) - image.astype(int)).mean() < 3


def test_dewarp_weights_if_present():
    pytest.importorskip("torch")
    from finlam_prod.preprocess.dewarp import WEIGHTS_NAME, dewarp_page, load_model

    weights = Path("models") / WEIGHTS_NAME
    if not weights.is_file():
        pytest.skip("poids UVDoc non téléchargés")
    model = load_model(weights)
    page = folded_page(seed=8)
    out = dewarp_page(page.image, model)
    assert out.shape[:2] == page.image.shape[:2]
