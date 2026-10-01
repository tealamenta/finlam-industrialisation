"""Prétraitement d'image avant OCR et segmentation (journaux pliés, éclairage, inclinaison)."""

from finlam_prod.preprocess.fold import (
    FoldReport,
    PreprocessResult,
    deskew_angle,
    detect_fold,
    estimate_background,
    normalise_illumination,
    preprocess_page,
    straighten_halves,
)

__all__ = [
    "FoldReport",
    "PreprocessResult",
    "deskew_angle",
    "detect_fold",
    "estimate_background",
    "normalise_illumination",
    "preprocess_page",
    "straighten_halves",
]
