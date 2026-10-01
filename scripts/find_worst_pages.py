"""Cherche les pires scans d'un split de La Liberté : pli marqué, forte inclinaison, bords sombres.

Les parquets déjà téléchargés par ``finlam-prod download`` sont relus depuis le cache ; chaque
page passe par ``detect_fold`` (seuils relâchés) et ``deskew_angle``. Les pages sont classées
par profondeur de bande au centre, puis par inclinaison, et les ``--top`` pires sont écrites
dans ``data/samples/`` avec un ``worst_pages.csv``.

    python scripts/find_worst_pages.py --split test --top 8
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    """Point d'entrée."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import cv2
    from datasets import load_dataset

    from finlam_prod.data.la_liberte import HF_DATASET, SPLIT_FILE_PREFIX
    from finlam_prod.preprocess import deskew_angle, detect_fold

    prefix = SPLIT_FILE_PREFIX.get(args.split, args.split)
    dataset = load_dataset(
        HF_DATASET,
        split=prefix,
        data_files={prefix: f"data/{prefix}-*.parquet"},
        verification_mode="no_checks",
    )
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(len(dataset), "pages")

    rows = []
    for i, row in enumerate(dataset):
        image = row.get("page_image")
        if image is None:
            continue
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        fold = detect_fold(gray, min_strength=2.0, min_depth=3.0)
        skew = deskew_angle(gray)
        dark_border = float(np.mean(gray[:, : gray.shape[1] // 50] < 60))
        rows.append(
            {
                "index": i,
                "page_id": row.get("page_id") or row.get("id") or f"row{i}",
                "fold_depth": fold.depth,
                "fold_strength": fold.strength,
                "fold_x": fold.x,
                "fold_detected": fold.detected,
                "skew_deg": skew,
                "dark_left_border": round(dark_border, 3),
            }
        )
        if i % 50 == 0:
            print(i, rows[-1])

    rows.sort(key=lambda r: (-abs(r["fold_depth"]), -abs(r["skew_deg"])))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "worst_pages.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nPires pages (profondeur de bande centrale, inclinaison) :")
    for r in rows[: args.top]:
        print(r)
        image = dataset[r["index"]]["page_image"].convert("RGB")
        name = f"worst_{args.split}_{abs(r['fold_depth']):04.1f}_{str(r['page_id'])[:8]}.jpg"
        image.save(args.out / name, quality=90)
        print("  →", args.out / name)
    most_skewed = max(rows, key=lambda r: abs(r["skew_deg"]))
    image = dataset[most_skewed["index"]]["page_image"].convert("RGB")
    pid = str(most_skewed["page_id"])[:8]
    name = f"skew_{args.split}_{abs(most_skewed['skew_deg']):.1f}_{pid}.jpg"
    image.save(args.out / name, quality=90)
    print("plus inclinée :", most_skewed, "→", args.out / name)


if __name__ == "__main__":
    main()
