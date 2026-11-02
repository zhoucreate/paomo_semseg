"""
Export paomo test predictions to colored PLY files for visualization.

Each <name>_pred.npy in the result dir holds per-point predicted labels that
are already inverse-mapped back to the original point order, i.e. they align
1:1 with data_root/<split>/<name>/coord.npy. Ground truth is read from
segment20.npy in the same folder.

Modes:
    pred     color by prediction   (trim = red,    non-trim = gray)
    gt       color by ground truth (trim = blue,   non-trim = gray)
    compare  error analysis        (TP = red, FN = blue, FP = yellow, TN = gray)

Usage:
    python tools/visualize_paomo.py \
        --data_root data/paomo --split val \
        --pred_dir exp/paomo/semseg-pt-v1-0-base_paomo/result \
        --out_dir vis/paomo --mode compare

Open the exported .ply files with CloudCompare / MeshLab / Open3D.
"""

import argparse
import glob
import os

import numpy as np

GRAY = (190, 190, 190)
COLORS = {
    "pred": {0: GRAY, 1: (230, 30, 30)},
    "gt": {0: GRAY, 1: (30, 110, 230)},
    # (gt, pred): TP red, FN blue, FP yellow, TN gray
    "compare": {(1, 1): (230, 30, 30), (1, 0): (30, 80, 230),
                (0, 1): (240, 220, 40), (0, 0): GRAY},
}


def write_ply(path, coord, color):
    n = coord.shape[0]
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.concatenate([coord, color], axis=1)
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")


def colorize(mode, pred, gt):
    table = COLORS[mode]
    color = np.zeros((pred.shape[0], 3), dtype=np.uint8)
    if mode == "compare":
        for (g, p), rgb in table.items():
            color[(gt == g) & (pred == p)] = rgb
    else:
        labels = pred if mode == "pred" else gt
        for label, rgb in table.items():
            color[labels == label] = rgb
    return color


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/paomo",
                        help="Pointcept-format dataset root")
    parser.add_argument("--split", default="val")
    parser.add_argument("--pred_dir", required=True,
                        help="Folder containing <name>_pred.npy from tools/test.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", default="compare",
                        choices=["pred", "gt", "compare"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_paths = sorted(glob.glob(os.path.join(args.pred_dir, "*_pred.npy")))
    if not pred_paths:
        raise SystemExit(f"No *_pred.npy found in {args.pred_dir}")

    n_scenes, skipped = 0, 0
    total = {"tp": 0, "fn": 0, "fp": 0}
    for pred_path in pred_paths:
        name = os.path.basename(pred_path)[: -len("_pred.npy")]
        scene_dir = os.path.join(args.data_root, args.split, name)
        coord_path = os.path.join(scene_dir, "coord.npy")
        if not os.path.isfile(coord_path):
            print(f"[skip] {name}: {coord_path} not found")
            skipped += 1
            continue

        coord = np.load(coord_path).astype(np.float64)
        pred = np.load(pred_path).reshape(-1).astype(np.int64)
        if pred.shape[0] != coord.shape[0]:
            print(f"[skip] {name}: pred has {pred.shape[0]} labels but "
                  f"coord has {coord.shape[0]} points")
            skipped += 1
            continue

        gt_path = os.path.join(scene_dir, "segment20.npy")
        gt = (np.load(gt_path).reshape(-1).astype(np.int64)
              if os.path.isfile(gt_path) else np.zeros(coord.shape[0], np.int64))

        color = colorize(args.mode, pred, gt)
        out_path = os.path.join(args.out_dir, f"{name}_{args.mode}.ply")
        write_ply(out_path, coord, color)
        n_scenes += 1

        if args.mode == "compare":
            total["tp"] += int(((gt == 1) & (pred == 1)).sum())
            total["fn"] += int(((gt == 1) & (pred == 0)).sum())
            total["fp"] += int(((gt == 0) & (pred == 1)).sum())

    print(f"\nExported {n_scenes} scenes to {args.out_dir} (skipped {skipped})")
    if args.mode == "compare" and (total["tp"] + total["fn"]) > 0:
        recall = total["tp"] / (total["tp"] + total["fn"])
        prec = total["tp"] / max(total["tp"] + total["fp"], 1)
        print(f"trim points: TP={total['tp']} FN={total['fn']} FP={total['fp']} "
              f"| recall={recall:.3f} precision={prec:.3f}")
