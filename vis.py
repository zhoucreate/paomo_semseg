"""
Visualize a paomo scene before and after the training transform pipeline,
using the real Pointcept transforms from pointcept.datasets.transform.

Run inside the training environment (requires torch + pointcept):

python vis.py \
    --data_root /root/autodl-tmp/paomo_dataset_v1 \
    --split train --name 1_0 \
    --out_dir vis/transform_before_after \
    --seed 0 --grid_size 0.3

Points are colored by ground-truth label:
    non-trim (0) = gray, trim (1) = red

Open the two .ply files with CloudCompare / MeshLab / Open3D.
"""

import argparse
import os

import numpy as np

from pointcept.datasets.transform import Compose

GRAY = (190, 190, 190)
RED = (230, 30, 30)

# The exact transform chain from the config, up to (but not including)
# ToTensor / Collect, because we want numpy arrays for PLY export.
TRANSFORM = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(
        type="GridSample",
        grid_size=0.1,  # overridden by --grid_size
        hash_type="fnv",
        mode="train",
        return_min_coord=True,
    ),
    # dict(type="SphereCrop", sample_rate=0.6, mode="random"),
    dict(type="CenterShift", apply_z=False),
    dict(type="ShufflePoint"),
]


def write_ply(path, coord, segment):
    n = coord.shape[0]
    color = np.zeros((n, 3), dtype=np.uint8)
    color[segment == 0] = GRAY
    color[segment == 1] = RED
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.concatenate([coord, color.astype(np.float32)], axis=1)
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/paomo_dataset_v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out_dir", default="vis/transform_before_after")
    parser.add_argument("--grid_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    scene_dir = os.path.join(args.data_root, args.split, args.name)
    coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(scene_dir, "color.npy")).astype(np.float32)
    normal = np.load(os.path.join(scene_dir, "normal.npy")).astype(np.float32)
    segment = (
        np.load(os.path.join(scene_dir, "segment20.npy"))
        .reshape(-1)
        .astype(np.int32)
    )

    os.makedirs(args.out_dir, exist_ok=True)
    before_path = os.path.join(args.out_dir, f"{args.name}_before.ply")
    after_path = os.path.join(args.out_dir, f"{args.name}_after.ply")

    # before: raw point cloud
    n_before = coord.shape[0]
    write_ply(before_path, coord, segment)

    # after: run the real Pointcept transform chain
    transform_cfg = [dict(t) for t in TRANSFORM]
    for t in transform_cfg:
        if t["type"] == "GridSample":
            t["grid_size"] = args.grid_size
    transform = Compose(transform_cfg)

    data_dict = {
        "coord": coord,
        "color": color,
        "normal": normal,
        "segment": segment,
    }
    data_dict = transform(data_dict)
    coord = data_dict["coord"]
    segment = data_dict["segment"]
    n_after = coord.shape[0]
    write_ply(after_path, coord, segment)

    print(f"before: {n_before} points -> {before_path}")
    print(f"after : {n_after} points -> {after_path}")
    print(
        f"removed {n_before - n_after} points "
        f"({100 * (n_before - n_after) / max(n_before, 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
