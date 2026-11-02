"""
Preprocess the custom Paomo dataset (ply + trim-point index txt) into the
Pointcept ScanNet-style layout, so it can be consumed directly by ScanNetDataset.

Input layout (dataset_root):
    train/<name>.ply   point cloud (xyz required; nx/ny/nz and rgb optional)
    train/<name>.txt   trim-point indices, one per line (empty file allowed)
    val/...            same as train

Output layout (output_root):
    train/<name>/coord.npy      (N, 3) float32, raw xyz
    train/<name>/color.npy      (N, 3) uint8, gray 127 if the ply has no color
    train/<name>/normal.npy     (N, 3) float32, zeros if the ply has no normals
    train/<name>/segment20.npy  (N,)   int32, 0 = non-trim, 1 = trim
    val/<name>/...              same as train

Usage:
    python pointcept/datasets/preprocessing/paomo/preprocess_paomo.py \
        --dataset_root <path/to/paomo_box_dataset2> \
        --output_root data/paomo
"""

import argparse
import glob
import os

import numpy as np

PLY_DTYPES = {
    "char": np.int8, "int8": np.int8,
    "uchar": np.uint8, "uint8": np.uint8,
    "short": np.int16, "int16": np.int16,
    "ushort": np.uint16, "uint16": np.uint16,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
}

COLOR_KEYS = (("red", "green", "blue"), ("r", "g", "b"))
NORMAL_KEYS = ("nx", "ny", "nz")


def read_ply_vertices(path):
    """Read the vertex element of a PLY file. Returns a structured array.

    Supports binary_little_endian and ascii, vertex-only (face data after the
    vertex block is ignored, so meshes stored as ply also work).
    """
    with open(path, "rb") as f:
        fmt, n_vertex, props = None, 0, []
        current_element = None
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                current_element, count = line.split()[1:3]
                if current_element == "vertex":
                    n_vertex = int(count)
            elif line.startswith("property") and current_element == "vertex":
                # list properties (e.g. face indices) are not expected on vertices
                ptype, pname = line.split()[-2:]
                props.append((pname, PLY_DTYPES[ptype]))
            elif line == "end_header":
                break

        if n_vertex == 0:
            raise ValueError(f"No vertex element found in {path}")
        dtype = np.dtype(props)

        if fmt == "binary_little_endian":
            vertices = np.fromfile(f, dtype=dtype, count=n_vertex)
        elif fmt == "ascii":
            rows = []
            for _ in range(n_vertex):
                values = f.readline().split()
                rows.append(tuple(float(v) for v in values[: len(props)]))
            vertices = np.array(rows, dtype=dtype)
        else:
            raise NotImplementedError(f"Unsupported ply format '{fmt}' in {path}")
    return vertices


def estimate_nn_distance(coords, max_sample=1024):
    """Mean nearest-neighbor distance on a subsample, used to suggest grid_size."""
    n = len(coords)
    if n < 2:
        return 0.0
    idx = np.random.choice(n, min(n, max_sample), replace=False)
    pts = coords[idx].astype(np.float64)
    best = np.full(len(pts), np.inf)
    chunk = 256
    for start in range(0, len(pts), chunk):
        block = pts[start : start + chunk]
        d2 = np.sum((block[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        best[start : start + chunk] = np.min(d2, axis=1)
    return float(np.mean(np.sqrt(best)))


def load_trim_indices(txt_path, n_points, index_base):
    if not os.path.isfile(txt_path) or os.path.getsize(txt_path) == 0:
        return np.zeros(0, dtype=np.int64)
    indices = np.loadtxt(txt_path, dtype=np.int64, ndmin=1) - index_base
    if indices.size == 0:
        return indices
    if indices.min() < 0 or indices.max() > n_points - 1:
        raise ValueError(
            f"{txt_path}: index range [{indices.min()}, {indices.max()}] is out of "
            f"bounds for {n_points} points with --index_base {index_base}. "
            f"If the txt is {'0' if index_base == 1 else '1'}-based, "
            f"rerun with --index_base {1 - index_base}."
        )
    return indices


def process_split(dataset_root, output_root, split, index_base):
    ply_paths = sorted(glob.glob(os.path.join(dataset_root, split, "*.ply")))
    if not ply_paths:
        print(f"[{split}] no ply files found, skipped")
        return None

    stats = dict(
        scenes=0, points=0, trim=0, no_txt=0, empty=0,
        nn_dists=[], bbox_diags=[], no_color=0, no_normal=0,
    )
    for i, ply_path in enumerate(ply_paths):
        name = os.path.splitext(os.path.basename(ply_path))[0]
        txt_path = os.path.splitext(ply_path)[0] + ".txt"
        vertices = read_ply_vertices(ply_path)
        n = len(vertices)
        fields = vertices.dtype.names

        coord = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
        coord = coord.astype(np.float32)

        if all(k in fields for k in NORMAL_KEYS):
            normal = np.stack([vertices[k] for k in NORMAL_KEYS], axis=1)
            normal = normal.astype(np.float32)
        else:
            normal = np.zeros((n, 3), dtype=np.float32)
            stats["no_normal"] += 1

        color_key = next(
            (keys for keys in COLOR_KEYS if all(k in fields for k in keys)), None
        )
        if color_key is not None:
            color = np.stack([vertices[k] for k in color_key], axis=1)
            if color.dtype != np.uint8:  # e.g. float color stored in [0, 1]
                color = np.clip(color, 0, 1) * 255
            color = color.astype(np.uint8)
        else:
            color = np.full((n, 3), 127, dtype=np.uint8)
            stats["no_color"] += 1

        segment = np.zeros(n, dtype=np.int32)
        if not os.path.isfile(txt_path):
            stats["no_txt"] += 1
        trim_idx = load_trim_indices(txt_path, n, index_base)
        if trim_idx.size == 0:
            stats["empty"] += 1
        segment[trim_idx] = 1

        out_dir = os.path.join(output_root, split, name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "coord.npy"), coord)
        np.save(os.path.join(out_dir, "color.npy"), color)
        np.save(os.path.join(out_dir, "normal.npy"), normal)
        np.save(os.path.join(out_dir, "segment20.npy"), segment)

        stats["scenes"] += 1
        stats["points"] += n
        stats["trim"] += trim_idx.size
        stats["bbox_diags"].append(
            float(np.linalg.norm(coord.max(0) - coord.min(0)))
        )
        # if len(stats["nn_dists"]) < 50:  # sample a subset for speed
        stats["nn_dists"].append(estimate_nn_distance(coord))

        if (i + 1) % 100 == 0:
            print(f"[{split}] {i + 1}/{len(ply_paths)} scenes processed")
    return stats


def report(split, stats):
    if stats is None:
        return
    trim_ratio = stats["trim"] / max(stats["points"], 1)
    print(f"\n===== {split} =====")
    print(f"scenes            : {stats['scenes']}")
    print(f"points            : {stats['points']} "
          f"(avg {stats['points'] // max(stats['scenes'], 1)} per scene)")
    print(f"trim points       : {stats['trim']} ({100 * trim_ratio:.2f}% of all points)")
    print(f"scenes w/o txt    : {stats['no_txt']} (treated as all non-trim)")
    print(f"scenes w/o trim   : {stats['empty']}")
    print(f"scenes w/o color  : {stats['no_color']} (filled with gray 127)")
    print(f"scenes w/o normal : {stats['no_normal']} (filled with zeros)")
    print(f"bbox diagonal     : median {np.median(stats['bbox_diags']):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", default="/root/autodl-tmp/paomo_box", required=True,
        help="Path to the raw paomo dataset containing train/ and val/ folders",
    )
    parser.add_argument(
        "--output_root", default="/root/autodl-tmp/paomo_box_0928",required=True,
        help="Output root, e.g. data/paomo (train/ and val/ will be created)",
    )
    parser.add_argument(
        "--index_base", type=int, default=0, choices=[0, 1],
        help="Whether the txt indices are 0-based (default) or 1-based",
    )

    args = parser.parse_args(
        [
        "--dataset_root", "/root/autodl-tmp/paomo_dataset_1009",
        "--output_root", "/root/autodl-tmp/paomo_dataset",
        "--index_base", "0",
    ]
    )

    np.random.seed(0)
    for split in ["train", "val"]:
        stats = process_split(
            args.dataset_root, args.output_root, split, args.index_base
        )
        report(split, stats)
