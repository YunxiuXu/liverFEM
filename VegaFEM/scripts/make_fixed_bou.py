#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path


AXIS_TO_I = {"x": 0, "y": 1, "z": 2}


def read_node_positions_in_file_order(path: Path) -> list[tuple[float, float, float]]:
    lines = path.read_text(errors="ignore").splitlines()
    header_i = None
    header = None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        header_i = i
        header = s.split()
        break
    if header_i is None or header is None:
        raise ValueError("Missing .node header.")

    num_points = int(float(header[0]))
    dim = int(float(header[1]))
    if dim < 3:
        raise ValueError(f"{path}: dim={dim}, expected 3")

    positions_by_id: dict[int, tuple[float, float, float]] = {}
    ids_in_order: list[int] = []
    for raw in lines[header_i + 1 :]:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        idx = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        if idx not in positions_by_id:
            ids_in_order.append(idx)
        positions_by_id[idx] = (x, y, z)
        if len(positions_by_id) >= num_points:
            break

    if len(positions_by_id) != num_points:
        raise ValueError(f"{path}: expected {num_points} vertices, got {len(positions_by_id)}")

    return [positions_by_id[i] for i in ids_in_order]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate VegaFEM fixed-vertex list (.bou) from a TetGen .node file.")
    ap.add_argument("node", help="Input .node path")
    ap.add_argument("out_bou", help="Output .bou path (1-indexed vertex ids)")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="y", help="Axis to pick a boundary band on")
    ap.add_argument("--side", choices=["min", "max"], default="min", help="Which side along the axis")
    ap.add_argument("--band_ratio", type=float, default=0.02, help="Thickness of the boundary band (fraction of bbox extent)")
    ap.add_argument(
        "--patch_ratio",
        type=float,
        default=0.15,
        help="Patch size on the other two axes (fraction of bbox extent, centered at bbox center)",
    )
    ap.add_argument("--min_count", type=int, default=12, help="Ensure at least this many fixed vertices")
    args = ap.parse_args()

    positions = read_node_positions_in_file_order(Path(args.node))
    axis_i = AXIS_TO_I[args.axis]
    other = [0, 1, 2]
    other.remove(axis_i)
    o0, o1 = other

    mins = [min(p[i] for p in positions) for i in range(3)]
    maxs = [max(p[i] for p in positions) for i in range(3)]
    ctrs = [(mins[i] + maxs[i]) * 0.5 for i in range(3)]
    ext = [maxs[i] - mins[i] for i in range(3)]

    band = ext[axis_i] * max(0.0, args.band_ratio)
    if args.side == "min":
        band_threshold = mins[axis_i] + band
        candidates = [i for i, p in enumerate(positions, start=1) if p[axis_i] <= band_threshold]
    else:
        band_threshold = maxs[axis_i] - band
        candidates = [i for i, p in enumerate(positions, start=1) if p[axis_i] >= band_threshold]

    if not candidates:
        raise SystemExit("No candidates found; try increasing --band_ratio.")

    half0 = ext[o0] * max(0.0, args.patch_ratio) * 0.5
    half1 = ext[o1] * max(0.0, args.patch_ratio) * 0.5
    patch = [
        i
        for i in candidates
        if abs(positions[i - 1][o0] - ctrs[o0]) <= half0 and abs(positions[i - 1][o1] - ctrs[o1]) <= half1
    ]

    if len(patch) < args.min_count:
        # Fallback: take closest vertices (within the band) to the patch center.
        cx, cy, cz = ctrs
        target = (cx, cy, cz)
        patch = sorted(
            candidates,
            key=lambda i: math.dist(positions[i - 1], target),
        )[: args.min_count]

    patch = sorted(set(patch))
    Path(args.out_bou).write_text(",".join(map(str, patch)) + ",\n")
    print(f"Wrote {args.out_bou} with {len(patch)} fixed vertices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

