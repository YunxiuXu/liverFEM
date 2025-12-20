#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def _read_tetgen_header(lines: list[str]) -> tuple[list[str], int]:
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        return s.split(), i
    raise ValueError("Missing TetGen header.")


def read_node(path: Path) -> tuple[list[tuple[float, float, float]], dict[int, int]]:
    lines = path.read_text(errors="ignore").splitlines()
    header, header_i = _read_tetgen_header(lines)
    num_points = int(float(header[0]))
    dim = int(float(header[1]))
    if dim < 3:
        raise ValueError(f"{path}: dim={dim}, expected 3")

    # TetGen indices can be 0-based (-z) or 1-based, and not necessarily contiguous.
    # Build an explicit remapping to 1..N for Vega.
    ids_in_order: list[int] = []
    positions_by_id: dict[int, tuple[float, float, float]] = {}
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

    id_to_new: dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(ids_in_order, start=1)}
    vertices = [positions_by_id[old_id] for old_id in ids_in_order]
    return vertices, id_to_new


def read_ele(path: Path, id_to_new: dict[int, int]) -> list[tuple[int, int, int, int]]:
    lines = path.read_text(errors="ignore").splitlines()
    header, header_i = _read_tetgen_header(lines)
    num_tets = int(float(header[0]))
    nodes_per_tet = int(float(header[1]))
    if nodes_per_tet != 4:
        raise ValueError(f"{path}: nodes_per_tet={nodes_per_tet}, expected 4")

    elements: list[tuple[int, int, int, int]] = []
    for raw in lines[header_i + 1 :]:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        old = [int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))]
        try:
            new = tuple(id_to_new[i] for i in old)
        except KeyError as e:
            raise ValueError(f"{path}: tet references missing node id: {e.args[0]}") from None
        elements.append(new)  # already 1-based sequential for Vega/OBJ
        if len(elements) >= num_tets:
            break
    if len(elements) != num_tets:
        raise ValueError(f"{path}: expected {num_tets} tets, got {len(elements)}")
    return elements


def write_veg(
    out_path: Path,
    vertices: list[tuple[float, float, float]],
    tets: list[tuple[int, int, int, int]],
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> None:
    with out_path.open("w") as f:
        f.write("# Vega mesh file.\n")
        f.write(f"# {len(vertices)} vertices, {len(tets)} elements\n\n")

        f.write("*VERTICES\n")
        f.write(f"{len(vertices)} 3 0 0\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            f.write(f"{i} {x} {y} {z}\n")
        f.write("\n")

        f.write("*ELEMENTS\n")
        f.write("TET\n")
        f.write(f"{len(tets)} 4 0\n")
        for i, (a, b, c, d) in enumerate(tets, start=1):
            # Keep TetGen's 1-based indices as-is.
            f.write(f"{i} {a} {b} {c} {d}\n")
        f.write("\n")

        f.write("*MATERIAL defaultMaterial\n")
        f.write(f"ENU, {density}, {youngs_modulus}, {poisson_ratio}\n\n")
        f.write("*REGION\n")
        f.write("allElements, defaultMaterial\n")


def write_surface_obj(out_path: Path, vertices: list[tuple[float, float, float]], tets: list[tuple[int, int, int, int]]) -> None:
    # A face is on the boundary if it appears in only one tet (ignoring orientation).
    face_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    face_oriented: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    def add_face(i: int, j: int, k: int) -> None:
        key = tuple(sorted((i, j, k)))
        face_counts[key] += 1
        # Keep first orientation seen; good enough for rendering.
        face_oriented.setdefault(key, (i, j, k))

    for (a, b, c, d) in tets:
        add_face(a, b, c)
        add_face(a, b, d)
        add_face(a, c, d)
        add_face(b, c, d)

    boundary_faces = [face_oriented[k] for k, cnt in face_counts.items() if cnt == 1]

    with out_path.open("w") as f:
        f.write("# Surface extracted from TetGen tets\n")
        for (x, y, z) in vertices:
            f.write(f"v {x} {y} {z}\n")
        for (i, j, k) in boundary_faces:
            f.write(f"f {i} {j} {k}\n")


def write_fixed_bou_from_min_y(out_path: Path, vertices: list[tuple[float, float, float]], band_ratio: float, min_count: int) -> int:
    ys = [v[1] for v in vertices]
    miny = min(ys)
    maxy = max(ys)
    band = (maxy - miny) * band_ratio
    threshold = miny + band

    fixed = [i for i, (_, y, _) in enumerate(vertices, start=1) if y <= threshold]
    if len(fixed) < min_count:
        fixed = sorted(range(1, len(vertices) + 1), key=lambda idx: vertices[idx - 1][1])[:min_count]
    fixed = sorted(set(fixed))

    out_path.write_text(",".join(map(str, fixed)) + ",\n")
    return len(fixed)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert TetGen .node/.ele to VegaFEM .veg (and optional surface .obj).")
    ap.add_argument("basename", help="TetGen basename (expects <basename>.node and <basename>.ele)")
    ap.add_argument("out_veg", help="Output .veg path")
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--E", type=float, default=1e8)
    ap.add_argument("--nu", type=float, default=0.45)
    ap.add_argument("--out_surface_obj", default=None, help="If set, writes extracted boundary surface as OBJ")
    ap.add_argument("--out_fixed_bou", default=None, help="If set, writes a simple fixed-vertices .bou")
    ap.add_argument("--fixed_band_ratio", type=float, default=0.02, help="Bottom band ratio for auto-fixed vertices")
    ap.add_argument("--fixed_min_count", type=int, default=8, help="Minimum fixed vertices")
    args = ap.parse_args()

    basename = Path(args.basename)
    node_path = basename.with_suffix(".node")
    ele_path = basename.with_suffix(".ele")

    vertices, id_to_new = read_node(node_path)
    tets = read_ele(ele_path, id_to_new)

    out_veg = Path(args.out_veg)
    write_veg(out_veg, vertices, tets, args.density, args.E, args.nu)

    if args.out_surface_obj:
        write_surface_obj(Path(args.out_surface_obj), vertices, tets)

    if args.out_fixed_bou:
        count = write_fixed_bou_from_min_y(Path(args.out_fixed_bou), vertices, args.fixed_band_ratio, args.fixed_min_count)
        print(f"Wrote {args.out_fixed_bou} with {count} fixed vertices.")

    print(f"Wrote {out_veg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
