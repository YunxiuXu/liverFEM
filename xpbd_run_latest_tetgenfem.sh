#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the export location created when running TetgenFEM from its own directory.
NODE_PATH="${ROOT_DIR}/TetgenFEM/out/tetgenfem_exports/latest.node"
ELE_PATH="${ROOT_DIR}/TetgenFEM/out/tetgenfem_exports/latest.ele"

# Fallback (if you exported to repo-root out/)
if [[ ! -f "${NODE_PATH}" || ! -f "${ELE_PATH}" ]]; then
  NODE_PATH="${ROOT_DIR}/out/tetgenfem_exports/latest.node"
  ELE_PATH="${ROOT_DIR}/out/tetgenfem_exports/latest.ele"
fi

if [[ ! -f "${NODE_PATH}" || ! -f "${ELE_PATH}" ]]; then
  echo "Cannot find TetgenFEM export files." >&2
  echo "Expected one of:" >&2
  echo "  ${ROOT_DIR}/TetgenFEM/out/tetgenfem_exports/latest.node + latest.ele" >&2
  echo "  ${ROOT_DIR}/out/tetgenfem_exports/latest.node + latest.ele" >&2
  exit 1
fi

# Compute fixed (static) particle indices similar to TetgenFEM's liver anchoring:
# - Compute bounds
# - Take the back-most 12% slice in Z, compute centroid
# - Push slightly inward (2% of depth)
# - Fix all vertices within radius = 20% of depth
# Note: In bash, '#' starts a comment only if it's the first char of a word.
XPBD_FIX_AXIS="${XPBD_FIX_AXIS:-z}"                  # x|y|z
XPBD_FIX_SLICE_FRAC="${XPBD_FIX_SLICE_FRAC:-0.12}"   # TetgenFEM: 0.12
XPBD_FIX_INWARD_FRAC="${XPBD_FIX_INWARD_FRAC:-0.02}" # TetgenFEM: 0.02
XPBD_FIX_RADIUS_FRAC="${XPBD_FIX_RADIUS_FRAC:-0.2}"  # TetgenFEM: 0.2
XPBD_FIX_MIN_FRAC="${XPBD_FIX_MIN_FRAC:-0.01}"       # auto-expand until >= this fraction fixed
XPBD_FIX_MIN_ABS="${XPBD_FIX_MIN_ABS:-32}"           # ...and at least this many
XPBD_FIX_AUTO_EXPAND="${XPBD_FIX_AUTO_EXPAND:-1}"    # 1: enable, 0: disable

STATIC_PARTICLES_JSON="$(python3 - "${NODE_PATH}" "${XPBD_FIX_AXIS}" "${XPBD_FIX_SLICE_FRAC}" "${XPBD_FIX_INWARD_FRAC}" "${XPBD_FIX_RADIUS_FRAC}" "${XPBD_FIX_MIN_FRAC}" "${XPBD_FIX_MIN_ABS}" "${XPBD_FIX_AUTO_EXPAND}" <<'PY'
import sys, json
from pathlib import Path

node_path = Path(sys.argv[1])
axis = (sys.argv[2] or "z").lower()
slice_frac = float(sys.argv[3])
inward_frac = float(sys.argv[4])
radius_frac = float(sys.argv[5])
min_frac = float(sys.argv[6])
min_abs = int(float(sys.argv[7]))
auto_expand = int(float(sys.argv[8])) != 0

points = []  # in file order (0..n-1)
with node_path.open() as f:
    # header
    for line in f:
        s=line.strip()
        if not s or s.startswith("#"):
            continue
        header=s.split()
        n=int(header[0])
        break
    # body
    for line in f:
        s=line.strip()
        if not s or s.startswith("#"):
            continue
        parts=s.split()
        if len(parts) < 4:
            continue
        points.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if len(points) >= n:
            break

if not points:
    print("[]")
    sys.exit(0)

minx=min(p[0] for p in points); maxx=max(p[0] for p in points)
miny=min(p[1] for p in points); maxy=max(p[1] for p in points)
minz=min(p[2] for p in points); maxz=max(p[2] for p in points)

axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
mins = [minx, miny, minz]
maxs = [maxx, maxy, maxz]
depth = maxs[axis_idx] - mins[axis_idx]

back_slice = mins[axis_idx] + depth * slice_frac
back_pts = [p for p in points if p[axis_idx] <= back_slice]
if back_pts:
    cx=sum(p[0] for p in back_pts)/len(back_pts)
    cy=sum(p[1] for p in back_pts)/len(back_pts)
    cz=sum(p[2] for p in back_pts)/len(back_pts)
else:
    cx=0.5*(minx+maxx); cy=0.5*(miny+maxy); cz=minz

center = [cx, cy, cz]
center[axis_idx] = min(center[axis_idx] + depth * inward_frac, maxs[axis_idx])
cx, cy, cz = center

radius = max(depth * radius_frac, 1e-6)
r2 = radius * radius

def select(radius_val: float):
    r2 = radius_val * radius_val
    out=[]
    for i,(x,y,z) in enumerate(points):
        dx=x-cx; dy=y-cy; dz=z-cz
        if dx*dx + dy*dy + dz*dz <= r2:
            out.append(i)  # 0-based indices expected by PBD scenes
    return out

fixed = select(radius)

target = max(min_abs, int(len(points) * min_frac))
if auto_expand and (len(fixed) < target) and (depth > 0.0):
    # Increase radius until we have enough fixed points (keeps "TetgenFEM-like" behavior but avoids an empty/tiny anchor).
    max_radius = depth * 0.6
    while (len(fixed) < target) and (radius < max_radius):
        radius *= 1.25
        fixed = select(radius)

if not fixed:
    # last-resort: fix a thin top slice in Y so something is always anchored
    yr = maxy - miny
    ycut = maxy - yr * 0.02
    fixed = [i for i,(_,y,_) in enumerate(points) if y >= ycut]

print(f"[XPBD] fixedParticles: {len(fixed)}/{len(points)} (axis={axis}, depth={depth:g}, slice={slice_frac:g}, radius={radius:g})", file=sys.stderr)
print(json.dumps(fixed))
PY)"

# Basic integrity checks: header count vs actual rows.
python3 - "${NODE_PATH}" "${ELE_PATH}" <<'PY'
import sys
from pathlib import Path

node_path = Path(sys.argv[1])
ele_path = Path(sys.argv[2])

def first_data_line(path: Path):
    with path.open() as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith("#"):
                continue
            return s
    return None

def count_data_lines(path: Path):
    n=0
    with path.open() as f:
        for line in f:
            s=line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    return n

node_header = first_data_line(node_path)
ele_header = first_data_line(ele_path)
if node_header is None or ele_header is None:
    print(f"[XPBD] ERROR: empty .node or .ele file", file=sys.stderr)
    sys.exit(2)

try:
    node_n = int(node_header.split()[0])
    ele_m = int(ele_header.split()[0])
except Exception:
    print(f"[XPBD] ERROR: cannot parse header counts", file=sys.stderr)
    sys.exit(2)

node_lines = count_data_lines(node_path) - 1  # exclude header
ele_lines = count_data_lines(ele_path) - 1

if node_lines != node_n:
    print(f"[XPBD] ERROR: .node rows mismatch: header={node_n}, actual={node_lines}", file=sys.stderr)
    print(f"        file: {node_path}", file=sys.stderr)
    sys.exit(2)

if ele_lines != ele_m:
    print(f"[XPBD] ERROR: .ele rows mismatch: header={ele_m}, actual={ele_lines}", file=sys.stderr)
    print(f"        file: {ele_path}", file=sys.stderr)
    print(f"        This usually means the export is incomplete/truncated, so most vertices become free particles (looks like fluid).", file=sys.stderr)
    sys.exit(2)

print(f"[XPBD] TetGen files look consistent: {node_n} nodes, {ele_m} tets")
PY

SCENE_DIR="${ROOT_DIR}/out/xpbd_autoscenes"
SCENE_PATH="${SCENE_DIR}/TetgenFEM_LatestScene.json"
mkdir -p "${SCENE_DIR}"

PBD_MODELS_DIR="${ROOT_DIR}/PositionBasedDynamics/data/models"

# Note: We omit "visFile" on purpose so PBD renders the tet surface mesh derived from .node/.ele.
cat > "${SCENE_PATH}" <<EOF
{
  "Name": "TetgenFEM_Latest",
  "Simulation": {
    "timeStepSize": 0.01,
    "numberOfStepsPerRenderUpdate": 4,
    "subSteps": 5,
    "maxIterations": 1,
    "maxIterationsV": 5,
    "velocityUpdateMethod": 0,
    "contactTolerance": 0.0,
    "solidSimulationMethod": 3,
    "solid_stiffness": 1.0,
    "solid_volumeStiffness": 1.0,
    "contactStiffnessRigidBody": 1.0,
    "contactStiffnessParticleRigidBody": 100.0,
    "gravitation": [0, -9.81, 0],
    "solid_poissonRatio": 0.28,
    "solid_normalizeStretch": 0,
    "solid_normalizeShear": 0
  },
  "cameraLookat": [0, 0, 0],
  "cameraPosition": [0, 6, 20],
  "TetModels": [
    {
      "id": 0,
      "nodeFile": "${NODE_PATH}",
      "eleFile": "${ELE_PATH}",
      "translation": [0, 6, 0],
      "rotationAxis": [1, 0, 0],
      "rotationAngle": 0.0,
      "scale": [1, 1, 1],
      "staticParticles": ${STATIC_PARTICLES_JSON},
      "restitution": 0.0,
      "friction": 0.3,
      "collisionObjectType": 0,
      "collisionObjectFileName": "",
      "collisionObjectScale": [1, 1, 1],
      "testMesh": 1
    }
  ],
  "RigidBodies": [
    {
      "id": 1,
      "geometryFile": "${PBD_MODELS_DIR}/cube.obj",
      "flatShading": true,
      "isDynamic": 0,
      "density": 500,
      "translation": [0, 0, 0],
      "rotationAxis": [1, 0, 0],
      "rotationAngle": 0.0,
      "scale": [100, 1, 100],
      "restitution": 0.0,
      "friction": 0.4,
      "collisionObjectType": 2,
      "collisionObjectScale": [100, 1, 100]
    }
  ],
  "BallJoints": [],
  "BallOnLineJoints": [],
  "DamperJoints": [],
  "DistanceJoints": [],
  "HingeJoints": [],
  "UniversalJoints": [],
  "SliderJoints": [],
  "TargetAngleMotorHingeJoints": [],
  "TargetVelocityMotorHingeJoints": [],
  "TargetPositionMotorSliderJoints": [],
  "TargetVelocityMotorSliderJoints": [],
  "RigidBodyParticleBallJoints": []
}
EOF

echo "[XPBD] Using node: ${NODE_PATH}"
echo "[XPBD] Using ele:  ${ELE_PATH}"
echo "[XPBD] Scene:      ${SCENE_PATH}"

"${ROOT_DIR}/pbd_build_and_run_macos.sh" "${SCENE_PATH}"
