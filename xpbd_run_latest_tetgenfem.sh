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

# Basic integrity checks: header count vs actual rows.
python3 - <<'PY'
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
PY "${NODE_PATH}" "${ELE_PATH}"

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
    "numberOfStepsPerRenderUpdate": 1,
    "subSteps": 5,
    "maxIterations": 5,
    "maxIterationsV": 5,
    "velocityUpdateMethod": 0,
    "contactTolerance": 0.05,
    "solidSimulationMethod": 3,
    "contactStiffnessRigidBody": 1.0,
    "contactStiffnessParticleRigidBody": 100.0,
    "solid_poissonRatio": 0.2,
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
      "translation": [0, 4, 0],
      "rotationAxis": [1, 0, 0],
      "rotationAngle": 0.0,
      "scale": [1, 1, 1],
      "staticParticles": [],
      "restitution": 0.1,
      "friction": 0.2,
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
      "restitution": 0.6,
      "friction": 0.0,
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
