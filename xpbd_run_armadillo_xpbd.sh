#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBD_DIR="${ROOT_DIR}/PositionBasedDynamics"

NODE_PATH="${PBD_DIR}/data/models/armadillo_4k.node"
ELE_PATH="${PBD_DIR}/data/models/armadillo_4k.ele"
VIS_PATH="${PBD_DIR}/data/models/armadillo.obj"

if [[ ! -f "${NODE_PATH}" || ! -f "${ELE_PATH}" ]]; then
  echo "Missing armadillo tet mesh under: ${PBD_DIR}/data/models" >&2
  exit 1
fi

SCENE_DIR="${ROOT_DIR}/out/xpbd_autoscenes"
SCENE_PATH="${SCENE_DIR}/Armadillo_XPBD.json"
mkdir -p "${SCENE_DIR}"

# Fix a thin top slice (2%) so it stays in place instead of rolling away.
STATIC_PARTICLES_JSON="$(python3 - "${NODE_PATH}" <<'PY'
import sys, json
from pathlib import Path

node_path = Path(sys.argv[1])
pts=[]
with node_path.open() as f:
    for line in f:
        s=line.strip()
        if not s or s.startswith("#"):
            continue
        n=int(s.split()[0])
        break
    for line in f:
        s=line.strip()
        if not s or s.startswith("#"):
            continue
        parts=s.split()
        if len(parts) < 4:
            continue
        pts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if len(pts) >= n:
            break

if not pts:
    print("[]")
    sys.exit(0)

ys=[p[1] for p in pts]
miny,maxy=min(ys),max(ys)
cut=maxy - (maxy-miny)*0.02
fixed=[i for i,(_,y,_) in enumerate(pts) if y>=cut]
if len(fixed)<64:
    order=sorted(range(len(pts)), key=lambda i: pts[i][1], reverse=True)
    fixed=order[:64]
print(f"[XPBD] fixedParticles(top-slice): {len(fixed)}/{len(pts)} (ycut={cut:g})", file=sys.stderr)
print(json.dumps(fixed))
PY)"

cat > "${SCENE_PATH}" <<EOF
{
  "Name": "Armadillo_XPBD",
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
    "solid_poissonRatio": 0.2,
    "solid_normalizeStretch": 0,
    "solid_normalizeShear": 0,
    "contactStiffnessRigidBody": 1.0,
    "contactStiffnessParticleRigidBody": 100.0,
    "gravitation": [0, -9.81, 0]
  },
  "cameraLookat": [0, 6, 0],
  "cameraPosition": [0, 12, 30],
  "TetModels": [
    {
      "id": 0,
      "nodeFile": "${NODE_PATH}",
      "eleFile": "${ELE_PATH}",
      "visFile": "${VIS_PATH}",
      "resolutionSDF": [20,20,20],
      "translation": [0, 10, 0],
      "rotationAxis": [0, 1, 0],
      "rotationAngle": 1.57,
      "scale": [2, 2, 2],
      "staticParticles": ${STATIC_PARTICLES_JSON},
      "restitution": 0.1,
      "friction": 0.3,
      "collisionObjectType": 5,
      "collisionObjectFileName": "",
      "collisionObjectScale": [2, 2, 2],
      "testMesh": 1
    }
  ],
  "RigidBodies": [
    {
      "id": 2,
      "geometryFile": "${PBD_DIR}/data/models/cube.obj",
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

echo "[XPBD] Scene: ${SCENE_PATH}"
echo "[XPBD] Tip: press Space to start/pause; Alt+LMB rotate; Shift+LMB pan; Ctrl+LMB zoom."

"${ROOT_DIR}/pbd_build_and_run_macos.sh" "${SCENE_PATH}"
