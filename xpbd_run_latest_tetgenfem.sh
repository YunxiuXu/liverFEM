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
    "tetModelSimulationMethod": 2,
    "clothSimulationMethod": 2,
    "clothBendingMethod": 2,
    "contactStiffnessRigidBody": 1.0,
    "contactStiffnessParticleRigidBody": 100.0,
    "solid_stiffness": 1.0,
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

