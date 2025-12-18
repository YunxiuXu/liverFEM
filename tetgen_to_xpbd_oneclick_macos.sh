#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TETFEM_BUILD_DIR="${ROOT_DIR}/build/tetgenfem"
PBD_SCENE_DIR="${ROOT_DIR}/out/tetgenfem_exports"
PBD_SCENE_PATH="${PBD_SCENE_DIR}/TetgenFEM_LatestScene.json"

mkdir -p "${PBD_SCENE_DIR}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Install it first." >&2
  exit 1
fi

# 1) Build TetgenFEM
cmake -S "${ROOT_DIR}" -B "${TETFEM_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${TETFEM_BUILD_DIR}" -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"

# 2) Export latest.node/ele to repo-root out/
"${TETFEM_BUILD_DIR}/TetgenFEM" --export-tetgen --export-dir "${PBD_SCENE_DIR}"

if [[ ! -f "${PBD_SCENE_DIR}/latest.node" || ! -f "${PBD_SCENE_DIR}/latest.ele" ]]; then
  echo "Missing exported TetGen files:" >&2
  echo "  ${PBD_SCENE_DIR}/latest.node" >&2
  echo "  ${PBD_SCENE_DIR}/latest.ele" >&2
  exit 1
fi

PBD_MODELS_DIR="${ROOT_DIR}/PositionBasedDynamics/data/models"

# 3) Generate a PBD scene that loads latest.node/ele
cat > "${PBD_SCENE_PATH}" <<EOF
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
      "nodeFile": "latest.node",
      "eleFile": "latest.ele",
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

echo "[OneClick] PBD scene: ${PBD_SCENE_PATH}"

# 4) Build & run PBD SceneLoaderDemo on that scene
"${ROOT_DIR}/pbd_build_and_run_macos.sh" "${PBD_SCENE_PATH}"

