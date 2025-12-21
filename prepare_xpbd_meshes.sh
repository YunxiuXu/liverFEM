#!/bin/bash

# 检查输入参数
if [ -z "$1" ]; then
    echo "使用方法: ./prepare_xpbd_meshes.sh <Proposed_Method_Output_Dir>"
    echo "例如: ./prepare_xpbd_meshes.sh out/experiment4/20251221_123456"
    exit 1
fi

INPUT_DIR=$1
MODELS_DIR="models"
XPBD_SCENE_DIR="PositionBasedDynamics/data/scenes"

echo "开始同步网格到 XPBD..."

# 1. 拷贝网格文件到 models 目录
for target in 5000 20000 50000; do
    # 查找对应的 node 和 ele 文件
    NODE_FILE=$(ls $INPUT_DIR/target${target}*.node 2>/dev/null | head -n 1)
    ELE_FILE=$(ls $INPUT_DIR/target${target}*.ele 2>/dev/null | head -n 1)
    
    if [ -f "$NODE_FILE" ] && [ -f "$ELE_FILE" ]; then
        cp "$NODE_FILE" "$MODELS_DIR/liver_target${target}.node"
        cp "$ELE_FILE" "$MODELS_DIR/liver_target${target}.ele"
        echo "已拷贝 target ${target} 网格"
        
        # 2. 生成 XPBD JSON 场景文件
        SCENE_FILE="$XPBD_SCENE_DIR/Liver_target${target}_XPBD.json"
        cat > "$SCENE_FILE" <<EOF
{
  "Name": "Liver_target${target}_XPBD",
  "Simulation": {
    "timeStepSize": 0.01,
    "numberOfStepsPerRenderUpdate": 1,
    "subSteps": 5,
    "maxIterations": 1,
    "maxIterationsV": 5,
    "velocityUpdateMethod": 0,
    "contactTolerance": 0.0,
    "solidSimulationMethod": 3,
    "solid_stiffness": 1.0,
    "solid_volumeStiffness": 1.0,
    "solid_poissonRatio": 0.28,
    "solid_normalizeStretch": 0,
    "solid_normalizeShear": 0,
    "contactStiffnessRigidBody": 1.0,
    "contactStiffnessParticleRigidBody": 100.0,
    "gravitation": [0, 0, 0]
  },
  "TetModels": [
    {
      "id": 0,
      "nodeFile": "../models/liver_target${target}.node",
      "eleFile": "../models/liver_target${target}.ele",
      "visFile": "../models/liver_HD_Low_surface.obj",
      "translation": [0, 0, 0],
      "scale": [1, 1, 1],
      "testMesh": 1
    }
  ]
}
EOF
        echo "已生成场景文件: $SCENE_FILE"
    else
        echo "错误: 未能在 $INPUT_DIR 中找到 target ${target} 的网格文件"
    fi
done

echo "同步完成！"
