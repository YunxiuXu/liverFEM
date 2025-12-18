# XPBD vs TetFEM 参数对照表

## 实验 1 参数配置

### 求解器迭代次数
| 参数 | TetFEM | XPBD |
|------|---------|------|
| Fast 迭代数 | 30 | 30 |
| Reference 迭代数 | 150 | 150 |

### 实验流程参数
| 参数 | TetFEM | XPBD |
|------|---------|------|
| Settle 步数 | 120 | 120 |
| LoadRamp 步数 | 240 | 240 |
| Hold 步数 | 240 | 240 |
| Pull 加速度 | 800/1500/2000 | 800/1500/2000 |
| 影响半径 | 0.6 | 0.6 |
| 拉力方向 | +X | +X |

### 物理材料参数（注意：XPBD 和 FEM 的参数含义不同！）

| 参数类型 | TetFEM | XPBD | 说明 |
|----------|---------|------|------|
| **求解方法** | Corotational FEM | XPBD (Position-Based) | **完全不同的求解器** |
| **杨氏模量 E** | 1,000,000 Pa | N/A | XPBD 不直接使用杨氏模量 |
| **泊松比 ν** | 0.28 | 0.28 | 相同 |
| **刚度参数** | 直接由 E 计算 | `stiffness=1.0` | XPBD 的 stiffness 是 compliance 倒数，**不等于** E |
| **密度** | 1000 kg/m³ | 默认 1.0 | 通过质量间接影响 |
| **重力** | -10.0 (实验时禁用) | -9.81 (实验时禁用) | 实验时都设为 0 |
| **时间步长** | 0.01 | 0.01 | 相同 |

## 关键差异说明

### 1. **求解器本质不同**
- **TetFEM（Corotational FEM）**：
  - 基于连续介质力学的有限元方法
  - 使用杨氏模量 E 和泊松比 ν 直接计算刚度矩阵
  - 隐式求解线性系统 \( Kx = f \)
  
- **XPBD（Extended Position-Based Dynamics）**：
  - 基于约束的位置修正方法
  - 使用 compliance（柔度）参数 \( \alpha = 1/k \)
  - 不直接对应杨氏模量，而是通过约束刚度 \( k \) 间接控制

### 2. **XPBD Stiffness 与杨氏模量的关系**

XPBD 的 `solid_stiffness` 参数 **不等于** 杨氏模量！它是约束刚度的缩放因子。

根据 PBD 论文，XPBD 的约束刚度 \( k \) 与杨氏模量 \( E \) 的粗略关系为：
\[
k \approx \frac{E \cdot V}{L^2}
\]
其中 \( V \) 是四面体体积，\( L \) 是特征长度。

**在本实验中**：
- 我们不追求刚度的绝对数值匹配（因为两种方法的物理意义不同）
- 而是比较**相对精度**：在相同拉力下，Fast 和 Reference 迭代数的位移差异
- 这样可以公平地评估两种求解器在有限迭代次数下的收敛性能

### 3. **实验目的澄清**

本实验 **不是** 比较 XPBD 和 TetFEM 的精度差异（它们是不同的求解器）。

实验目的是：
- **TetFEM 内部对比**：Fast (30 iters) vs Reference (150 iters)
- **XPBD 内部对比**：Fast (30 iters) vs Reference (150 iters)

分别证明两种方法在实时迭代数（30 次）下都能接近收敛解。

## 输出数据对比

| 数据文件 | 位置 | 内容 |
|----------|------|------|
| TetFEM | `out/experiment1/<timestamp>/` | positions, sweep_summary, metadata |
| XPBD | `out/experiment1/<timestamp>/` | positions, sweep_summary, metadata |

### CSV 格式一致性
两者的 CSV 文件格式完全一致，方便后续分析脚本复用。

## 运行方式

### TetFEM
```bash
./xpbd_run_latest_tetgenfem.sh
# 然后在 UI 中点击 "START EXP1" 按钮
```

### XPBD
```bash
./xpbd_run_exp1.sh
# 然后在 UI 中点击 "Start Experiment 1" 按钮
```

## 注意事项

1. **固定顶部**：两者都固定模型顶部 2% 的粒子，防止整体下落
2. **禁用重力**：实验期间自动禁用重力，只施加水平拉力
3. **目标顶点选择**：使用相同的确定性算法（Z 值最大的前 80%，X 值最大者）
4. **力场施加**：都使用径向衰减的力场，中心粒子加权 1.5 倍
