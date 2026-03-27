---
tags:
  - topic/modeling
  - method/simulation
  - method/optimization
---

![[Bouaziz2014Projective.png|640]]

## 摘要
我们提出了一种用于物理系统隐式时间积分的新方法。我们的方法在节点有限元方法和基于位置的动力学之间架起了一座桥梁，从而得到一个简单、高效、鲁棒且精确的求解器，能够支持多种不同类型的约束。我们提出了通过交替优化方法高效求解的专门设计的能量势。受连续介质力学的启发，我们推导出一组可高效嵌入到求解器中的连续介质基础势。我们在从固体、布料、壳体到基于示例的仿真等多种应用中展示了该方法的通用性和鲁棒性。与牛顿法为基础的求解器以及基于位置的动力学求解器的比较，突出显示了我们公式的优势。

# Projective Dynamics: Fusing Constraint Projections for Fast Simulation

**作者:** Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, Mark Pauly

## Abstract
We present a new method for implicit time integration of physical systems. Our approach builds a bridge between nodal Finite Element methods and Position Based Dynamics, leading to a simple, efficient, robust, yet accurate solver that supports many different types of constraints. We propose specially designed energy potentials that can be solved efficiently using an alternating optimization approach. Inspired by continuum mechanics, we derive a set of continuum-based potentials that can be efficiently incorporated within our solver. We demonstrate the generality and robustness of our approach in many different applications ranging from the simulation of solids, cloths, and shells, to example-based simulation. Comparisons to Newton-based and Position Based Dynamics solvers highlight the benefits of our formulation.

[阅读原文](../Pdf/Bouaziz2014Projective.pdf)
