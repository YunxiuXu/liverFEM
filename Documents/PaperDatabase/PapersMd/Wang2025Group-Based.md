---
tags:
  - method/simulation
  - topic/modeling
  - topic/biomechanics
  - topic/stiffness
  - medium/vr
---

![[Wang2025Group-Based.png|640]]

## 摘要
本文提出了一种新颖的局部线性共轭有限元法，通过直接求解和迭代求解的混合策略来解决实时仿真中大变形的计算效率与物理精度之间的矛盾。我们的方法将仿真域划分为元素组；组内的动力学通过预先计算的直接方法来求解，而组间的耦合通过高效的迭代约束来处理。这种域划分实现了大量预计算，将运行时的方程求解替换为快速的矩阵-向量乘法。实验结果表明，该方法在速度与精度之间取得更好的平衡，相较于同等速度的方法显示出更高的保真度。该方法支持各向异性材料，在近似不可压缩的条件下实现体积变化只有 2.95% 的误差。此外，其基于组的架构在多核处理器上显示出较高的可扩展性，在 64 线程下对一个含 12 万四面体的模型将计算时间降至 19.6 ms。这些特性使该方法非常适合用于如外科手术仿真、触觉反馈系统和实时数字孪生等对精度和性能均有高要求的应用。

# Group-Based Corotational FEM for Real-Time Large Deformation Simulation

**作者:** Siyu Wang, Yunxiu Xu, Shoichi Hasegawa

## Abstract
Real-time simulation of large deformations in soft bodies has long faced a trade-off between computational efficiency and physical accuracy. This paper presents a novel Local Linear Corotated Finite Element Method that addresses this challenge through a hybrid strategy combining direct and iterative solvers. Our method decomposes the simulation domain into element groups; the dynamics within each group are resolved using a pre-computed direct method, while inter-group interactions are handled by efficient iterative constraints. This domain decomposition enables significant pre-computation, replacing runtime equation solving with fast matrix-vector multiplications. Experimental results demonstrate that the method achieves a better balance of speed and accuracy, demonstrating higher fidelity than similarly fast methods. The method supports anisotropic materials and achieves low volume change (2.95% error) under nearly incompressible conditions. Furthermore, its group-based architecture exhibits high scalability on multi-core processors, reducing computation time to 19.6 ms for a 120,000-tetrahedron model using 64 threads. These characteristics make our method well-suited for demanding applications such as surgical simulation, haptic feedback systems, and real-time digital twins, where both accuracy and performance are critical.

[阅读原文](../Pdf/Wang2025Group-Based.pdf)
