---
tags:
  - method/simulation
  - topic/modeling
  - topic/biomechanics
  - topic/shape
  - method/optimization
---

![[Bui2019Corotational.png|640]]

## 摘要
我们提出用于实时外科手术仿真的共回转截断有限元法（Corotational CutFEM）。该方法的唯一要求是一个背景网格，该网格不必与所模拟对象的边界/界面严格一致。表面的细节可以直接从二值图像获得，并通过多级嵌入算法对被表面裁切的背景网格单元进行处理。表面上的Dirichlet边界条件可以通过拉格朗日乘子在表面上隐式施加，而在表面某些部分施加的牵引或Neumann边界条件可以通过形函数分配给背景节点。该实现通过几何与数值解的收敛性研究进行验证，显示出最佳的收敛速率。为了验证方法的可靠性，该方法被应用于多种针插入仿真（如活检或近前列放射治疗）到脑部和肝脏模型。数值结果表明，在保持与标准FEM相同的精度的同时，所提出的方法可以（1）使离散化与几何描述无关，（2）避免复杂几何体的网格生成复杂性，以及（3）提供适合实时仿真的计算速度。因此，该方法非常适合面向患者特定的仿真，在自动且恰当地考虑被仿真的几何形状的同时，保持较低的计算成本。

# Corotational cut finite element method for real-time surgical simulation: Application to needle insertion simulation

**作者:** Huu Phuoc Bui, Satyendra Tomar, Stéphane P.A. Bordas

## Abstract
We present the corotational cut Finite Element Method (FEM) for real-time surgical simulation. The only requirement of the proposed method is a background mesh, which is not necessarily conforming to the boundaries/interfaces of the simulated object. The details of the surface, which can be directly obtained from binary images, are taken into account by a multilevel embedding algorithm which is applied to elements of the background mesh that are cut by the surface. Dirichlet boundary conditions can be implicitly imposed on the surface using Lagrange multipliers, whereas traction or Neumann boundary conditions, which is/are applied on parts of the surface, can be distributed to the background nodes using shape functions. The implementation is verified by convergences studies, of the geometry and of numerical solutions, which exhibit optimal rates. To verify the reliability of the method, it is applied to various needle insertion simulations (e.g. for biopsy or brachytherapy) into brain and liver models. The numerical results show that, while retaining the accuracy of the standard FEM, the proposed method can (1) make the discretisation independent from geometric description, (2) avoid the complexity of mesh generation for complex geometries, and (3) provide computational speed suitable for real-time simulations. Thereby, the proposed method is very suitable for patient-specific simulations as it improves the simulation accuracy by automatically, and properly, taking the simulated geometry into account, while keeping the low computational cost.

[阅读原文](../Pdf/Bui2019Corotational.pdf)
