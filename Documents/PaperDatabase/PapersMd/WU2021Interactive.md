---
tags:
  - topic/haptics-overview
  - method/simulation
  - topic/modeling
  - topic/biomechanics
  - actuator/friction-modulation
---

![[WU2021Interactive.png|640]]

## 摘要
摘要
背景
肝切除涉及肝脏部分的外科切除。它用于治疗肝肿瘤和肝损伤。这一手术的复杂性和高风险性使初学医生难以在真实患者身上练习。开发了虚拟手术仿真，以模拟外科过程，使医疗专业人员无需患者、尸体或动物就能接受培训。因此，需要开发肝切除手术仿真系统。我们提出一个实时仿真系统，为肝实质切割提供逼真的视觉和触觉反馈。
方法
四面体结构和簇基形匹配用于物理模型构建、三维肝模型的拓扑更新、软变形仿真和触觉渲染加速。在肝实质分离仿真过程中，采用四面体网格对表面三角形进行细分并生成手术创面的表面。利用四面体网格构建的无向图，通过组件检测实现形状匹配簇的分离。
结果
在我们的系统中，簇基形状匹配在GPU上实现，而触觉渲染与拓扑更新在CPU上实现。实验结果表明，触觉渲染可以在很高的频率下完成（>900Hz），网格皮肤化和图形渲染可在45fps完成。拓扑更新可在单个CPU线程上以>10Hz的交互速率执行。
结论
我们提出了一种基于四面体结构的交互式肝实质切除仿真方法。该四面体网格同时支持物理模型构建、拓扑更新和触觉渲染加速。
关键词
虚拟手术；肝实质切除；基于位置的动力学

# Interactive hepatic parenchymal transection simulation with haptic feedback

**作者:** Hongyu WU, Haonan YU, Fan YE, Jian SUN, Yuan GAO, Ke TAN, Aimin HAO

## Abstract
Abstract
Background
Liver resection involves surgical removal of a portion of the liver. It is used to
treat liver tumors and liver injuries. The complexity and high-risk nature of this surgery prevents novice
doctors from practicing it on real patients. Virtual surgery simulation was developed to simulate surgical
procedures to enable medical professionals to be trained without requiring a patient, a cadaver, or an
animal. Therefore, there is a strong need for the development of a liver resection surgery simulation
system. We propose a real-time simulation system that provides realistic visual and tactile feedback for
hepatic parenchymal transection. Methods
The tetrahedron structure and cluster-based shape matching
are used for physical model construction, topology update of a three-dimensional liver model soft
deformation simulation, and haptic rendering acceleration. During the liver parenchyma separation
simulation, a tetrahedral mesh is used for surface triangle subdivision and surface generation of the
surgical wound. The shape-matching cluster is separated via component detection on an undirected graph
constructed using the tetrahedral mesh. Results
In our system, cluster-based shape matching is
implemented on a GPU, whereas haptic rendering and topology updates are implemented on a CPU.
Experimental results show that haptic rendering can be performed at a high frequency (>900Hz), whereas
mesh skinning and graphics rendering can be performed at 45fps. The topology update can be executed at
an interactive rate (>10Hz) on a single CPU thread. Conclusions
We propose an interactive hepatic
parenchymal transection simulation method based on a tetrahedral structure. The tetrahedral mesh
simultaneously supports physical model construction, topology update, and haptic rendering acceleration.
Keywords
Virtual surgery; Hepatic parenchymal transection; Position-based dynamics

[阅读原文](../Pdf/WU2021Interactive.pdf)
