This is the current research program I'm doing for doctor course. In this research, we divided the model into several groups and arrange each group local coordinates. Every group has its own rotation and solve its equations independently.
The formulation of this method is based on Linear Corotated FEM and Shape Matching for each group, and then binded together using a constraint force.
This method currently accelerates the Linear Corotated FEM (VegaFEM) by 7 times and keeps good accuracy. Our method can be implemented in various shapes.
For example, a Stanford Armadillo with right hand fixed and then dropped down by gravity.
We show better accuracy than an other corotated method, Operator Splitting, although they are quite fast.
The image compares dragging a beam with a fixed left side. The beige one is VegaFEM, the blue one is our method and the grey blue is Operator Splitting.

Our method can also simulate anisotropic materials by re-formulating the stiffness matrix.

For more details, you can refer to:
[vrsj_localFEM.pdf](https://github.com/user-attachments/files/15973518/vrsj_localFEM.pdf)

Thanks for seeing my research!
