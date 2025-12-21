#pragma once
#include "Vertex.h"
#include "Edge.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "tetgen.h"
#include <cstring> 
#include <fstream>
#include <iostream>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Eigen/Sparse"
#include "GMRES.h"
#include <Eigen/Dense>
#include "Eigen/IterativeLinearSolvers"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

class Tetrahedron {
public:
	Vertex* vertices[4];
	Edge* edges[6];  // Each tetrahedron has six edges
	float massTetra;
	float volumeTetra;
	float lastStress = 0.0f;
	Eigen::Matrix3f invDm; // Inverse of rest shape matrix for strain calculation
	Eigen::MatrixXf elementK;
	Eigen::MatrixXf elementKFEM;

	Tetrahedron(Vertex* v1, Vertex* v2, Vertex* v3, Vertex* v4) {
		vertices[0] = v1;
		vertices[1] = v2;
		vertices[2] = v3;
		vertices[3] = v4;
	}
	Eigen::MatrixXf createElementK(float E, float nu, const Eigen::Vector3f& groupCenterOfMass);
	Eigen::MatrixXf createElementKAni(float E1, float E2, float E3, float nu, const Eigen::Vector3f& groupCenterOfMass);
	float calMassTetra(float den);
	float calVolumeTetra();
	Eigen::MatrixXf createElementKFEM(float E, float nu);

};