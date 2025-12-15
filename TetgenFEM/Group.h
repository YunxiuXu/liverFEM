#pragma once
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
#include "Vertex.h"
#include "Edge.h"
#include "Tetrahedron.h"
#include "params.h"

class Group {
public:
	float groupVolume;
	std::vector<Tetrahedron*> tetrahedra;
	std::unordered_map<int, Vertex*> verticesMap;
	std::vector<Vertex*> verticesVector;
	Eigen::Vector3f centerofMass;
	float groupMass;
	Eigen::MatrixXf massMatrix;//group mass matrix
	Eigen::MatrixXf massDistribution;
	Eigen::MatrixXf groupK;//group stiffness matrix
	Eigen::MatrixXf groupKFEM;
	Eigen::VectorXf primeVec;
	Eigen::VectorXf groupVelocity;
	Eigen::VectorXf groupVelocityFEM;
	Eigen::VectorXf groupExf;
	Eigen::MatrixXf rotationMatrix;
	Eigen::Matrix3f rotate_matrix;
	Eigen::VectorXf gravity;
	Eigen::MatrixXf dampingMatrix;
	Eigen::MatrixXf inverseTerm;
	Eigen::Vector3f initCOM;//initial center of mass
	Eigen::VectorXf initLocalPos;//initial position - center of mass
	Eigen::MatrixXf FEMLHS;
	Eigen::MatrixXf FEMLHS_Inv;
	Eigen::VectorXf FEMRHS;
	Eigen::VectorXf Fbind;
	Eigen::VectorXf prevFbind;
	Eigen::VectorXf deltaX;
	Eigen::VectorXf deltaXFEM;
	Eigen::SparseMatrix<float> rotationSparse;
	Eigen::SparseMatrix<float> rotationTransSparse;//R^t sparse
	Eigen::SparseMatrix<float> kSparse;
	Eigen::SparseMatrix<float> kSparseFEM;
	Eigen::SparseMatrix<float> massSparse;
	Eigen::SparseMatrix<float> dampingSparse;
	Eigen::SparseMatrix<float> massDistributionSparse;
	Eigen::SparseMatrix<float> massDampingSparseInv; //(M+C').inv sparse
	Eigen::SparseMatrix<float> inverseTermSparse;
	Eigen::VectorXf currentPosition;//?�Zbindf�p�I�ʒu�M���C�s�p��ʒu�X�V
	Eigen::VectorXf currentPositionFEM;
	Eigen::VectorXf rotatedPrimeVec;
	Eigen::VectorXf rotatedFbind;
	Eigen::VectorXf distancesX; 
	std::array<int, 6> adjacentGroupIDs;
	int groupIndex;
	std::vector<std::pair<std::vector<Vertex*>, std::vector<Vertex*>>> commonVerticesInDirections;//�e����??�I�����_
	
	// XPBD Lambda accumulation storage: [adjacentGroupIdx][vertexPairIdx] -> Lambda_vector(3)
	std::map<int, std::map<int, Eigen::Vector3f>> constraintLambdas;
	
	// Debug mode for constraint monitoring (set to true to enable debug output)
	static bool debug_constraints;

	Eigen::MatrixXf LHS_I;
	Eigen::MatrixXf LHS_A;
	Eigen::MatrixXf LHS_B;
	Eigen::MatrixXf LHS_C;
	Eigen::MatrixXf LHSFEM;
	Eigen::VectorXf RHSFEM;
	Eigen::VectorXf RHS_A;
	Eigen::VectorXf RHS_B;
	Eigen::VectorXf RHS_C;
	Eigen::VectorXf RHS_D, RHS_AsubBplusC;
	Eigen::SparseMatrix<float> RHS_E, RHS_F, RHS_G, RHS_F_MassD;
	Eigen::VectorXf curLocalPos;
	Eigen::VectorXf RInvPos;
	bool gravityApplied = false;

	float Kinematics;



	void addTetrahedron(Tetrahedron* tet);
	std::vector<Vertex*> getUniqueVertices();
	void calCenterofMass();
	void calMassGroup();
	Eigen::MatrixXf calMassMatrix(float den);
	void setVertexMassesFromMassMatrix();
	void calMassDistributionMatrix();
	void calGroupK(float E, float nu);
	void calGroupKAni(float E1, float E2, float E3, float nu);
	void calPrimeVec(int w);
	void calPrimeVec1(int w);
	void calPrimeVecT(int w);
	void calPrimeVecS(const std::vector<int>& topVertexLocalIndices, const std::vector<int>& bottomVertexLocalIndices);
	void calDampingMatrix();
	void calInitCOM();
	void calRotationMatrix();
	void calLocalPos();//calculate initial local position
	Eigen::Vector3f axlAPD(Eigen::Matrix3f a);
	Eigen::Vector3f clamp2(Eigen::Vector3f x, float y, float z);
	Eigen::Quaternionf Exp2(Eigen::Vector3f a);
	void calLHS();
	void calLHSFEM();
	void calRHS();
	void calRHSFEM();
	void calDeltaX();
	void calDeltaXFEM();
	void applyRotation(const Eigen::VectorXf& src, Eigen::VectorXf& dst) const;
	void applyRotationTranspose(const Eigen::VectorXf& src, Eigen::VectorXf& dst) const;
	void calculateCurrentPositions();
	void calculateCurrentPositionsFEM();
	void calFbind(const Eigen::VectorXf& currentPositionThisGroup, const std::vector<Eigen::VectorXf>& allCurrentPositionsOtherGroups, float k);
	void updatePosition();
	void updatePositionFEM();
	void updateVelocity();
	void updateVelocityFEM();
	void initialize();
	void calPrimeVec2(int w);
	void calPrimeVec(const Eigen::Vector3f& externalForce = Eigen::Vector3f::Zero());
	//void updateVertexPositions();
	void calFbind1(const std::vector<Vertex*>& commonVerticesGroup1,
		const std::vector<Vertex*>& commonVerticesGroup2, 
		const Eigen::VectorXf& currentPositionGroup1, 
		const Eigen::VectorXf& currentPositionGroup2, 
		const Eigen::VectorXf& velGroup1, 
		const Eigen::VectorXf& velGroup2, 
		const Eigen::MatrixXf& massMatrixGroup1,
		const Eigen::MatrixXf& massMatrixGroup2,
		float youngs, 
		float hardness, 
		float dampingBeta, 
		int adjacentGroupIdx);
	void calRInvLocalPos();
	void calGroupKFEM(float E, float nu);

	void calBindFixed();

	Group()
		: centerofMass(Eigen::Vector3f::Zero()),  // Initialize Eigen vector
		groupMass(0.0),                         // Initialize float
		verticesMap(),
		adjacentGroupIDs({ -1, -1, -1, -1, -1, -1 }),
		commonVerticesInDirections(6)

	{
		// Additional initialization logic, if needed
	}
};
