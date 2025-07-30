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



class Vertex {
public:
	float x, y, z;
	const float initx, inity, initz; //初始化后不可更改
	int index;  // global index
	int localIndex; // 组内的本地索引
	float vertexMass; // mass of vertices
	float velx, vely, velz;//速度的三个分量
	bool isFixed;

	//Vertex(float x, float y, float z, int index) : x(x), y(y), z(z), index(index) {}
	Vertex(float x, float y, float z, int index)
		: initx(x), inity(y), initz(z), // 首先初始化const成员
		x(x), y(y), z(z), // 然后是可变成员
		index(index), vertexMass(1), // 其他成员可以直接赋值
		velx(0), vely(0), velz(0), // 初始化速度分量为0
		isFixed(false) // 默认不是固定点
	{}
	void setFixedIfBelowThreshold() {
		if (initx < -0.619) {//-0.619
			isFixed = true;
		}

	}
};

class Edge {
public:
	Vertex* vertices[2];
	bool isBoundary;

	Edge(Vertex* v1, Vertex* v2) : isBoundary(false) {
		vertices[0] = v1;
		vertices[1] = v2;
	}
};

class Tetrahedron {
public:
	Vertex* vertices[4];
	Edge* edges[6];  // Each tetrahedron has six edges
	float massTetra;
	float volumeTetra;
	Eigen::MatrixXf elementK;
	Eigen::MatrixXf elementKFEM;

	Tetrahedron(Vertex* v1, Vertex* v2, Vertex* v3, Vertex* v4) {
		vertices[0] = v1;
		vertices[1] = v2;
		vertices[2] = v3;
		vertices[3] = v4;
	}
	Eigen::MatrixXf createElementK(float E, float nu, const Eigen::Vector3f& groupCenterOfMass);
	float calMassTetra(float den);
	Eigen::MatrixXf createElementKFEM(float E, float nu);

};


class Group {
public:
	std::vector<Tetrahedron*> tetrahedra;
	std::unordered_map<int, Vertex*> verticesMap;
	std::vector<Vertex*> verticesVector;
	Eigen::Vector3f centerofMass;
	float groupMass;//每组的质量
	Eigen::MatrixXf massMatrix;//group mass matrix
	Eigen::MatrixXf massDistribution;
	Eigen::MatrixXf groupK;//group stiffness matrix
	Eigen::MatrixXf groupKFEM;
	Eigen::VectorXf primeVec;
	Eigen::VectorXf groupVelocity;
	Eigen::VectorXf groupVelocityFEM;
	Eigen::VectorXf groupExf;
	Eigen::MatrixXf rotationMatrix;
	Eigen::Matrix3f rotate_matrix;//3*3的旋转矩阵，扩展成组的旋转矩阵
	Eigen::VectorXf gravity;
	Eigen::MatrixXf dampingMatrix;
	Eigen::MatrixXf inverseTerm;
	Eigen::Vector3f initCOM;//initial center of mass
	Eigen::VectorXf initLocalPos;//initial position - center of mass
	Eigen::MatrixXf FEMLHS;
	Eigen::MatrixXf FEMLHS_Inv;
	Eigen::VectorXf FEMRHS;
	Eigen::VectorXf Fbind;
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
	Eigen::VectorXf currentPosition;//计算bindf用的位置信息，不用做位置更新
	Eigen::VectorXf currentPositionFEM;
	std::array<int, 6> adjacentGroupIDs;
	int groupIndex;//每组的编号
	std::vector<std::pair<std::vector<Vertex*>, std::vector<Vertex*>>> commonVerticesInDirections;//各个相邻组的共同点
	
	// XPBD Lambda accumulation storage: [adjacentGroupIdx][vertexPairIdx] -> Lambda_vector(3)
	std::map<int, std::map<int, Eigen::Vector3f>> constraintLambdas;

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
	Eigen::SparseMatrix<float> RHS_E, RHS_F, RHS_G;
	Eigen::VectorXf curLocalPos;
	Eigen::VectorXf RInvPos;
	bool gravityApplied = false;



	
	void addTetrahedron(Tetrahedron* tet);
	std::vector<Vertex*> getUniqueVertices();
	void calCenterofMass();
	void calMassGroup();
	Eigen::MatrixXf calMassMatrix(float den);
	void setVertexMassesFromMassMatrix();
	void calMassDistributionMatrix();
	void calGroupK(float E, float nu);
	void calPrimeVec(int w);
	void calPrimeVec1(int w);
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
	void calculateCurrentPositions();
	void calculateCurrentPositionsFEM();
	void calFbind(const Eigen::VectorXf& currentPositionThisGroup, const std::vector<Eigen::VectorXf>& allCurrentPositionsOtherGroups, float k);
	void updatePosition();
	void updatePositionFEM();
	void updateVelocity();
	void updateVelocityFEM();
	void initialize();
	void calPrimeVec2(int w);
	void calPrimeVec();
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


class Object {
public:
	std::vector<Group> groups; // change this
	//std::pair<std::vector<Vertex*>, std::vector<Vertex*>> commonPoints;
	
	int groupNum, groupNumX, groupNumY, groupNumZ;
	std::vector<Group> allGroup;

	Group& getGroup(int index);
	void findCommonVertices();// find common vertex
	void assignLocalIndicesToAllGroups(); // local Index
	void updateIndices();
	void generateUniqueVertices();//generate unique vertices
	void PBDLOOP(int looptime);
	void storeAllGroups();
	std::pair<std::vector<Vertex*>, std::vector<Vertex*>> findCommonVertices1(const Group& group1, const Group& group2);
	void updateAdjacentGroupIndices(int numX, int numY, int numZ);
	void calDistance(std::pair<std::vector<Vertex*>, std::vector<Vertex*>> commonpoints);
	void storeAdjacentGroupsCommonVertices(int groupIndex);

};

void findBoundaryEdges(tetgenio& out);
void divideIntoGroups(tetgenio& out, Object& object, int numX, int numY, int numZ);