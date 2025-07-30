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
#include "Group.h"





class Object {
public:
	std::vector<Group> groups; // change this
	//std::pair<std::vector<Vertex*>, std::vector<Vertex*>> commonPoints;

	int groupNum, groupNumX, groupNumY, groupNumZ;
	std::vector<Group> allGroup;
	float bodyVolume;

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
	void writeVerticesToFile(const std::string& filename);
	void fixTopLeft10PercentVertices();
};

void findBoundaryEdges(tetgenio& out);
void divideIntoGroups(tetgenio& out, Object& object, int numX, int numY, int numZ);