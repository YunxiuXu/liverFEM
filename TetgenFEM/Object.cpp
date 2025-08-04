#include "Object.h"
#include "params.h"
#include "Edge.h"
#include "Group.h"








std::unordered_set<std::string> boundaryEdgesSet;  // Set to store boundary edges

void findBoundaryEdges(tetgenio& out) {
	int indexOffset = out.firstnumber;  // Get the index offset (0 or 1)
	for (int i = 0; i < out.numberoftrifaces; ++i) {
		for (int j = 0; j < 3; ++j) {
			int vertexIndex1 = out.trifacelist[i * 3 + j] - indexOffset;
			int vertexIndex2 = out.trifacelist[i * 3 + ((j + 1) % 3)] - indexOffset;
			std::string edgeKey = vertexIndex1 < vertexIndex2 ?
				std::to_string(vertexIndex1) + "-" + std::to_string(vertexIndex2) :
				std::to_string(vertexIndex2) + "-" + std::to_string(vertexIndex1);
			boundaryEdgesSet.insert(edgeKey);
		}
	}
}

void divideIntoGroups(tetgenio& out, Object& object, int numX, int numY, int numZ) {
	//findBoundaryEdges(out);  // Populate the boundaryEdgesSet

	// Find min and max coordinates in all directions
	float minX = out.pointlist[0], minY = out.pointlist[1], minZ = out.pointlist[2];
	float maxX = minX, maxY = minY, maxZ = minZ;
	for (int i = 0; i < out.numberofpoints; ++i) {
		float x = out.pointlist[i * 3];
		float y = out.pointlist[i * 3 + 1];
		float z = out.pointlist[i * 3 + 2];
		if (x < minX) minX = x; if (x > maxX) maxX = x;
		if (y < minY) minY = y; if (y > maxY) maxY = y;
		if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
	}

	// Calculate group ranges for each direction
	float groupRangeX = (maxX - minX) / numX;
	float groupRangeY = (maxY - minY) / numY;
	float groupRangeZ = (maxZ - minZ) / numZ;

	// Create vertices
	std::vector<Vertex*> vertices;
	for (int i = 0; i < out.numberofpoints; ++i) {
		float x = out.pointlist[i * 3];
		float y = out.pointlist[i * 3 + 1];
		float z = out.pointlist[i * 3 + 2];
		vertices.push_back(new Vertex(x, y, z, i));
	}

	// Resize groups vector
	object.groups.resize(numX * numY * numZ);

	// Create tetrahedra and assign to groups based on XYZ coordinates
	for (int i = 0; i < out.numberoftetrahedra; ++i) {
		Vertex* v1 = vertices[out.tetrahedronlist[i * 4] - 1];
		Vertex* v2 = vertices[out.tetrahedronlist[i * 4 + 1] - 1];
		Vertex* v3 = vertices[out.tetrahedronlist[i * 4 + 2] - 1];
		Vertex* v4 = vertices[out.tetrahedronlist[i * 4 + 3] - 1];
		Tetrahedron* tet = new Tetrahedron(v1, v2, v3, v4); // Pack vertices into tetrahedron

		// Determine group based on average coordinates
		float avgX = (v1->x + v2->x + v3->x + v4->x) / 4;
		float avgY = (v1->y + v2->y + v3->y + v4->y) / 4;
		float avgZ = (v1->z + v2->z + v3->z + v4->z) / 4;

		int groupIndexX = std::min(static_cast<int>((avgX - minX) / groupRangeX), numX - 1);
		int groupIndexY = std::min(static_cast<int>((avgY - minY) / groupRangeY), numY - 1);
		int groupIndexZ = std::min(static_cast<int>((avgZ - minZ) / groupRangeZ), numZ - 1);

		int groupIdx = groupIndexZ * numX * numY + groupIndexY * numX + groupIndexX;
		object.groups[groupIdx].addTetrahedron(tet); // Pack tetrahedron into group
		object.groups[groupIdx].groupIndex = groupIdx;

		// Set up edges for each tetrahedron
		static int edgeIndices[6][2] = { {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3} };
		for (int j = 0; j < 6; ++j) {
			Vertex* vertex1 = tet->vertices[edgeIndices[j][0]];
			Vertex* vertex2 = tet->vertices[edgeIndices[j][1]];
			Edge* edge = new Edge(vertex1, vertex2);
			std::string edgeKey = vertex1->index < vertex2->index ?
				std::to_string(vertex1->index) + "-" + std::to_string(vertex2->index) :
				std::to_string(vertex2->index) + "-" + std::to_string(vertex1->index);
			edge->isBoundary = boundaryEdgesSet.count(edgeKey) > 0;
			tet->edges[j] = edge;
		}
	}
}



void Object::assignLocalIndicesToAllGroups() { // local index generation
	for (Group& group : groups) {
		int currentLocalIndex = 0;
		std::unordered_set<Vertex*> processedVertices; // 用于跟踪已处理的顶点

		for (Tetrahedron* tetra : group.tetrahedra) {
			for (int i = 0; i < 4; ++i) {
				Vertex* vertex = tetra->vertices[i];

				// 检查顶点是否已经处理过
				if (processedVertices.find(vertex) == processedVertices.end()) {
					vertex->localIndex = currentLocalIndex++; // 分配本地
					processedVertices.insert(vertex); // 标记为已处理
				}
			}
		}
	}
}

void Object::updateIndices() {
	std::unordered_set<int> globalIndices;
	std::unordered_map<int, Vertex*> indexToVertexMap; // 旧索引到新顶点的映射
	int maxIndex = 0;

	// 首先遍历所有顶点以找到最大索引值
	for (Group& group : groups) {
		for (Tetrahedron* tetra : group.tetrahedra) {
			for (int i = 0; i < 4; ++i) {
				Vertex* vertex = tetra->vertices[i];
				maxIndex = std::max(maxIndex, vertex->index);
			}
		}
	}

	int nextAvailableIndex = maxIndex + 1;

	// 再次遍历所有顶点以更新索引
	for (Group& group : groups) {
		std::unordered_set<int> localIndices; // 每个组内的本地索引集合

		for (Tetrahedron* tetra : group.tetrahedra) {
			for (int i = 0; i < 4; ++i) {
				Vertex* vertex = tetra->vertices[i];

				if (localIndices.find(vertex->index) == localIndices.end()) { //如果在 localIndices 集合中找不到 vertex->index 的值
					localIndices.insert(vertex->index);

					if (globalIndices.find(vertex->index) != globalIndices.end()) {
						// 如果索引已在全局集合中，创建新顶点并更新映射
						Vertex* newVertex = new Vertex(vertex->x, vertex->y, vertex->z, nextAvailableIndex++);
						indexToVertexMap[vertex->index] = newVertex;
						tetra->vertices[i] = newVertex;
						vertex = newVertex;
					}
					globalIndices.insert(vertex->index);
				}
				else if (indexToVertexMap.find(vertex->index) != indexToVertexMap.end()) {
					// 更新为新的顶点引用
					tetra->vertices[i] = indexToVertexMap[vertex->index];
				}
			}
		}
	}
}

void Object::generateUniqueVertices() { //执行这个函数以后，verticesMap就会装满这个组的vertices， 不重复
	//std::vector<Vertex*> uniqueVertices;

	for (Group& group : groups) {
		group.verticesMap.clear(); // 清空现有的映射

		for (Tetrahedron* tetra : group.tetrahedra) {
			for (int i = 0; i < 4; ++i) {
				Vertex* vertex = tetra->vertices[i];

				// 如果顶点尚未在verticesMap中，则添加
				if (group.verticesMap.find(vertex->index) == group.verticesMap.end()) {
					group.verticesMap[vertex->index] = vertex;
				}
			}
		}
		group.initialize();
	}

}

std::pair<std::vector<Vertex*>, std::vector<Vertex*>> Object::findCommonVertices1(const Group& group1, const Group& group2) { //寻找共同点
	std::vector<Vertex*> commonVerticesGroup1;
	std::vector<Vertex*> commonVerticesGroup2;

	// 遍历group1的verticesMap中的所有顶点
	for (auto& mapEntry1 : group1.verticesMap) {
		Vertex* vertex1 = mapEntry1.second;

		// 遍历group2的verticesMap中的所有顶点
		for (auto& mapEntry2 : group2.verticesMap) {
			Vertex* vertex2 = mapEntry2.second;

			// 检查坐标是否相同
			if (vertex1->x == vertex2->x && vertex1->y == vertex2->y && vertex1->z == vertex2->z) {
				commonVerticesGroup1.push_back(vertex1);
				commonVerticesGroup2.push_back(vertex2);
			}
		}
	}

	return { commonVerticesGroup1, commonVerticesGroup2 };
}

void Object::findCommonVertices() {
	// Assuming 'groups' is a member of Object class and contains all groups
	for (Group& group : groups) {
		// Initialize commonVerticesInDirections for the current group
		//group.commonVerticesInDirections = std::vector<std::pair<std::vector<Vertex*>, std::vector<Vertex*>>>(6);

		// Iterate through all possible directions
		for (int direction = 0; direction < 6; ++direction) {
			int adjacentGroupIdx = group.adjacentGroupIDs[direction];

			// Check if there is an adjacent group in this direction
			if (adjacentGroupIdx != -1) {
				Group& adjacentGroup = groups[adjacentGroupIdx];

				std::vector<Vertex*> commonVerticesCurrentGroup;
				std::vector<Vertex*> commonVerticesAdjacentGroup;

				// Find common vertices between group and adjacentGroup
				for (auto& vertexCurrent : group.verticesMap) {
					for (auto& vertexAdjacent : adjacentGroup.verticesMap) {
						if (vertexCurrent.second->x == vertexAdjacent.second->x &&
							vertexCurrent.second->y == vertexAdjacent.second->y &&
							vertexCurrent.second->z == vertexAdjacent.second->z) {
							commonVerticesCurrentGroup.push_back(vertexCurrent.second);
							commonVerticesAdjacentGroup.push_back(vertexAdjacent.second);
						}
					}
				}

				// Store the common vertices in the appropriate direction
				group.commonVerticesInDirections[direction] = { commonVerticesCurrentGroup, commonVerticesAdjacentGroup };
			}
		}
	}
}

void Object::storeAdjacentGroupsCommonVertices(int groupIndex) {
	// 确保指定的组索引在有效范围内
	if (groupIndex < 0 || groupIndex >= groups.size()) {
		std::cerr << "Invalid group index." << std::endl;
		return;
	}

	// 存储共同顶点的结构，每个条目对应一个方向的相邻组
	//std::vector<std::pair<std::vector<Vertex*>, std::vector<Vertex*>>> commonVerticesInDirections(6);

	// 获取当前组
	Group& currentGroup = groups[groupIndex];

	// 遍历所有6个方向的相邻组
	for (int direction = 0; direction < 6; ++direction) {
		int adjacentGroupIdx = currentGroup.adjacentGroupIDs[direction];

		// 检查是否存在相邻组
		if (adjacentGroupIdx != -1) {
			Group& adjacentGroup = groups[adjacentGroupIdx];

			// 使用 findCommonVertices1 函数找到共同顶点
			std::pair<std::vector<Vertex*>, std::vector<Vertex*>> commonVertices = findCommonVertices1(currentGroup, adjacentGroup);

			// 存储找到的共同顶点
			currentGroup.commonVerticesInDirections[direction] = commonVertices;
		}
	}
}
void Object::calDistance(std::pair<std::vector<Vertex*>, std::vector<Vertex*>> commonPoints) {
	const auto& verticesGroup1 = commonPoints.first;
	const auto& verticesGroup2 = commonPoints.second;

	// 确保两组点的数量相等
	if (verticesGroup1.size() != verticesGroup2.size()) {
		// 处理错误情况或返回
		return;
	}

	// 遍历顶点对并计算距离
	for (size_t i = 0; i < verticesGroup1.size(); ++i) {
		Vertex* vertex1 = verticesGroup1[i];
		Vertex* vertex2 = verticesGroup2[i];

		// 计算两点之间的欧几里得距离
		double distance = std::sqrt(std::pow(vertex1->x - vertex2->x, 2) +
			std::pow(vertex1->y - vertex2->y, 2) +
			std::pow(vertex1->z - vertex2->z, 2));


		std::cout << "Distance of: " << i << "is" << distance << std::endl;
	}
}

void Object::writeVerticesToFile(const std::string& filename) {
	std::ofstream outfile(filename, std::ios::out);
	if (!outfile) {
		std::cerr << "无法打开文件！" << std::endl;
		return;
	}

	std::unordered_set<int> writtenIndices;  // 用于存储已写入的顶点索引

	for (const auto& group : groups) {  // 遍历所有组
		for (const auto& vertex : group.verticesVector) {  // 遍历组内所有顶点
			if (vertex != nullptr && writtenIndices.find(vertex->index) == writtenIndices.end()) {
				// 如果顶点索引尚未写入，则写入文件
				outfile << "Index: " << vertex->index
					<< ", X: " << vertex->x
					<< ", Y: " << vertex->y
					<< ", Z: " << vertex->z << std::endl;
				writtenIndices.insert(vertex->index);  // 标记索引为已写入
			}
		}
	}

	outfile.close();
	std::cout << "顶点数据已成功写入文件: " << filename << std::endl;
}
void Object::PBDLOOP(int looptime) {


		 //#pragma omp parallel for
	float reference = 0.0f; // float类型的参考值
	float epsilon = std::numeric_limits<float>::epsilon(); // float类型的epsilon
#pragma omp parallel for
	for (int i = 0; i < groupNum; ++i) {
		auto& g = groups[i];
		//g.Fbind = 0.75 * g.prevFbind;
		g.Fbind = Eigen::VectorXf::Zero(3 * g.verticesMap.size()); // 假设 Group 类有一个方法来清除 Fbind
		g.rotationTransSparse = g.rotationMatrix.transpose().sparseView(reference, epsilon);

		g.RHS_F = g.RHS_E * g.rotationTransSparse;//RHS的部分
		g.RHS_B = g.RHS_F * g.primeVec; //46ms

		//g.RHS_F_MassD = g.RHS_F * g.massDistributionSparse;

		//auto fff = g.RHS_F.toDense();
		//auto massssss = g.massDistributionSparse.toDense();

		//Eigen::MatrixXf producttt = (10000*fff) * (10000*massssss);
		//auto aa = g.RHS_F_MassD.toDense();

		//g.RHS_C = g.RHS_F_MassD * g.primeVec; //54ms
		g.RHS_G = timeStep * timeStep * g.massDampingSparseInv * g.rotationTransSparse;
		g.RHS_AsubBplusC = g.RHS_A - g.RHS_B;// +g.RHS_C; //24ms

	}

	

	for (int iter = 0; iter < looptime; ++iter) {
		

#pragma omp parallel for //500fps to 300, -optimization
		for (int i = 0; i < groupNum; ++i) {
			auto& g = groups[i];
			g.calRHS();
			g.calDeltaX();
			g.calculateCurrentPositions();
			//g.calBindFixed();
			//g.calFbind(allGroup, bindForce);

		}

		for (int groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
			Group& currentGroup = groups[groupIdx];

		
			for (int direction = 0; direction < 6; ++direction) {
				int adjacentGroupIdx = currentGroup.adjacentGroupIDs[direction];

				
				if (adjacentGroupIdx != -1) {
					Group& adjacentGroup = groups[adjacentGroupIdx];
					const auto& commonVerticesPair = currentGroup.commonVerticesInDirections[direction];

					//Calculate bind force with damping
					extern float youngs;
					extern float constraintHardness;
					extern float dampingConst; // 使用现有的阻尼常数作为Beta
					
					currentGroup.calFbind1(commonVerticesPair.first, commonVerticesPair.second,
						currentGroup.currentPosition, adjacentGroup.currentPosition, 
						currentGroup.groupVelocity, adjacentGroup.groupVelocity, 
						currentGroup.massMatrix, adjacentGroup.massMatrix,
						youngs, constraintHardness, dampingConst, adjacentGroupIdx);
					//if (direction == 0 || direction == 1) {
					//	currentGroup.distancesX = Eigen::VectorXf::Zero(commonVerticesPair.first.size() * 3);

					//	for (size_t i = 0; i < commonVerticesPair.first.size(); ++i) {
					//		Vertex* vertexThisGroup = commonVerticesPair.first[i];
					//		Vertex* vertexOtherGroup = commonVerticesPair.second[i];

					//		// Get the common points coordinates
					//		Eigen::Vector3f posThisGroup = currentGroup.currentPosition.segment<3>(3 * vertexThisGroup->localIndex);
					//		Eigen::Vector3f posOtherGroup = adjacentGroup.currentPosition.segment<3>(3 * vertexOtherGroup->localIndex);

					//		if (iter == looptime - 1)
					//		{
					//			currentGroup.distancesX.segment<3>(3 * i) = (posThisGroup - posOtherGroup);
					//			float totalNorm = 0.0f;
					//			int numVertices = currentGroup.distancesX.size() / 3;

					//			for (int i = 0; i < numVertices; ++i) {
					//				totalNorm += currentGroup.distancesX.segment<3>(3 * i).norm();
					//			}

					//			float averageNorm = totalNorm / numVertices;
					//			std::cout << "Average norm of distancesX for group " << groupIdx << " in direction " << direction << ": " << averageNorm << std::endl;
					//		}

					//		
					//	}

					//	
					//}
				}

			}

		}

	}
	for (int i = 0; i < groupNum; ++i) {
		auto& g = groups[i];
		g.updateVelocity();
		g.updatePosition();
		g.prevFbind = g.Fbind;
		/*g.groupVolume = 0.0f;
		for (auto& tet : g.tetrahedra) {
			float tetMass = tet->calMassTetra(1000);
			g.groupVolume += tet->volumeTetra;
		}*/
		/*float bodyVolume = 0.0f;
		bodyVolume += g.groupVolume;
		std::cout << bodyVolume << std::endl;*/
		//g.calRInvLocalPos();
	}

	//calDistance(commonPoints);
	//calDistance(commonPoints1);
	// 迭代完成后更新位置和速度
	//for (int i = 0; i < 3; ++i) {
	//	// 更新位置，这里可能需要一些逻辑来获取最后一次迭代的结果
	//	groups[i].updateFinalPositions(); // 假设这个方法用最后一次迭代的结果更新顶点位置

	//	// 更新速度
	//	groups[i].updateVelocities(timestep); // 假设这个方法用 (现在位置 - 上一帧位置) / timestep 计算速度
	//}

	// ... 现在，所有的组都应该有了更新后的位置和速度，可以传递给绘图功能
	// drawGroups(); // 假设有一个方法来绘制或输出最新的组状态
}
Group& Object::getGroup(int index) {
	return groups[index];
}

void Object::storeAllGroups() {
	allGroup.reserve(groupNum);
	for (int i = 0; i < groupNum; ++i) {
		Group& group = getGroup(i); // 获取第 i 个 Group 对象的引用
		allGroup.push_back(group); // 将 Group 对象添加到集合中
	}
}

void Object::updateAdjacentGroupIndices(int numX, int numY, int numZ) {
	for (int z = 0; z < numZ; ++z) {
		for (int y = 0; y < numY; ++y) {
			for (int x = 0; x < numX; ++x) {
				int groupIdx = z * numX * numY + y * numX + x;
				Group& currentGroup = groups[groupIdx];

				// +x方向
				if (x < numX - 1) currentGroup.adjacentGroupIDs[0] = groupIdx + 1;
				// -x方向
				if (x > 0) currentGroup.adjacentGroupIDs[1] = groupIdx - 1;
				// +y方向
				if (y < numY - 1) currentGroup.adjacentGroupIDs[2] = groupIdx + numX;
				// -y方向
				if (y > 0) currentGroup.adjacentGroupIDs[3] = groupIdx - numX;
				// +z方向
				if (z < numZ - 1) currentGroup.adjacentGroupIDs[4] = groupIdx + numX * numY;
				// -z方向
				if (z > 0) currentGroup.adjacentGroupIDs[5] = groupIdx - numX * numY;
			}
		}
	}
}

void Object::fixTopLeft10PercentVertices() {
	std::vector<Vertex*> allVertices;
	
	std::cout << "=== Starting fixTopLeft10PercentVertices ===" << std::endl;
	std::cout << "Number of groups: " << groups.size() << std::endl;
	
	// 收集所有顶点
	for (Group& group : groups) {
		std::cout << "Group has " << group.verticesMap.size() << " vertices" << std::endl;
		for (const auto& vertexPair : group.verticesMap) {
			allVertices.push_back(vertexPair.second);
		}
	}
	
	std::cout << "Total vertices collected: " << allVertices.size() << std::endl;
	
	if (allVertices.empty()) {
		std::cout << "No vertices found! Returning." << std::endl;
		return;
	}
	
	// 找到x、y、z坐标的范围
	float minX = allVertices[0]->initx;
	float maxX = allVertices[0]->initx;
	float minY = allVertices[0]->inity;
	float maxY = allVertices[0]->inity;
	float minZ = allVertices[0]->initz;
	float maxZ = allVertices[0]->initz;
	
	for (Vertex* vertex : allVertices) {
		if (vertex->initx < minX) minX = vertex->initx;
		if (vertex->initx > maxX) maxX = vertex->initx;
		if (vertex->inity < minY) minY = vertex->inity;
		if (vertex->inity > maxY) maxY = vertex->inity;
		if (vertex->initz < minZ) minZ = vertex->initz;
		if (vertex->initz > maxZ) maxZ = vertex->initz;
	}
	
	// 计算左上前10%的阈值
	float rangeX = maxX - minX;
	float rangeY = maxY - minY;
	float rangeZ = maxZ - minZ;
	float leftThresholdX = minX + rangeX * 0.1f;  // 左边10%
	float topThresholdY = maxY - rangeY * 0.1f;   // 上边10%
	float frontThresholdZ = maxZ - rangeZ * 0.1f; // 前面10%
	
	// 固定满足条件的顶点
	int fixedCount = 0;
	int candidateCount = 0; // 记录满足部分条件的点
	for (Vertex* vertex : allVertices) {
		bool xCondition = vertex->initx <= leftThresholdX;
		bool yCondition = vertex->inity >= topThresholdY;
		bool zCondition = vertex->initz >= frontThresholdZ;
		
		if (xCondition || yCondition || zCondition) {
			candidateCount++;
			if (candidateCount <= 5) { // 只打印前5个候选点的信息
				std::cout << "Vertex(" << vertex->initx << ", " << vertex->inity << ", " << vertex->initz 
					<< ") - X:" << (xCondition ? "YES" : "NO") 
					<< " Y:" << (yCondition ? "YES" : "NO") 
					<< " Z:" << (zCondition ? "YES" : "NO") << std::endl;
			}
		}
		
		// 暂时简化条件，先只用x方向测试
		if (xCondition) {
			vertex->isFixed = true;
			fixedCount++;
			if (fixedCount <= 3) { // 打印前3个被固定的点
				std::cout << "FIXED vertex(" << vertex->initx << ", " << vertex->inity << ", " << vertex->initz << ")" << std::endl;
			}
		}
	}
	
	std::cout << "Candidate vertices (satisfying at least one condition): " << candidateCount << std::endl;
	
	std::cout << "Fixed " << fixedCount << " vertices in the top-left-front 10% region." << std::endl;
	std::cout << "X range: [" << minX << ", " << maxX << "], left 10% threshold: " << leftThresholdX << std::endl;
	std::cout << "Y range: [" << minY << ", " << maxY << "], top 10% threshold: " << topThresholdY << std::endl;
	std::cout << "Z range: [" << minZ << ", " << maxZ << "], front 10% threshold: " << frontThresholdZ << std::endl;
}