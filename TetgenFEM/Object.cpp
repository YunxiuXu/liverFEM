#include "Object.h"
#include "params.h"
#include "Edge.h"
#include "Group.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <iterator>








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

	// 收集所有四面体及其质心，保持在各自空间块内（不跨格子搬运，避免破坏邻接假设）
	struct TetInfo {
		Tetrahedron* tet;
		Eigen::Vector3f centroid;
	};
	std::vector<TetInfo> tetInfos;
	tetInfos.reserve(out.numberoftetrahedra);

	for (int i = 0; i < out.numberoftetrahedra; ++i) {
		Vertex* v1 = vertices[out.tetrahedronlist[i * 4] - 1];
		Vertex* v2 = vertices[out.tetrahedronlist[i * 4 + 1] - 1];
		Vertex* v3 = vertices[out.tetrahedronlist[i * 4 + 2] - 1];
		Vertex* v4 = vertices[out.tetrahedronlist[i * 4 + 3] - 1];
		Tetrahedron* tet = new Tetrahedron(v1, v2, v3, v4);

		Eigen::Vector3f centroid(
			(v1->x + v2->x + v3->x + v4->x) / 4.0f,
			(v1->y + v2->y + v3->y + v4->y) / 4.0f,
			(v1->z + v2->z + v3->z + v4->z) / 4.0f);

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

		tetInfos.push_back({ tet, centroid });
	}

	// 自适应分位点切分：沿 x/y/z 轴按质心分布划分 numX/numY/numZ 段，尽量均匀又保持规则网格
	auto makeEdges = [](std::vector<float>& values, int bins, float minVal, float maxVal) {
		std::vector<float> edges;
		edges.reserve(bins + 1);
		if (values.empty() || bins <= 0) return edges;
		std::sort(values.begin(), values.end());

		edges.push_back(minVal - 1e-4f); // 左边界偏移一点点
		for (int i = 1; i < bins; ++i) {
			int idx = static_cast<int>(std::round(static_cast<float>(i) * values.size() / bins));
			idx = std::max(0, std::min(static_cast<int>(values.size()) - 1, idx));
			edges.push_back(values[idx]);
		}
		edges.push_back(maxVal + 1e-4f);

		// 保证严格递增，避免上界相同导致 upper_bound 失败
		float step = std::max(1e-5f, (maxVal - minVal) * 1e-5f);
		for (size_t i = 1; i < edges.size(); ++i) {
			if (edges[i] <= edges[i - 1]) {
				edges[i] = edges[i - 1] + step;
			}
		}
		return edges;
	};

	std::vector<float> xVals, yVals, zVals;
	xVals.reserve(tetInfos.size());
	yVals.reserve(tetInfos.size());
	zVals.reserve(tetInfos.size());
	for (const auto& ti : tetInfos) {
		xVals.push_back(ti.centroid.x());
		yVals.push_back(ti.centroid.y());
		zVals.push_back(ti.centroid.z());
	}

	std::vector<float> xEdges = makeEdges(xVals, numX, minX, maxX);
	std::vector<float> yEdges = makeEdges(yVals, numY, minY, maxY);
	std::vector<float> zEdges = makeEdges(zVals, numZ, minZ, maxZ);

	auto findBin = [](float v, const std::vector<float>& edges, int fallbackBins) {
		if (edges.size() < 2) {
			return std::max(0, std::min(fallbackBins - 1, fallbackBins - 1));
		}
		auto it = std::upper_bound(edges.begin(), edges.end(), v);
		int idx = static_cast<int>(it - edges.begin()) - 1;
		idx = std::max(0, std::min(static_cast<int>(edges.size()) - 2, idx));
		return idx;
	};

	// 重新按照自适应网格分组
	for (const auto& ti : tetInfos) {
		int gx = findBin(ti.centroid.x(), xEdges, numX);
		int gy = findBin(ti.centroid.y(), yEdges, numY);
		int gz = findBin(ti.centroid.z(), zEdges, numZ);

		int groupIdx = gz * numX * numY + gy * numX + gx;
		object.groups[groupIdx].addTetrahedron(ti.tet);
		object.groups[groupIdx].groupIndex = groupIdx;
	}

	// 统计分布情况
	int totalGroups = numX * numY * numZ;
	int finalEmptyGroups = 0;
	int minSize = INT_MAX, maxSize = 0;
	for (int i = 0; i < totalGroups; ++i) {
		int size = object.groups[i].tetrahedra.size();
		if (size == 0) finalEmptyGroups++;
		minSize = std::min(minSize, size);
		maxSize = std::max(maxSize, size);
	}

	std::cout << "分组完成: 空组 " << finalEmptyGroups
		<< ", 组大小范围 [" << minSize << ", " << maxSize << "]" << std::endl;
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
#pragma omp parallel for
	for (int i = 0; i < groupNum; ++i) {
		auto& g = groups[i];
		//g.Fbind = 0.75 * g.prevFbind;
		g.Fbind = Eigen::VectorXf::Zero(3 * g.verticesMap.size()); // 假设 Group 类有一个方法来清除 Fbind
		g.applyRotationTranspose(g.primeVec, g.rotatedPrimeVec);
		g.RHS_B = g.RHS_E * g.rotatedPrimeVec; //46ms
		g.RHS_AsubBplusC = g.RHS_A - g.RHS_B; //24ms

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

#pragma omp parallel for
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
		// Sample a few constraints (serially) for early-exit heuristic without paying OMP overhead
		float avgConstraintNorm = 0.0f;
		int samples = 0;
		for (int groupIdx = 0; groupIdx < groups.size() && samples < 16; ++groupIdx) {
			Group& currentGroup = groups[groupIdx];
			for (int direction = 0; direction < 6 && samples < 16; ++direction) {
				int adjacentGroupIdx = currentGroup.adjacentGroupIDs[direction];
				if (adjacentGroupIdx != -1) {
					Group& adjacentGroup = groups[adjacentGroupIdx];
					const auto& commonVerticesPair = currentGroup.commonVerticesInDirections[direction];
					if (!commonVerticesPair.first.empty()) {
						Eigen::Vector3f posThis = currentGroup.currentPosition.segment<3>(3 * commonVerticesPair.first[0]->localIndex);
						Eigen::Vector3f posOther = adjacentGroup.currentPosition.segment<3>(3 * commonVerticesPair.second[0]->localIndex);
						avgConstraintNorm += (posThis - posOther).norm();
						++samples;
					}
				}
			}
		}
		if (samples > 0) {
			avgConstraintNorm /= static_cast<float>(samples);
			if (avgConstraintNorm < 5e-4f) {
				break;
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

void Object::fixRegion(const Eigen::Vector3f& center, float radius) {
	std::unordered_set<Vertex*> visited;
	std::vector<Vertex*> uniqueVertices;

	for (Group& group : groups) {
		for (const auto& vertexPair : group.verticesMap) {
			Vertex* vertex = vertexPair.second;
			if (visited.insert(vertex).second) {
				uniqueVertices.push_back(vertex);
			}
		}
	}

	if (uniqueVertices.empty()) {
		std::cout << "fixRegion: no vertices available to fix." << std::endl;
		return;
	}

	const float radiusSq = radius * radius;
	int fixedCount = 0;
	for (Vertex* vertex : uniqueVertices) {
		Eigen::Vector3f pos(vertex->initx, vertex->inity, vertex->initz);
		if ((pos - center).squaredNorm() <= radiusSq) {
			vertex->isFixed = true;
			++fixedCount;
		}
	}

	std::cout << "fixRegion: center " << center.transpose()
		<< ", radius " << radius
		<< ", fixed " << fixedCount << "/" << uniqueVertices.size()
		<< " vertices." << std::endl;
}
