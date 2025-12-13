#include "Group.h"
#include "params.h"

// Initialize debug flag (set to false by default, change to true for debugging)
bool Group::debug_constraints = false;

void Group::initialize() {
	groupVelocity = Eigen::VectorXf::Zero(3 * verticesMap.size());
	groupVelocityFEM = Eigen::VectorXf::Zero(3 * verticesMap.size());
	Fbind = Eigen::VectorXf::Zero(3 * verticesMap.size());
	prevFbind = Eigen::VectorXf::Zero(3 * verticesMap.size());
	currentPosition = Eigen::VectorXf::Zero(3 * verticesMap.size());
	currentPositionFEM = Eigen::VectorXf::Zero(3 * verticesMap.size());
	gravity = Eigen::VectorXf::Zero(3 * verticesMap.size());

	verticesVector.reserve(verticesMap.size());
	for (const auto& pair : verticesMap) {
		verticesVector.push_back(pair.second);
	}
}
void Group::addTetrahedron(Tetrahedron* tet) {
	tetrahedra.push_back(tet);
	//for (int i = 0; i < 4; ++i) {
	//	verticesMap[tet->vertices[i]->index] = tet->vertices[i];
	//}
}
std::vector<Vertex*> Group::getUniqueVertices() { 
	std::vector<Vertex*> uniqueVertices;
	for (auto& pair : verticesMap) {
		uniqueVertices.push_back(pair.second);
	}
	return uniqueVertices;
}

void Group::calCenterofMass() {
	float totalMass = 0.0;
	Eigen::Vector3f weightedSum(0.0, 0.0, 0.0);

	for (const auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		totalMass += vertex->vertexMass;
		weightedSum += vertex->vertexMass * Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
	}

	if (totalMass > 0.0) {
		centerofMass = weightedSum / totalMass;
	}
	else {
		// Handle the case where totalMass is zero to avoid division by zero.
		// You can set centerofMass to a default value or handle it according to your requirements.
		centerofMass = Eigen::Vector3f(0.0, 0.0, 0.0);
	}
}
void Group::calLocalPos() {
	
	initLocalPos.resize(3 * verticesMap.size());

	for (const auto& vertexPair : verticesMap) {
		const Vertex* vertex = vertexPair.second;
		Eigen::Vector3f initial_position(vertex->initx, vertex->inity, vertex->initz);
	//initial local position
		Eigen::Vector3f local_position = initial_position - initCOM;

		// 将局部位置存?在initLocalPos中，注意index需要乘以3因??个?点有3个坐??
		initLocalPos.segment<3>(vertex->localIndex * 3) = local_position;
	}
}
Eigen::Vector3f Group::axlAPD(Eigen::Matrix3f a) {
	Eigen::Vector3f g = Eigen::Vector3f::Zero();
	g[0] = a(1, 2) - a(2, 1);
	g[1] = a(2, 0) - a(0, 2);
	g[2] = a(0, 1) - a(1, 0);
	return g;
}
Eigen::Vector3f Group::clamp2(Eigen::Vector3f x, float y, float z) {
	if (x.norm() < y) {
		return (y / x.norm()) * x;
	}
	else if (x.norm() > z) {
		return (z / x.norm()) * x;
	}
	else {
		return x;
	}
}
Eigen::Quaternionf Group::Exp2(Eigen::Vector3f a) {
	float s = sin((a * 0.5).norm());
	float x = s * a.x() / a.norm();
	float y = s * a.y() / a.norm();
	float z = s * a.z() / a.norm();
	Eigen::Quaternionf qq = Eigen::Quaternionf(cos((a * 0.5).norm()), x, y, z);
	return  qq;
}
//void Group::calRotationMatrix() {
//	Eigen::MatrixXf Apq = Eigen::MatrixXf::Zero(3, 3);
//	Eigen::Matrix3f tempA = Eigen::Matrix3f::Zero();
//	Eigen::Vector3f center_grid = Eigen::Vector3f::Zero();
//
//	Eigen::Vector3f tempCenterGrid = Eigen::Vector3f::Zero();
//#pragma omp parallel num_threads(4)
//	{
//		Eigen::Vector3f localCenterGrid = Eigen::Vector3f::Zero();
//#pragma omp for
//		for (int i = 0; i < static_cast<int>(verticesMap.size()); ++i) {
//			auto it = verticesMap.begin();
//			std::advance(it, i);
//			const std::pair<const int, Vertex*>& vertexEntry = *it;
//
//			localCenterGrid.noalias() += massDistribution(0, 3 * vertexEntry.second->localIndex) * primeVec.segment<3>(3 * vertexEntry.second->localIndex);
//		}
//#pragma omp critical
//		{
//			tempCenterGrid += localCenterGrid;
//		}
//	}
//	center_grid = tempCenterGrid;
//
//
//	// Apq
//	Eigen::MatrixXf tempApq = Eigen::MatrixXf::Zero(3, 3);
//#pragma omp parallel num_threads(4)
//	{
//		Eigen::MatrixXf localApq = Eigen::MatrixXf::Zero(3, 3);
//#pragma omp for
//		for (int i = 0; i < static_cast<int>(verticesMap.size()); ++i) {
//			auto it = verticesMap.begin();
//			std::advance(it, i);
//			const std::pair<const int, Vertex*>& vertexEntry = *it;
//
//			tempA.noalias() = (primeVec.segment<3>(3 * vertexEntry.second->localIndex) - center_grid) * initLocalPos.segment<3>(3 * vertexEntry.second->localIndex).transpose();
//			localApq.noalias() += massMatrix(3 * vertexEntry.second->localIndex, 3 * vertexEntry.second->localIndex) * tempA;
//		}
//#pragma omp critical
//		{
//			tempApq += localApq;
//		}
//	}
//	Apq = tempApq;
//
//	
//	Eigen::Vector3f omega = Eigen::Vector3f::Identity();
//	Eigen::Quaternionf quaternion(Eigen::Quaternionf::Identity());
//	rotate_matrix = Eigen::Matrix3f::Identity();
//	Eigen::Vector3f gradR = Eigen::Vector3f::Zero();
//	Eigen::Matrix3f HesseR = Eigen::Matrix3f::Zero();
//	Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
//
//	// iterate to find rotation
//	for (unsigned int ci = 0; ci < 20; ci++) {
//		Eigen::Matrix3f R = quaternion.matrix();
//		Eigen::Matrix3f S = R.transpose() * Apq;
//		Eigen::Vector3f gradR = axlAPD(S);
//		Eigen::Matrix3f HesseR = S.trace() * Eigen::Matrix3f::Identity() - (S + S.transpose()) * 0.5;
//		Eigen::Vector3f omega = -HesseR.inverse() * gradR;
//		float w = omega.norm();
//		if (w < 1.0e-9) {
//			break;
//		}
//		omega = clamp2(omega, -PI, PI);
//		Eigen::Quaternionf temp2 = Exp2(omega);
//		quaternion = quaternion * temp2;
//	}
//
//	rotate_matrix = quaternion.matrix();
//
//	// 3N*3N rotation matrix
//	rotationMatrix = Eigen::MatrixXf::Zero(3 * verticesMap.size(), 3 * verticesMap.size());
//
//#pragma omp parallel for
//	for (int pi = 0; pi < static_cast<int>(verticesMap.size()); pi++) {
//		rotationMatrix.block<3, 3>(3 * pi, 3 * pi) = rotate_matrix;
//	}
//}
//
void Group::calRotationMatrix() {
	Eigen::MatrixXf Apq = Eigen::MatrixXf::Zero(3, 3);
	Eigen::Vector3f center_grid = Eigen::Vector3f::Zero();

	// 计算质心
	Eigen::Vector3f tempCenterGrid = Eigen::Vector3f::Zero();
	//#pragma omp parallel num_threads(4)
	{
		Eigen::Vector3f localCenterGrid = Eigen::Vector3f::Zero();
		//#pragma omp for
		for (int i = 0; i < static_cast<int>(verticesMap.size()); ++i) {
			auto it = verticesMap.begin();
			std::advance(it, i);
			const std::pair<const int, Vertex*>& vertexEntry = *it;
			localCenterGrid.noalias() += massDistribution(0, 3 * vertexEntry.second->localIndex) * primeVec.segment<3>(3 * vertexEntry.second->localIndex);
		}
		//#pragma omp critical
		{
			tempCenterGrid += localCenterGrid;
		}
	}
	center_grid = tempCenterGrid;

	// 计算Apq矩阵
//#pragma omp parallel num_threads(4)
	{
		Eigen::MatrixXf localApq = Eigen::MatrixXf::Zero(3, 3);
		//#pragma omp for
		for (int i = 0; i < static_cast<int>(verticesMap.size()); ++i) {
			auto it = verticesMap.begin();
			std::advance(it, i);
			const std::pair<const int, Vertex*>& vertexEntry = *it;
			Eigen::MatrixXf tempA = (primeVec.segment<3>(3 * vertexEntry.second->localIndex) - center_grid) * initLocalPos.segment<3>(3 * vertexEntry.second->localIndex).transpose();
			localApq.noalias() += massMatrix(3 * vertexEntry.second->localIndex, 3 * vertexEntry.second->localIndex) * tempA;
		}
		//#pragma omp critical
		{
			Apq += localApq;
		}
	}

	// 进行极分解以确保Apq为正交矩阵
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(Apq, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf U = svd.matrixU();
	Eigen::MatrixXf V = svd.matrixV();
	Eigen::MatrixXf R = U * V.transpose();

	// 如果行列式为负，调整U矩阵
	if (R.determinant() < 0) {
		U.col(2) *= -1;
		R = U * V.transpose();
	}

	// 最终的旋转矩阵
	rotate_matrix = R;

	// 构建旋转矩阵的3N x 3N版本
	rotationMatrix = Eigen::MatrixXf::Zero(3 * verticesMap.size(), 3 * verticesMap.size());
	//#pragma omp parallel for
	for (int pi = 0; pi < static_cast<int>(verticesMap.size()); pi++) {
		rotationMatrix.block<3, 3>(3 * pi, 3 * pi) = rotate_matrix;
	}

	// 打印调试信息
	//std::cout << "Final rotation matrix: " << rotate_matrix << std::endl;
}
void Group::calGroupK(float E, float nu) {
	// Initialize groupK to the right size. Assuming 3 degrees of freedom per vertex
	int dof = verticesMap.size() * 3;
	groupK.resize(dof, dof);
	groupK.setZero();

	// Iterate over each tetrahedron to assemble the global stiffness matrix
	for (auto& tetra : tetrahedra) {
		// Get the local stiffness matrix for the current tetrahedron
		Eigen::MatrixXf localK = tetra->createElementK(E, nu, initCOM);

		// Determine where to add the local stiffness matrix in the global stiffness matrix
		for (int i = 0; i < 4; ++i) { // Each tetrahedron has 4 vertices
			Vertex* vertex = tetra->vertices[i];
			int localIndex = vertex->localIndex * 3; // Use local index

			for (int j = 0; j < 4; ++j) {
				Vertex* otherVertex = tetra->vertices[j];
				int otherLocalIndex = otherVertex->localIndex * 3;

				// Add the 3x3 submatrix of localK to the correct place in groupK
				groupK.block<3, 3>(localIndex, otherLocalIndex) += localK.block<3, 3>(i * 3, j * 3);
			}
		}
	}

	kSparse = groupK.sparseView();
}
void Group::calGroupKAni(float E1, float E2, float E3, float nu) {
	// Initialize groupK to the right size. Assuming 3 degrees of freedom per vertex
	int dof = verticesMap.size() * 3;
	groupK.resize(dof, dof);
	groupK.setZero();

	// Iterate over each tetrahedron to assemble the global stiffness matrix
	for (auto& tetra : tetrahedra) {
		// Get the local stiffness matrix for the current tetrahedron
		Eigen::MatrixXf localK = tetra->createElementKAni(E1, E2, E3, nu, initCOM);

		// Determine where to add the local stiffness matrix in the global stiffness matrix
		for (int i = 0; i < 4; ++i) { // Each tetrahedron has 4 vertices
			Vertex* vertex = tetra->vertices[i];
			int localIndex = vertex->localIndex * 3; // Use local index

			for (int j = 0; j < 4; ++j) {
				Vertex* otherVertex = tetra->vertices[j];
				int otherLocalIndex = otherVertex->localIndex * 3;

				// Add the 3x3 submatrix of localK to the correct place in groupK
				groupK.block<3, 3>(localIndex, otherLocalIndex) += localK.block<3, 3>(i * 3, j * 3);
			}
		}
	}

	kSparse = groupK.sparseView();
}
void Group::calGroupKFEM(float E, float nu) {
	// Initialize groupK to the right size. Assuming 3 degrees of freedom per vertex
	int dof = verticesMap.size() * 3;
	groupKFEM.resize(dof, dof);
	groupKFEM.setZero();

	// Iterate over each tetrahedron to assemble the global stiffness matrix
	for (auto& tetra : tetrahedra) {
		// Get the local stiffness matrix for the current tetrahedron
		Eigen::MatrixXf localK = tetra->createElementKFEM(E, nu);

		// Determine where to add the local stiffness matrix in the global stiffness matrix
		for (int i = 0; i < 4; ++i) { // Each tetrahedron has 4 vertices
			Vertex* vertex = tetra->vertices[i];
			int localIndex = vertex->localIndex * 3; // Use local index

			for (int j = 0; j < 4; ++j) {
				Vertex* otherVertex = tetra->vertices[j];
				int otherLocalIndex = otherVertex->localIndex * 3;

				// Add the 3x3 submatrix of localK to the correct place in groupK
				groupKFEM.block<3, 3>(localIndex, otherLocalIndex) += localK.block<3, 3>(i * 3, j * 3);
			}
		}
	}

	kSparseFEM = groupKFEM.sparseView();
}


void Group::calMassGroup() {
	groupMass = 0.0; 
	for (auto& tet : tetrahedra) { 
		groupMass += tet->massTetra; 
	}
}

Eigen::MatrixXf Group::calMassMatrix(float den) {
	int N = verticesMap.size();  // Number of unique vertices
	Eigen::MatrixXf M = Eigen::MatrixXf::Zero(3 * N, 3 * N);  // Initialize mass matrix

	for (auto& tet : tetrahedra) {
		float tetMass = tet->calMassTetra(den);
		float vertexMass = tetMass / 4.0;  // Assume uniform distribution of mass among vertices

		for (int i = 0; i < 4; i++) {
			int localIdx = tet->vertices[i]->localIndex;  // Use local index
			M(3 * localIdx, 3 * localIdx) += vertexMass;     // x-direction
			M(3 * localIdx + 1, 3 * localIdx + 1) += vertexMass; // y-direction
			M(3 * localIdx + 2, 3 * localIdx + 2) += vertexMass; // z-direction
		}
	}
	massMatrix = M;
	return M;
	massSparse = massMatrix.sparseView();  // Commented out as it's not part of the provided code snippet
}


void Group::calDampingMatrix() {
	dampingMatrix.setZero();
	dampingMatrix = dampingConst * massMatrix;
	dampingSparse = dampingMatrix.sparseView();
}

void Group::calMassDistributionMatrix() {
	int N = verticesMap.size();  // Number of unique vertices
	massDistribution = Eigen::MatrixXf::Zero(3 * N, 3 * N);
	Eigen::MatrixXf SUMsub_M_Matrix = Eigen::MatrixXf::Zero(3, 3 * N);

	for (const auto& vertexEntry : verticesMap) {
		Vertex* vertex = vertexEntry.second;
		int localIdx = vertex->localIndex;  // Use local index

		// Create a 3x3 identity matrix scaled by the vertex's mass
		SUMsub_M_Matrix.block(0, 3 * localIdx, 3, 3) = (vertex->vertexMass / groupMass) * Eigen::Matrix3f::Identity();
	}

	for (const auto& vertexEntry : verticesMap) {
		Vertex* vertex = vertexEntry.second;
		int localIdx = vertex->localIndex;  // Use local index

		massDistribution.block(3 * localIdx, 0, 3, 3 * N) = SUMsub_M_Matrix;
	}

	// Reset SUM_M to a sparse view
	massDistributionSparse = massDistribution.sparseView();
}

void Group::setVertexMassesFromMassMatrix() {
	int N = verticesMap.size();  // Number of unique vertices in the hash map

	for (auto& vertexPair : verticesMap) {
		int localIdx = vertexPair.second->localIndex;  // Access local index from the second value of the pair
		float mass = massMatrix(3 * localIdx, 3 * localIdx);
		vertexPair.second->vertexMass = mass;
	}
}



void Group::calInitCOM() {
	float totalMass = 0.0;
	Eigen::Vector3f weightedSum(0.0, 0.0, 0.0);

	for (const auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		totalMass += vertex->vertexMass;
		weightedSum += vertex->vertexMass * Eigen::Vector3f(vertex->initx, vertex->inity, vertex->initz);
	}

	if (totalMass > 0.0) {
		initCOM = weightedSum / totalMass;
	}
	else {
		// Handle the case where totalMass is zero to avoid division by zero.
		// You can set centerofMass to a default value or handle it according to your requirements.
		initCOM = Eigen::Vector3f(0.0, 0.0, 0.0);
	}
}
void Group::calPrimeVec1(int w) {
	// Assuming Gravity and timeStep are defined elsewhere
	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());
	gravity = Eigen::VectorXf::Zero(3 * verticesVector.size());

	// Find min and max x-values
	float minX = std::numeric_limits<float>::max();
	float maxX = std::numeric_limits<float>::lowest();
	for (const auto& vertex : verticesVector) {
		if (vertex->x < minX) minX = vertex->x;
		if (vertex->x > maxX) maxX = vertex->x;
	}

	// Calculate the midpoint of the x-axis
	float midX = (minX + maxX) / 2;

	// Apply forces based on position relative to the midpoint
	for (size_t i = 0; i < verticesVector.size(); ++i) {
		float x = verticesVector[i]->x;
		int yIndex = 3 * i + 1; // Index for the y-component in the gravity vector

		if (x <= midX) {
			// Front half, apply +y force
			gravity(yIndex) = Gravity;
		}
		else {
			// Back half, apply -y force
			gravity(yIndex) = -Gravity;
		}
	}

	// Update groupVelocity
	groupVelocity += gravity * timeStep;

	// Compute velocityUpdate
	Eigen::VectorXf velocityUpdate = inverseTermSparse * (massMatrix * groupVelocity) * timeStep;

	// Update primeVec and vertex positions
	for (auto& vertex : verticesVector) {
		int localPi = vertex->localIndex; // Use local index

		// Get the current vertex's velocity update part
		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);

		// Calculate new position
		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate;

		// Update primeVec
		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
	}
}
//void Group::calPrimeVecT(int w) {
//	// ?保primeVec已?初始化且?置?正?的尺寸
//	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//	// 初始化旋?力向量
//	Eigen::VectorXf twistForce = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//	// ??是否是第三?，如果是，??特定?点施加旋?力
//	if (this->groupIndex == 2) {
//		// ?置?个?点旋?力的大小，?里?化??一大小，具体?需要根据??情况?整
//		float forceMagnitude = 10.0f; // 示例力的大小
//
//		// ?点局部索引数?
//		std::vector<int> indices = { 116, 91, 24, 22 };
//
//		// ??个?点施加旋?力
//		for (int index : indices) {
//			// ?定力的方向：根据?点在yz面上的???旋?方向施加
//			if (index == 116) {
//				// ?于116，假?力向下（y??方向）
//				twistForce(3 * index + 1) = -forceMagnitude;
//			}
//			else if (index == 91) {
//				// ?于91，假?力向右（z?正方向）
//				twistForce(3 * index + 2) = forceMagnitude;
//			}
//			else if (index == 24) {
//				// ?于24，假?力向上（y?正方向）
//				twistForce(3 * index + 1) = forceMagnitude;
//			}
//			else if (index == 22) {
//				// ?于22，假?力向左（z??方向）
//				twistForce(3 * index + 2) = -forceMagnitude;
//			}
//		}
//	}
//
//	// 更新groupVelocity
//	groupVelocity += twistForce * timeStep;
//
//	// 使用整个矩??算velocityUpdate
//	Eigen::VectorXf velocityUpdate = inverseTermSparse * (massMatrix * groupVelocity) * timeStep;
//
//	// 更新primeVec和?点位置
//	for (auto& vertexPair : verticesVector) {
//		Vertex* vertex = vertexPair;
//		int localPi = vertex->localIndex; // 使用局部索引
//
//		// ?取当前?点的速度更新部分
//		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
//
//		// ?算新的位置
//		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate;
//
//		// 更新primeVec
//		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
//	}
//}
//void Group::calPrimeVecS(int wKey) {
//	// ?保primeVec已?初始化且?置?正?的尺寸
//	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//	// 初始化一个力向量
//	Eigen::VectorXf appliedForce = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//	// 特定?点局部索引数?
//	std::vector<int> indices = { 55, 35, 57, 70 };
//
//	// 施加力的大小，可以根据??情况?整
//	float forceMagnitude = 1000.0f;
//
//	// 如果是第三?，根据wKey的??特定?点施加力
//	if (this->groupIndex == 2) {
//		for (int localPi : indices) {
//			int indexX = 3 * localPi; // x方向的索引
//
//			// 根据???入wKey的?在x方向上施加力
//			if (wKey == 3) {
//				appliedForce(indexX) -= forceMagnitude; // 向左施加力
//			}
//			else if (wKey == 4) {
//				appliedForce(indexX) += forceMagnitude; // 向右施加力
//			}
//			// ?于wKey == 1或2，我?不在x方向上施加力
//		}
//	}
//
//	// 更新groupVelocity
//	groupVelocity += appliedForce * timeStep;
//
//	// 使用整个矩??算velocityUpdate
//	Eigen::VectorXf velocityUpdate = inverseTermSparse * (massMatrix * groupVelocity) * timeStep;
//
//	// 更新primeVec和?点位置
//	for (auto& vertexPair : verticesVector) {
//		Vertex* vertex = vertexPair;
//		int localPi = vertex->localIndex; // 使用局部索引
//
//		// ?取当前?点的速度更新部分
//		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
//
//		// ?算新的位置
//		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate;
//
//		// 更新primeVec
//		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
//	}
//}
void Group::calPrimeVecS(const std::vector<int>& topVertexLocalIndices, const std::vector<int>& bottomVertexLocalIndices) {
	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());

	if (!gravityApplied) {
		// Initialize gravity vector, apply gravity only in the y direction
		gravity = Eigen::VectorXf::Zero(3 * verticesVector.size());

		for (int i = 0; i < 3 * verticesVector.size(); i += 3) {
			gravity(i) = -0; // No gravity in the y direction initially
		}

		gravityApplied = true; // Mark gravity as applied to prevent future updates
	}

	float meanX = 0.0f;
	for (const auto& vertexPair : verticesVector) {
		meanX += vertexPair->initx;
	}
	meanX /= verticesVector.size();

	for (const auto& vertexPair : verticesVector) {
		Vertex* vertex = vertexPair;
		if (vertex->initx < meanX) {
			// Vertex is left of meanX
			gravity(3 * vertex->localIndex) = -0.5 * Gravity;
		}
		else {
			// Vertex is right of meanX
			gravity(3 * vertex->localIndex) = 0.5 * Gravity;
		}

		// Check if vertex is in topVertexLocalIndices
		if (std::find(topVertexLocalIndices.begin(), topVertexLocalIndices.end(), vertex->index) != topVertexLocalIndices.end()) {
			gravity(3 * vertex->localIndex + 1) = -Gravity; // Apply upward force in y direction
		}

		// Check if vertex is in bottomVertexLocalIndices
		if (std::find(bottomVertexLocalIndices.begin(), bottomVertexLocalIndices.end(), vertex->index) != bottomVertexLocalIndices.end()) {
			gravity(3 * vertex->localIndex + 1) = Gravity; // Apply downward force in y direction
		}
	}

	Eigen::VectorXf exfUpdate = timeStep * timeStep * inverseTerm * massMatrix * gravity;
	Eigen::VectorXf velocityUpdate = inverseTerm * massMatrix * groupVelocity * timeStep;

	for (auto& vertexPair : verticesVector) {
		Vertex* vertex = vertexPair;
		int localPi = vertex->localIndex; // Use local index
		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
		Eigen::Vector3f currentExfUpdate = exfUpdate.segment<3>(3 * localPi);
		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate + currentExfUpdate;

		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
	}
}



void Group::calPrimeVecT(int w) {//twist
	
	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());

	Eigen::VectorXf twistForce = Eigen::VectorXf::Zero(3 * verticesVector.size());

	// which vertex
	std::vector<int> indices = { 22, 17, 9, 15 };
	std::vector<int> indices1 = { 24, 18, 5, 17 };
	if (this->groupIndex == 0) {
		//avg position
		Eigen::Vector3f avgPosition = Eigen::Vector3f::Zero();
		for (int index : indices1) {
			Vertex* vertex = verticesVector[index];
			avgPosition += Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
		}
		avgPosition /= indices1.size();

	
		for (int index : indices1) {
			Vertex* vertex = verticesVector[index];
			Eigen::Vector3f pos = Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
			Eigen::Vector3f toCenter = pos - avgPosition;

			// tagential direction
			Eigen::Vector3f tangentForce = Eigen::Vector3f(0, -toCenter.z(), toCenter.y()).normalized();

			
			float forceMagnitude = 150.0f; 
			twistForce(3 * index) += 0; 
			twistForce(3 * index + 1) += forceMagnitude * tangentForce.y(); 
			twistForce(3 * index + 2) += forceMagnitude * tangentForce.z(); 
		}
	}

	
	if (this->groupIndex == 3) {
		
		Eigen::Vector3f avgPosition = Eigen::Vector3f::Zero();
		for (int index : indices) {
			Vertex* vertex = verticesVector[index];
			avgPosition += Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
		}
		avgPosition /= indices.size();

		
		for (int index : indices) {
			Vertex* vertex = verticesVector[index];
			Eigen::Vector3f pos = Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
			Eigen::Vector3f toCenter = pos - avgPosition;

			
			Eigen::Vector3f tangentForce = Eigen::Vector3f(0, toCenter.z(), -toCenter.y()).normalized();

			
			float forceMagnitude = 150.0f; 
			twistForce(3 * index) += 0; 
			twistForce(3 * index + 1) += forceMagnitude * tangentForce.y();
			twistForce(3 * index + 2) += forceMagnitude * tangentForce.z();
		}
	}


	groupVelocity += twistForce * timeStep;

	
	Eigen::VectorXf velocityUpdate = inverseTermSparse * (massMatrix * groupVelocity) * timeStep;

	
	for (auto& vertexPair : verticesVector) {
		Vertex* vertex = vertexPair;
		int localPi = vertex->localIndex; 

		
		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);

		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate;

		
		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
	}
}


void Group::calPrimeVec2(int w) {
	
	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());


	gravity = Eigen::VectorXf::Zero(3 * verticesVector.size());

	int localPi = 2; //set a specific vertex
	int globalPi = 69;
	if (globalPi < verticesVector.size()) {
		Vertex* v = verticesVector[globalPi];
		int gravityIndex = 3 * (v->localIndex) + 1;
		if (w == 1 || w == 2) {
			gravity(gravityIndex) = (w == 1) ? Gravity : -Gravity; 
		}
	}
	groupVelocity += gravity * timeStep;

	
	Eigen::VectorXf velocityUpdate = inverseTermSparse * (massMatrix * groupVelocity) * timeStep;

	
	for (auto& vertexPair : verticesVector) {
		Vertex* vertex = vertexPair;
		int localPi = vertex->localIndex; // 使用局部索引

		
		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);

	
		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate;

		
		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;

	}
	
}
//void Group::calPrimeVec(int w) {
//	
//	groupVelocity = Eigen::VectorXf::Zero(3 * verticesMap.size());
//	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//
//	gravity = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//	
//	if (w == 4) {
//		for (int i = 0; i < 3 * verticesVector.size(); i += 3) {
//			gravity(i + 2) = Gravity; // +x
//			
//		}
//	}
//	else if (w == 2) {
//		for (int i = 1; i < 3 * verticesVector.size(); i += 3) {
//			gravity(i) = -Gravity; // -y
//		}
//	}
//	else if (w == 1) {
//		for (int i = 1; i < 3 * verticesVector.size(); i += 3) {
//			gravity(i) = -Gravity; // +y
//		}
//	}
//	else if (w == 3) {
//		for (int i = 0; i < 3 * verticesVector.size(); i += 3) {
//			gravity(i) = Gravity; // -x
//		}
//	}
//	
//	groupVelocityFEM += gravity * timeStep;
//	Eigen::VectorXf exfUpdate = timeStep * timeStep * inverseTerm * massMatrix * gravity;
//	Eigen::VectorXf velocityUpdate = inverseTerm * massMatrix * groupVelocity * timeStep;
//
//	for (auto& vertexPair : verticesVector) {
//		Vertex* vertex = vertexPair;
//		int localPi = vertex->localIndex;
//
//		 检查顶点是否满足 inity > 0.48
//		if (vertex->initz > 1.1) {
//			Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
//			Eigen::Vector3f currentExfUpdate = exfUpdate.segment<3>(3 * localPi);
//			Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->initz + 4);
//			primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
//		}
//		else {
//			 不满足条件时，保持原位置
//			primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
//		}
//	}
//
//}
void Group::calPrimeVec(const Eigen::Vector3f& externalForce) {
	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());
	static int frameCount = 0;
	frameCount++;

	if (!gravityApplied) {
		for (int i = 0; i < 3 * verticesVector.size(); i += 3) {
			gravity(i + 1) = Gravity;
			/*float rotatedGravityX = -Gravity * sqrt(2) / 2;
			float rotatedGravityY = -Gravity * sqrt(2) / 2;
			gravity(i) = rotatedGravityX;
			gravity(i + 1) = rotatedGravityY;*/
		}
		//for (auto& vertexPair : verticesVector) {
		//	Vertex* vertex = vertexPair;
		//	if (vertex->initx > -0.02 && vertex->initx < 0.02)
		//	{
		//		gravity(vertex->localIndex * 3 + 1) = -Gravity;
		//		//gravity(vertex->localIndex * 3) = -Gravity;
		//	}
		//	//if (vertex->inity > 0.53)
		//	//{
		//	//	gravity(vertex->localIndex * 3 + 1) = +Gravity;
		//	//	//gravity(vertex->localIndex * 3) = -Gravity;
		//	//}
		//}
		gravityApplied = true; // 
	} // 撤去力，将重力设为零
	if (frameCount > 100000) {
		gravity.setZero(); // 撤去力，将重力设为零

	}

	// Combine base gravity with user-applied force for this frame.
	Eigen::VectorXf totalForce = gravity;
	if (externalForce.squaredNorm() > 0.0f) {
		for (const auto* vertex : verticesVector) {
			int base = 3 * vertex->localIndex;
			totalForce.segment<3>(base) += externalForce;
		}
	}

	Eigen::VectorXf exfUpdate = timeStep * timeStep * inverseTerm * massMatrix * totalForce;
	Eigen::VectorXf velocityUpdate = inverseTerm * massMatrix * groupVelocity * timeStep;

	for (auto& vertexPair : verticesVector) {
		Vertex* vertex = vertexPair;
		int localPi = vertex->localIndex;
		/*	if (vertexPair->isFixed == true)
			{
				primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = Eigen::Vector3f(vertex->initx, vertex->inity, vertex->initz);
			}
			else
			{
				Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
				Eigen::Vector3f currentExfUpdate = exfUpdate.segment<3>(3 * localPi);
				Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate + currentExfUpdate;
				primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
			}*/
		Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
		Eigen::Vector3f currentExfUpdate = exfUpdate.segment<3>(3 * localPi);
		Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate + currentExfUpdate;
		primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;


	}

}
//

//void Group::calPrimeVec() {
//	primeVec = Eigen::VectorXf::Zero(3 * verticesVector.size());
//
//	if (!gravityApplied) {
//		// 初始化重力向量
//		for (int i = 0; i < 3 * verticesVector.size(); i += 3) {
//			gravity(i + 2) = Gravity; // 重力施加在 y 方向
//		}
//		gravityApplied = true;
//	}
//	
//	// 计算加速度和速度更新
//	Eigen::VectorXf exfUpdate = timeStep * timeStep * inverseTerm * massMatrix * gravity;
//	Eigen::VectorXf velocityUpdate = inverseTerm * massMatrix * groupVelocity * timeStep;
//
//	// 遍历顶点，更新 primeVec
//	for (auto& vertexPair : verticesVector) {
//		Vertex* vertex = vertexPair;
//		int localPi = vertex->localIndex;
//
//		// 检查顶点是否满足 inity > 0.48
//		if (vertex->initz > 1.1) {
//			Eigen::Vector3f currentVelocityUpdate = velocityUpdate.segment<3>(3 * localPi);
//			Eigen::Vector3f currentExfUpdate = exfUpdate.segment<3>(3 * localPi);
//			Eigen::Vector3f newPosition = Eigen::Vector3f(vertex->x, vertex->y, vertex->z) + currentVelocityUpdate + currentExfUpdate;
//			primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = newPosition;
//		}
//		else {
//			// 不满足条件时，保持原位置
//			primeVec.segment<3>(3 * static_cast<Eigen::Index>(localPi)) = Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
//		}
//	}
//	
//}


void Group::calLHS() {
	//A = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK;
	//B = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK * massDistribution;

	float reference = 0.0f; 
	float epsilon = std::numeric_limits<float>::epsilon(); 
	massDampingSparseInv = (massMatrix + timeStep * dampingMatrix).inverse().sparseView(reference, epsilon);
	LHS_A = timeStep * timeStep * massDampingSparseInv * kSparse;
	LHS_B = LHS_A * massDistributionSparse;


	inverseTerm = (massMatrix + dampingMatrix * timeStep).inverse();
	inverseTermSparse = inverseTerm.sparseView();
	RHS_E = timeStep * timeStep * massDampingSparseInv * kSparse;
	RHS_A = RHS_E * initLocalPos;

	FEMLHS = LHS_I + LHS_A - LHS_B;
	FEMLHS_Inv = FEMLHS.inverse();
}
void Group::calLHSFEM() {
	//A = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK;
	//B = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK * massDistribution;

	float reference = 0.0f; 
	float epsilon = std::numeric_limits<float>::epsilon(); // float?型的epsilon
	LHSFEM = (massMatrix.sparseView() + timeStep * dampingMatrix.sparseView() + timeStep * timeStep * kSparseFEM);
}
void Group::calRHS() {

	//Fbind = Eigen::VectorXf::Zero(3 * verticesMap.size());
	//A = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK * initLocalPos;
	//B = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK * rotationMatrix.transpose() * primeVec;
	//C = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * groupK * rotationMatrix.transpose() * massDistribution * primeVec;
	//D = timeStep * timeStep * (massMatrix + timeStep * dampingMatrix).inverse() * rotationMatrix.inverse() * Fbind;
	//rotationTransSparse = rotationMatrix.transpose().sparseView();




	RHS_D = RHS_G * Fbind;
	FEMRHS = RHS_AsubBplusC + RHS_D;

}
void Group::calRHSFEM()
{
	RHSFEM = (massMatrix.sparseView() + timeStep * dampingMatrix.sparseView()) * primeVec + timeStep * timeStep * kSparseFEM * initLocalPos;
}
void Group::calRInvLocalPos() {
	curLocalPos.resize(3 * verticesMap.size());
	RInvPos.resize(3 * verticesMap.size());

	for (const auto& vertexPair : verticesMap) {
		const Vertex* vertex = vertexPair.second;
		Eigen::Vector3f local_position(vertex->x, vertex->y, vertex->z);
		
		Eigen::Vector3f positiondifference = local_position - centerofMass;

		
		curLocalPos.segment<3>(vertex->localIndex * 3) = positiondifference;
	}
	RInvPos = rotationMatrix.inverse() * curLocalPos;
	std::cout << RInvPos << std::endl;
}




void Group::calDeltaX() {

	// Solve Ax=b
	deltaX = FEMLHS_Inv * FEMRHS;
	//deltaX = FEMLHS.colPivHouseholderQr().solve(FEMRHS);

	
	//float threshold = 1e-18;
	//Eigen::SparseMatrix<float> sparseFEMLHS = FEMLHS.sparseView(threshold);

	//Eigen::BiCGSTAB<Eigen::SparseMatrix<float>> solver;
	//solver.compute(sparseFEMLHS);
	//deltaX = solver.solve(FEMRHS);

	deltaX = rotationMatrix * deltaX;
}
void Group::calDeltaXFEM() {// For nonlinear FEM

	
	deltaXFEM = LHSFEM.inverse() * RHSFEM;
}
void Group::calculateCurrentPositions() {
	
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		int localidx;
		localidx = vertex->localIndex;

		
		Eigen::Vector3f primePosition = primeVec.segment<3>(3 * localidx);

		
		Eigen::Vector3f displacement = deltaX.segment<3>(3 * localidx);

		
		Eigen::Vector3f currentPos = primePosition + displacement;
		currentPosition.segment<3>(3 * localidx) = currentPos;

	}
}
void Group::calculateCurrentPositionsFEM() {
	
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		int localidx;
		localidx = vertex->localIndex;


		
		Eigen::Vector3f currentPos = deltaXFEM.segment<3>(3 * localidx);
		currentPositionFEM.segment<3>(3 * localidx) = currentPos;

	}
}
//void Group::updateVertexPositions() {
//	for (auto& vertexPair : verticesMap) {
//		Vertex* vertex = vertexPair.second;
//
//		// 使用局部索引来?取正?的矩??和primeVec部分
//		int localIndex = vertex->localIndex;
//		Eigen::Matrix3f rotationBlock = rotationMatrix.block<3, 3>(3 * localIndex, 3 * localIndex);
//		Eigen::Vector3f positionInPrimeVec = primeVec.segment<3>(3 * localIndex);
//
//		if (vertex->isFixed) {
//			// ?于固定点，将位置?置?初始位置
//			vertex->x = vertex->initx;
//			vertex->y = vertex->inity;
//			vertex->z = vertex->initz;
//		}
//		else {
//			// 使用旋?矩??乘以primeVec中的位置
//			Eigen::Vector3f newPosition = rotationBlock * positionInPrimeVec;
//
//			// 更新?点位置
//			vertex->x = newPosition.x();
//			vertex->y = newPosition.y();
//			vertex->z = newPosition.z();
//		}
//	}
//}

void Group::calFbind1(const std::vector<Vertex*>& commonVerticesGroup1,
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
	int adjacentGroupIdx) {
	
	// XPBD Position Consistency Constraint with Damping Implementation
	// Following the mathematical derivation from bind.txt
	// All matrices are diagonal: M, β, α, γ
	
	// Process all common vertex pairs between groups
	#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(commonVerticesGroup1.size()); ++i) {
		Vertex* vtx_j = commonVerticesGroup1[i];  // 本组顶点
		Vertex* vtx_k = commonVerticesGroup2[i];  // 邻组同物理顶点

		// Get current positions and velocities for both vertices
		Eigen::Vector3f x_j = currentPositionGroup1.segment<3>(3 * vtx_j->localIndex);
		Eigen::Vector3f x_k = currentPositionGroup2.segment<3>(3 * vtx_k->localIndex);
		Eigen::Vector3f v_j = velGroup1.segment<3>(3 * vtx_j->localIndex);
		Eigen::Vector3f v_k = velGroup2.segment<3>(3 * vtx_k->localIndex);
		
		// Position consistency constraint: C(x) = x_j - x_k = 0
		Eigen::Vector3f C = x_j - x_k;              // 约束值
		Eigen::Vector3f Cdot = v_j - v_k;           // 约束速度
		
		// Extract 3x3 mass matrix blocks from group mass matrices
		// M_j from group1's mass matrix (this group)
		Eigen::Matrix3f M_j = massMatrixGroup1.block<3, 3>(3 * vtx_j->localIndex, 3 * vtx_j->localIndex);
		
		// M_k from group2's mass matrix (adjacent group)
		Eigen::Matrix3f M_k = massMatrixGroup2.block<3, 3>(3 * vtx_k->localIndex, 3 * vtx_k->localIndex);
		
		// Ensure numerical stability by checking for zero or very small eigenvalues
		if (M_j.determinant() < 1e-12f) {
			M_j += Eigen::Matrix3f::Identity() * 1e-6f;
		}
		if (M_k.determinant() < 1e-12f) {
			M_k += Eigen::Matrix3f::Identity() * 1e-6f;
		}
		
		Eigen::Matrix3f W_sum = M_j.inverse() + M_k.inverse();  // W = M^(-1)
		
		// Alpha matrix (diagonal compliance) - α = 1/(E·hardness) * I
		Eigen::Matrix3f alpha = Eigen::Matrix3f::Identity() * (1.0f / (youngs * hardness));
		
		// Beta matrix (diagonal damping parameter) - β * I
		Eigen::Matrix3f beta = Eigen::Matrix3f::Identity() * M_j * dampingBeta;
		
		// Gamma matrix (diagonal): γ = α * β
		Eigen::Matrix3f gamma = alpha * beta / timeStep;
		
		// α/Δt² matrix
		Eigen::Matrix3f alphaDivDt2 = alpha / (timeStep * timeStep);
		
		// Thread-safe Lambda access and initialization
		Eigen::Vector3f lambda;
		#pragma omp critical
		{
			auto& constraintMap = constraintLambdas[adjacentGroupIdx];
			if (constraintMap.find(i) == constraintMap.end()) {
				constraintMap[i] = Eigen::Vector3f::Zero();
			}
			lambda = constraintMap[i];
		}
		
		// Gauss-Seidel single constraint update (XPBD-damping)
		// numerator = C + α/Δt² * λ + γ * Cdot * Δt
		Eigen::Vector3f numerator = C + alphaDivDt2 * lambda + gamma * Cdot;
		
		// denominator = (I + γ) * W_sum + α/Δt²
		Eigen::Matrix3f denominator_matrix = (Eigen::Matrix3f::Identity() + gamma) * W_sum + alphaDivDt2;
		
		// Solve for Δλ using matrix inverse: Δλ = -denominator^(-1) * numerator
		Eigen::Vector3f deltaLambda;
		try {
			deltaLambda = -denominator_matrix.inverse() * numerator;
		} catch (...) {
			// Fallback to regularized inverse if singular
			Eigen::Matrix3f regularized = denominator_matrix + Eigen::Matrix3f::Identity() * 1e-6f;
			deltaLambda = -regularized.inverse() * numerator;
		}
		
		// Limit deltaLambda magnitude to prevent explosive behavior
		float deltaLambda_norm = deltaLambda.norm();
		float max_deltaLambda = 1000.0f;
		if (deltaLambda_norm > max_deltaLambda) {
			deltaLambda = deltaLambda * (max_deltaLambda / deltaLambda_norm);
		}
		
		// Thread-safe Lambda update
		#pragma omp critical
		{
			constraintLambdas[adjacentGroupIdx][i] += deltaLambda;
		}
		
		// Position correction: Δx_j = M_j^(-1) * Δλ
		Eigen::Vector3f deltaX_j = M_j.inverse() * deltaLambda;
		
		// Convert to force: F = M * Δx / Δt²
		Eigen::Vector3f forceJ = M_j * deltaX_j / (timeStep * timeStep);
		
		// Additional force magnitude limiting for numerical stability
		/*float force_norm = forceJ.norm();
		float max_force = 100000.0f;
		if (force_norm > max_force) {
			forceJ = forceJ * (max_force / force_norm);
		}*/
		
		// Check for NaN or infinite values
		if (!forceJ.allFinite()) {
			forceJ = Eigen::Vector3f::Zero();
		}
		
		// Optional debug output
		if (debug_constraints && i == 0) {
			std::cout << "Group " << groupIndex << " -> Group " << adjacentGroupIdx 
					  << ": C=" << C.norm()
					  << ", Cdot=" << Cdot.norm()
					  << ", γ_trace=" << gamma.trace()
					  << ", Force=" << forceJ.norm() << std::endl;
		}
		
		// Apply constraint force to this group's vertex
		// (邻组在它自己的 calFbind1 调用里会写入反向力)
		Fbind.segment<3>(3 * vtx_j->localIndex) += forceJ;
	}
}

void Group::calBindFixed() {
	// Fixed vertices: directly set position and velocity to initial values
	// No constraint forces needed - hard enforcement
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		if (vertex->isFixed) {
			// Set current position to initial position
			currentPosition.segment<3>(3 * vertex->localIndex) = 
				Eigen::Vector3f(vertex->initx, vertex->inity, vertex->initz);
			
			// Set velocity to zero (fixed points don't move)
			groupVelocity.segment<3>(3 * vertex->localIndex) = 
				Eigen::Vector3f::Zero();
		}
	}
}

void Group::calFbind(const Eigen::VectorXf& currentPositionThisGroup, const std::vector<Eigen::VectorXf>& allCurrentPositionsOtherGroups, float k) {
	Eigen::Vector3f posThisGroup;
	Eigen::Vector3f posOtherGroup;
	Eigen::Vector3f avgPosition;
	Eigen::Vector3f posDifference;
	Eigen::Vector3f force;
	
	Fbind = Eigen::VectorXf::Zero(currentPositionThisGroup.size()); 

	for (int direction = 0; direction < 6; ++direction) {
		int adjacentGroupIdx = adjacentGroupIDs[direction];

		if (adjacentGroupIdx != -1) {
			const auto& currentPositionOtherGroup = allCurrentPositionsOtherGroups[adjacentGroupIdx];
			const auto& commonVerticesPair = commonVerticesInDirections[direction];

			for (size_t i = 0; i < commonVerticesPair.first.size(); ++i) {
				Vertex* vertexThisGroup = commonVerticesPair.first[i];
				Vertex* vertexOtherGroup = commonVerticesPair.second[i];

				
				posThisGroup = currentPositionThisGroup.segment<3>(3 * vertexThisGroup->localIndex);
				posOtherGroup = currentPositionOtherGroup.segment<3>(3 * vertexOtherGroup->localIndex);
				avgPosition = (posThisGroup + posOtherGroup) / 2;
				posDifference = posThisGroup - avgPosition;
				force = k * posDifference;

				Fbind.segment<3>(3 * vertexThisGroup->localIndex) = force;
			}
		}
	}
}


//

void Group::updatePositionFEM() {
	Eigen::Vector3f pos = Eigen::Vector3f::Zero();
	
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		int localIndex = vertex->localIndex;

		
		pos = deltaXFEM.segment<3>(3 * localIndex);
		/*vertex->x = pos.x();
		vertex->y = pos.y();
		vertex->z = pos.z();*/
		if (vertex->isFixed) {
			
			vertex->x = vertex->initx;
			vertex->y = vertex->inity;
			vertex->z = vertex->initz;
		}
		else {
			


			vertex->x = pos.x();
			vertex->y = pos.y();
			vertex->z = pos.z();
		}

		

	}
}
void Group::updatePosition() {
	float frameTime = 0;
	frameTime += timeStep;
	Eigen::Vector3f pos = Eigen::Vector3f::Zero();
	// ｱ・ﾋﾐ?ｵ・
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		int localIndex = vertex->localIndex;

		if (vertex->isFixed) {
			// Fixed vertices: keep at initial position
			vertex->x = vertex->initx;
			vertex->y = vertex->inity;
			vertex->z = vertex->initz;
			
			// Also ensure currentPosition reflects the fixed position
			currentPosition.segment<3>(3 * localIndex) = 
				Eigen::Vector3f(vertex->initx, vertex->inity, vertex->initz);
		} else {
			pos = currentPosition.segment<3>(3 * localIndex);
			
			// 位置平滑处理 - 防止突然的大幅度跳变（仅限制极端情况）
			Eigen::Vector3f oldPos(vertex->x, vertex->y, vertex->z);
			Eigen::Vector3f displacement = pos - oldPos;
			float maxDisplacement = 5.0f;  // 单帧最大位移限制（放宽以保持性能）
			float dispMag = displacement.norm();
			if (dispMag > maxDisplacement) {
				pos = oldPos + displacement * (maxDisplacement / dispMag);
			}
			
			vertex->x = pos.x();
			vertex->y = pos.y();
			vertex->z = pos.z();
		}
		/*vertex->x = pos.x();
		vertex->y = pos.y();
		vertex->z = pos.z();*/
	/*	if (vertex->initx > -0.01 && vertex->initx < 0.01) {
			vertex->y = vertex->inity + (45 * frameTime);
			vertex->x = pos.x();
			vertex->z = pos.z();
		}*/
		//else if (vertex->inity < -0.52)
		//{
		//	vertex->y = vertex->inity + (9 * frameTime);
		//	/*vertex->x = pos.x();
		//	vertex->z = pos.z();*/
		//} 	
		//else {
		//	// ﾊｹﾓﾃﾐ?ｾﾘ??ｳﾋﾒﾔprimeVecﾖﾐｵﾄﾎｻﾖﾃ


		//	vertex->x = pos.x();
		//	vertex->y = pos.y();
		//	vertex->z = pos.z();
		//}

		/* ｸ・ﾂ?ｵ羞ﾄﾎｻﾖﾃ
		if (vertex->isFixed == true)
		{
			vertex->x += 0.01;
		}*/
	}
}

void Group::updateVelocity() {
	Eigen::Vector3f previousPos = Eigen::Vector3f::Zero();
	Eigen::Vector3f currentPos = Eigen::Vector3f::Zero();
	Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
	Kinematics = 0.0;

	
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		int localIndex = vertex->localIndex;
		if (vertex->isFixed) {
			// Fixed vertices: set velocity to zero
			groupVelocity.segment<3>(3 * localIndex) = Eigen::Vector3f::Zero();
		} else {
			previousPos.x() = vertex->x;
			previousPos.y() = vertex->y;
			previousPos.z() = vertex->z;


			currentPos = currentPosition.segment<3>(3 * localIndex);


			velocity = (currentPos - previousPos) / timeStep;
			
			// 速度限制 - 防止数值不稳定导致的抖动（仅限制极端情况）
			float maxVelocity = 100.0f;  // 最大速度限制
			float velocityMag = velocity.norm();
			if (velocityMag > maxVelocity) {
				velocity = velocity * (maxVelocity / velocityMag);
			}
			
			groupVelocity.segment<3>(3 * localIndex) = velocity;

			//Kinematics += 0.5 * vertex->vertexMass * velocity.norm();
		}
		/*std::cout << "\r";
		std::cout << groupVelocity << std::endl;*/
		

		// 更新 vertex 的速度
		// 例如：vertex->velocity = velocity;

		// 保存当前位置作?下一?的“上一?位置”
		//previousPosition.segment<3>(3 * localIndex) = currentPos;
	}

	//// 更新currentPosition?本?最后的位置
	//for (auto& vertexPair : verticesMap) {
	//	Vertex* vertex = vertexPair.second;
	//	int localIndex = vertex->localIndex;

	//	currentPosition.segment<3>(3 * localIndex) = Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
	//}
}
void Group::updateVelocityFEM() {
	Eigen::Vector3f previousPos = Eigen::Vector3f::Zero();
	Eigen::Vector3f currentPos = Eigen::Vector3f::Zero();
	Eigen::Vector3f velocity = Eigen::Vector3f::Zero();

	// 遍?所有?点，更新速度并保存当前位置
	for (auto& vertexPair : verticesMap) {
		Vertex* vertex = vertexPair.second;
		int localIndex = vertex->localIndex;

		// ?取当前位置
		previousPos.x() = vertex->x;
		previousPos.y() = vertex->y;
		previousPos.z() = vertex->z;

		// 从 previousPosition ?取上一?的位置
		currentPos = currentPositionFEM.segment<3>(3 * localIndex);

		// ?算速度
		velocity = (currentPos - previousPos) / timeStep;
		groupVelocityFEM.segment<3>(3 * localIndex) = velocity;
		// 更新 vertex 的速度
		// 例如：vertex->velocity = velocity;

		// 保存当前位置作?下一?的“上一?位置”
		//previousPosition.segment<3>(3 * localIndex) = currentPos;
	}

	//// 更新currentPosition?本?最后的位置
	//for (auto& vertexPair : verticesMap) {
	//	Vertex* vertex = vertexPair.second;
	//	int localIndex = vertex->localIndex;

	//	currentPosition.segment<3>(3 * localIndex) = Eigen::Vector3f(vertex->x, vertex->y, vertex->z);
	//}
}
