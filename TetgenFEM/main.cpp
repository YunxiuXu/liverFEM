
//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>  // for std::strcpy
#include "tetgen.h"  // Include the TetGen header file
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <filesystem>
#include <chrono>
#include <ctime>
#include "GL/glew.h" 
#include "GLFW/glfw3.h"
#include "params.h"
#include <cmath>
#include <random>
#include <omp.h>
#include "VisualOpenGL.h"
#include "SimpleUI.h"
#include "ReadSTL.h"
#include "Object.h"
#include "Vertex.h"
#include "Edge.h"
#include "AgentSphereContact.h"
#include "Experiment3.h"
#include "Experiment1.h"
#include "Experiment2.h"
#include "Experiment4.h"



//C:/Users/xu_yu/Desktop/tmp/arial.ttf
 
// Global variables

// Force recording variables
bool isRecordingForce = false;
std::vector<float> recordedForces;
std::vector<float> recordedTime;
double recordStartTime = 0.0;

// Benchmark Mode Variables
bool isBenchmarkActive = false;
std::vector<Vertex*> benchmarkVertices;
const double benchmarkForceMag = 10.0f; // Newtons
Eigen::Vector3f benchmarkForceDir(1.0f, 0.0f, 0.0f); // Direction
double benchmarkStartTime = 0.0;

// Auto-Test Variables (Restored)
bool isAutoTestActive = false;
int autoTestAxis = 0; // 0:X, 1:Y
double autoTestStartTime = 0.0;
const double autoTestDuration = 2.0;
const float autoTestDistance = 0.5f;
Vertex* g_selectedVertex = nullptr;
Eigen::Vector3f autoTestStartPos;

namespace {
struct KeyLatch {
	bool latched = false;
	bool consume(GLFWwindow* window, int key) {
		const bool down = glfwGetKey(window, key) == GLFW_PRESS;
		if (down && !latched) {
			latched = true;
			return true;
		}
		if (!down) {
			latched = false;
		}
		return false;
	}
};

static std::string formatSignedInt(float v)
{
	const long long iv = static_cast<long long>(std::llround(static_cast<double>(v)));
	return std::to_string(iv);
}

static void drawWireSphereCircles(const Eigen::Vector3f& center, float radius, int segments)
{
	const int seg = std::max(8, segments);
	const float r = std::max(0.0f, radius);
	auto drawCircle = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < seg; ++i) {
			const float t = (2.0f * PI) * (static_cast<float>(i) / static_cast<float>(seg));
			const float ct = std::cos(t);
			const float st = std::sin(t);
			const Eigen::Vector3f p = center + r * (ct * a + st * b);
			glVertex3f(p.x(), p.y(), p.z());
		}
		glEnd();
	};
	drawCircle(Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY());
	drawCircle(Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitZ());
	drawCircle(Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ());
}
} // namespace

void saveForceData(const std::string& filename) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open force data file.\n";
		return;
	}
	file << "Time(s) ForceMagnitude\n";
	for (size_t i = 0; i < recordedForces.size(); ++i) {
		file << recordedTime[i] << " " << recordedForces[i] << "\n";
	}
	file.close();
	std::cout << "Force data saved to " << filename << " (" << recordedForces.size() << " samples)\n";
}

void saveOBJ(const std::string& filename, std::vector<Group>& groups) {
	std::ofstream objFile(filename);
	if (!objFile.is_open()) {
		std::cerr << "Failed to open file for writing.\n";
		return;
	}

	std::unordered_map<Vertex*, int> vertexIndexMap;
	int currentIndex = 1;

	// 遍历组，找出所有边界边并记录其顶点
	for (const auto& group : groups) {
		for (const auto* tet : group.tetrahedra) {
			for (const auto* edge : tet->edges) {
				if (edge->isBoundary) {
					for (Vertex* vertex : edge->vertices) {
						if (vertexIndexMap.find(vertex) == vertexIndexMap.end()) {
							vertexIndexMap[vertex] = currentIndex++;
							objFile << "v " << vertex->x << " " << vertex->y << " " << vertex->z << "\n";
						}
					}
				}
			}
		}
	}

	// 再次遍历，这次是为了构建面
	for (const auto& group : groups) {
		for (const auto* tet : group.tetrahedra) {
			for (const auto* edge : tet->edges) {
				if (edge->isBoundary) {
					objFile << "f";
					for (Vertex* vertex : edge->vertices) {
						objFile << " " << vertexIndexMap[vertex];
					}
					objFile << "\n";
				}
			}
		}
	}

	objFile.close();
	std::cout << "OBJ file saved: " << filename << "\n";
}

namespace {
struct TetgenExportPaths {
	std::string nodePath;
	std::string elePath;
	std::string nodePathAbs;
	std::string elePathAbs;
};

static std::string nowTimestampForFilename()
{
	const auto now = std::chrono::system_clock::now();
	const std::time_t t = std::chrono::system_clock::to_time_t(now);
	std::tm tm{};
#if defined(_WIN32)
	localtime_s(&tm, &t);
#else
	localtime_r(&t, &tm);
#endif
	char buf[32];
	std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
	return std::string(buf);
}

static std::string tryAbsolutePath(const std::string& path)
{
	try {
		return std::filesystem::absolute(std::filesystem::path(path)).string();
	}
	catch (...) {
		return path;
	}
}

static TetgenExportPaths exportTetgenNodeEleSnapshot(
	const Object& object,
	const std::vector<Vertex*>& objectUniqueVertices,
	const std::string& outDir,
	const std::string& baseName)
{
	std::filesystem::create_directories(outDir);

	const std::string nodePath = (std::filesystem::path(outDir) / (baseName + ".node")).string();
	const std::string elePath = (std::filesystem::path(outDir) / (baseName + ".ele")).string();

	std::unordered_map<const Vertex*, int> exportIndex;
	exportIndex.reserve(objectUniqueVertices.size());
	for (size_t i = 0; i < objectUniqueVertices.size(); ++i) {
		exportIndex[objectUniqueVertices[i]] = static_cast<int>(i + 1); // 1-based for TetGen
	}

	// Write .node
	{
		std::ofstream nodeFile(nodePath);
		if (!nodeFile.is_open()) {
			throw std::runtime_error("Failed to open for writing: " + nodePath);
		}
		nodeFile << objectUniqueVertices.size() << " 3 0 0\n";
		for (size_t i = 0; i < objectUniqueVertices.size(); ++i) {
			const Vertex* v = objectUniqueVertices[i];
			nodeFile << (i + 1) << " " << v->x << " " << v->y << " " << v->z << "\n";
		}
	}

	// Count and write .ele
	size_t tetCount = 0;
	for (const auto& g : object.groups) {
		tetCount += g.tetrahedra.size();
	}
	{
		std::ofstream eleFile(elePath);
		if (!eleFile.is_open()) {
			throw std::runtime_error("Failed to open for writing: " + elePath);
		}
		eleFile << tetCount << " 4 0\n";
		size_t tetIndex = 1;
		for (const auto& g : object.groups) {
			for (const auto* tet : g.tetrahedra) {
				const int a = exportIndex.at(tet->vertices[0]);
				const int b = exportIndex.at(tet->vertices[1]);
				const int c = exportIndex.at(tet->vertices[2]);
				const int d = exportIndex.at(tet->vertices[3]);
				eleFile << tetIndex++ << " " << a << " " << b << " " << c << " " << d << "\n";
			}
		}
	}

	TetgenExportPaths paths;
	paths.nodePath = nodePath;
	paths.elePath = elePath;
	paths.nodePathAbs = tryAbsolutePath(nodePath);
	paths.elePathAbs = tryAbsolutePath(elePath);
	return paths;
}
} // namespace

void writeOBJ(const Object& object, const std::string& filename) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open file for writing.\n";
		return;
	}

	int vertexIndexOffset = 1;
	std::unordered_map<Vertex*, int> vertexIndexMap;

	for (const auto& group : object.groups) {
		for (const auto& tetrahedron : group.tetrahedra) {
			// vertex
			for (int i = 0; i < 4; ++i) {
				Vertex* vertex = tetrahedron->vertices[i];
				if (vertexIndexMap.find(vertex) == vertexIndexMap.end()) {
					vertexIndexMap[vertex] = vertexIndexOffset++;
					file << "v " << vertex->x << " " << vertex->y << " " << vertex->z << "\n";
				}
			}

			// writing faces
			// indices of four faces
			int indices[4][3] = { {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3} };
			for (int i = 0; i < 4; ++i) {
				file << "f";
				for (int j = 0; j < 3; ++j) {
					file << " " << vertexIndexMap[tetrahedron->vertices[indices[i][j]]];
				}
				file << "\n";
			}
		}
	}

	file.close();
	std::cout << "OBJ file has been written.\n";
}
void writeSTL(const Object& object, const std::string& filename) {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file for writing.\n";
		return;
	}

	// Write 80-byte header
	char header[80];
	memset(header, 0, sizeof(header)); // Fill header with zeros
	std::string description = "Binary STL generated by writeSTL function";
	std::memcpy(header, description.c_str(), std::min(description.size(), sizeof(header)));
	file.write(header, sizeof(header));

	// Count total number of triangles
	uint32_t totalTriangles = 0;
	for (const auto& group : object.groups) {
		totalTriangles += static_cast<uint32_t>(group.tetrahedra.size()) * 4u; // 4 faces per tetrahedron
	}
	file.write(reinterpret_cast<char*>(&totalTriangles), sizeof(totalTriangles));

	// Write triangles
	for (const auto& group : object.groups) {
		for (const auto& tetrahedron : group.tetrahedra) {
			// Indices for the four triangular faces of a tetrahedron
			int indices[4][3] = { {0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3} };

			for (int i = 0; i < 4; ++i) {
				// Write normal vector (defaulting to 0,0,0)
				float normal[3] = { 0.0f, 0.0f, 0.0f };
				file.write(reinterpret_cast<char*>(normal), sizeof(normal));

				// Write vertices of the triangle
				for (int j = 0; j < 3; ++j) {
					Vertex* vertex = tetrahedron->vertices[indices[i][j]];
					float vertexCoords[3] = { static_cast<float>(vertex->x),
											 static_cast<float>(vertex->y),
											 static_cast<float>(vertex->z) };
					file.write(reinterpret_cast<char*>(vertexCoords), sizeof(vertexCoords));
				}

				// Write attribute byte count (2 bytes, set to 0)
				uint16_t attributeByteCount = 0;
				file.write(reinterpret_cast<char*>(&attributeByteCount), sizeof(attributeByteCount));
			}
		}
	}

	file.close();
	std::cout << "Binary STL file has been written.\n";
}
void findTopAndBottomVertices(const std::vector<Group>& groups, std::vector<int>& topVertexLocalIndices, std::vector<int>& bottomVertexLocalIndices) {
	for (const Group& g : groups) {
		for (const auto& vertexPair : g.verticesMap) {
			Vertex* vertex = vertexPair.second;
			if (vertex->inity > 0.53) {
				topVertexLocalIndices.push_back(vertex->index);
			}
			if (vertex->inity < -0.53) {
				bottomVertexLocalIndices.push_back(vertex->index);
			}
		}
	}
}
void findMaxAndMinYVertices(const std::vector<Group>& groups, int& maxYVertexIndex, int& minYVertexIndex) {
	bool isFirstVertex = true;
	double maxY = 0.0;
	double minY = 0.0;

	for (const Group& g : groups) {
		for (const auto& vertexPair : g.verticesMap) {
			Vertex* vertex = vertexPair.second;

			if (isFirstVertex) {
				// Initialize maxY and minY with the first vertex's y-coordinate
				maxY = minY = vertex->inity;
				maxYVertexIndex = minYVertexIndex = vertex->index;
				isFirstVertex = false;
			}
			else {
				if (vertex->inity > maxY) {
					maxY = vertex->inity;
					maxYVertexIndex = vertex->index;
				}
				if (vertex->inity < minY) {
					minY = vertex->inity;
					minYVertexIndex = vertex->index;
				}
			}
		}
	}
}

void findUpperAndLowerVertices(const std::vector<Group>& groups, std::vector<int>& upperVertices, std::vector<int>& lowerVertices) {
	double sumInity = 0.0;
	int count = 0;

	// First pass: calculate the average inity
	for (const Group& g : groups) {
		for (const auto& vertexPair : g.verticesMap) {
			Vertex* vertex = vertexPair.second;
			sumInity += vertex->inity;
			count++;
		}
	}

	double averageInity = sumInity / count;

	// Second pass: classify vertices based on average inity
	for (const Group& g : groups) {
		for (const auto& vertexPair : g.verticesMap) {
			Vertex* vertex = vertexPair.second;
			if (vertex->inity > averageInity) {
				upperVertices.push_back(vertex->index);
			}
			else {
				lowerVertices.push_back(vertex->index);
			}
		}
	}
}

struct DragState {
	bool active = false;
	Vertex* target = nullptr;
	double lastX = 0.0;
	double lastY = 0.0;
	float grabbedNdcZ = 0.0f;
	Eigen::Vector3f grabOffset = Eigen::Vector3f::Zero(); // targetPos - cursorWorldPos (prevents jump)
};

Eigen::Vector2f projectToScreen(const Eigen::Vector3f& pos,
	const Eigen::Matrix4f& model,
	const Eigen::Matrix4f& projection,
	int width,
	int height) {
	Eigen::Vector4f clip = projection * model * Eigen::Vector4f(pos.x(), pos.y(), pos.z(), 1.0f);
	Eigen::Vector3f ndc = clip.head<3>() / clip.w();
	float sx = (ndc.x() * 0.5f + 0.5f) * static_cast<float>(width);
	float sy = (1.0f - (ndc.y() * 0.5f + 0.5f)) * static_cast<float>(height);
	return Eigen::Vector2f(sx, sy);
}

Vertex* pickVertexAtCursor(const std::vector<Vertex*>& vertices,
	double mouseX,
	double mouseY,
	const Eigen::Matrix4f& model,
	const Eigen::Matrix4f& projection,
	int width,
	int height,
	float maxScreenDistance = 60.0f) {
	float bestDist2 = std::numeric_limits<float>::max();
	Vertex* bestVertex = nullptr;

	for (const auto* vertex : vertices) {
		Eigen::Vector2f screenPos = projectToScreen(Eigen::Vector3f(vertex->x, vertex->y, vertex->z), model, projection, width, height);
		float dx = static_cast<float>(mouseX) - screenPos.x();
		float dy = static_cast<float>(mouseY) - screenPos.y();
		float dist2 = dx * dx + dy * dy;
		if (dist2 < bestDist2) {
			bestDist2 = dist2;
			bestVertex = const_cast<Vertex*>(vertex);
		}
	}

	if (bestVertex && bestDist2 <= maxScreenDistance * maxScreenDistance) {
		return bestVertex;
	}
	return nullptr;
}

static Eigen::Vector3f unprojectCursorToWorld(double fbMouseX,
	double fbMouseY,
	float ndcZ,
	const Eigen::Matrix4f& invProjectionModel,
	int width,
	int height) {
	const float safeW = static_cast<float>(width ? width : 1);
	const float safeH = static_cast<float>(height ? height : 1);
	const float ndcX = static_cast<float>(fbMouseX) / safeW * 2.0f - 1.0f;
	const float ndcY = 1.0f - static_cast<float>(fbMouseY) / safeH * 2.0f;

	Eigen::Vector4f clip(ndcX, ndcY, ndcZ, 1.0f);
	Eigen::Vector4f world = invProjectionModel * clip;
	const float invW = (std::abs(world.w()) > 1e-8f) ? (1.0f / world.w()) : 1.0f;
	return world.head<3>() * invW;
}

int main(int argc, char** argv) {

	bool exportTetgenAndExit = false;
	std::string exportDirOverride;
	loadParams("parameters.txt");
	// Make sure both OpenMP and Eigen use all available cores
	omp_set_dynamic(0);
	omp_set_num_threads(std::max(1, omp_get_num_procs()));
	Eigen::initParallel();
	Eigen::setNbThreads(std::max(1, omp_get_num_procs()));

	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--export-tetgen") {
			exportTetgenAndExit = true;
			continue;
		}
		if (std::string(argv[i]) == "--export-dir" && i + 1 < argc) {
			exportDirOverride = argv[++i];
			continue;
		}
		if (std::string(argv[i]) == "--exp4") {
			Experiment4& experiment4 = Experiment4::instance();
			experiment4.requestStart();
			// Two updates: first transitions Idle->PendingStart, second runs benchmarks.
			experiment4.update();
			experiment4.update();
			return 0;
		}
	}

	tetgenio in, out;
	in.firstnumber = 1;  // All indices start from 1
	
	if (useDirectLoading) {
		// Direct loading mode: load node and element files without meshing
		std::cout << "Using direct loading mode with node file: " << nodeFile << " and element file: " << eleFile << std::endl;
		
		// Extract base filename without extension for TetGen (it will append .node and .ele automatically)
		std::string nodeFileBase = nodeFile;
		size_t nodeExtPos = nodeFileBase.find_last_of('.');
		if (nodeExtPos != std::string::npos) {
			nodeFileBase = nodeFileBase.substr(0, nodeExtPos);
		}
		
		std::string eleFileBase = eleFile;
		size_t eleExtPos = eleFileBase.find_last_of('.');
		if (eleExtPos != std::string::npos) {
			eleFileBase = eleFileBase.substr(0, eleExtPos);
		}
		
		char* nodeFileC = const_cast<char*>(nodeFileBase.c_str());
		char* eleFileC = const_cast<char*>(eleFileBase.c_str());
		
		std::cout << "Loading base filename: " << nodeFileBase << " (TetGen will append .node/.ele)" << std::endl;
		
		if (!in.load_node(nodeFileC)) {
			std::cerr << "Error loading .node file: " << nodeFileBase << ".node" << std::endl;
			return 1;
		}
		
		if (!in.load_tet(eleFileC)) {
			std::cerr << "Error loading .ele file: " << eleFileBase << ".ele" << std::endl;
			return 1;
		}
		
		// Copy input directly to output without meshing
		out = in;
	} else {
		// STL meshing mode: load STL file and use TetGen for meshing
		std::cout << "Using STL meshing mode with file: " << stlFile << std::endl;
		
		readSTL(stlFile.c_str(), in);
		
		// Configure TetGen behavior
		tetgenbehavior behavior;
		char* args = const_cast<char*>(tetgenArgs.c_str());
		behavior.parse_commandline(args);
		
		// Call TetGen to tetrahedralize the geometry
		tetrahedralize(&behavior, &in, &out);
	}
	



	Object object;
	groupNum = groupNumX * groupNumY * groupNumZ;
	object.groupNum = groupNum;
	object.groupNumX = groupNumX;
	object.groupNumY = groupNumY;
	object.groupNumZ = groupNumZ;
	divideIntoGroups(out, object, groupNumX, groupNumY, groupNumZ); //convert tetgen to our data structure

	// Use TetGen's native save function to save the initial mesh
	// This is robust and avoids issues with vertex duplication/deduplication in the Object structure
	if (!exportTetgenAndExit && autoSaveMesh) {
		std::string exportDir = exportDirOverride.empty() ? "out/tetgenfem_exports" : exportDirOverride;
		std::filesystem::create_directories(exportDir);
		
		std::string basePath = (std::filesystem::path(exportDir) / "latest").string();
		std::cout << "[TetgenFEM] Saving initial mesh using TetGen native functions to: " << basePath << ".*" << std::endl;
		
		// TetGen's save functions take a char* base name and append extension
		// Cast to char* is safe here because we're just passing the buffer address
		char* basePathC = const_cast<char*>(basePath.c_str());
		out.save_nodes(basePathC);
		out.save_elements(basePathC);
	}

	/*out.save_nodes("vbdbeam");
	out.save_elements("vbdbeam");*/
	//writeSTL(object, "vbdbeam.stl");
	//writeOBJ(object, "vbdbeam.obj");


	object.updateIndices(); 
	object.assignLocalIndicesToAllGroups(); 
	object.generateUniqueVertices();
	
	object.updateAdjacentGroupIndices(groupNumX, groupNumY, groupNumZ);
	for (int i = 0; i < groupNum; ++i) {
	
		object.storeAdjacentGroupsCommonVertices(i);
	}
	
	// Accessing and printing the groups and their tetrahedra
//#pragma omp parallel for
	int nonEmptyGroupCount = 0;
	for (int i = 0; i < groupNum; ++i) {  // Loop over the groups
		Group& group = object.getGroup(i);
		group.LHS_I = Eigen::MatrixXf::Identity(3 * group.verticesMap.size(), 3 * group.verticesMap.size()); //ｽﾚﾊ｡ﾊｱｼ菻｡ﾄﾜﾊﾖ
		if (group.tetrahedra.empty()) {
			continue; // Skip noisy logging for empty groups
		}
		++nonEmptyGroupCount;
		std::cout << "Group " << i << " has " << group.tetrahedra.size() << " tetrahedra." << std::endl;
	}
	std::cout << "Non-empty groups: " << nonEmptyGroupCount << "/" << groupNum << std::endl;


	// Initialize the GLFW library
	if (!glfwInit()) {
		return -1;
	}

	// Create a windowed mode window and its OpenGL context
	GLFWwindow* window = glfwCreateWindow(1080, 1080, "Tetrahedral Mesh Visualization", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);
	// Set scroll callback
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, cursorPosCallback);
	int fbWidth = 0;
	int fbHeight = 0;
	glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
	framebuffer_size_callback(window, fbWidth, fbHeight);
	applyProjectionMatrix();

	Eigen::Matrix4f mat;
	initFontData();
	//object.findCommonVertices();
	//object.commonPoints = object.findCommonVertices1(object.groups[0], object.groups[1]);
	//object.commonPoints1 = object.findCommonVertices1(object.groups[1], object.groups[2]);
	//object.commonPoints2 = object.findCommonVertices1(object.groups[2], object.groups[3]);
	//object.commonPoints3 = object.findCommonVertices1(object.groups[3], object.groups[4]);
	//std::pair<std::vector<Vertex*>, std::vector<Vertex*>> commonVertices2 = object.findCommonVertices1(object.groups[0], object.groups[1]);
	// 在肝门区域使用球形区域进行固定：取背面一定厚度的质心作为中心
	std::unordered_set<Vertex*> visitedVertices;
	std::vector<Vertex*> uniqueVertices;
	for (const auto& g : object.groups) {
		for (const auto& pair : g.verticesMap) {
			Vertex* v = pair.second;
			if (visitedVertices.insert(v).second) {
				uniqueVertices.push_back(v);
			}
		}
	}

	if (!uniqueVertices.empty()) {
		Eigen::Vector3f minBound(
			std::numeric_limits<float>::max(),
			std::numeric_limits<float>::max(),
			std::numeric_limits<float>::max());
		Eigen::Vector3f maxBound(
			-std::numeric_limits<float>::max(),
			-std::numeric_limits<float>::max(),
			-std::numeric_limits<float>::max());

		for (const auto* v : uniqueVertices) {
			minBound.x() = std::min(minBound.x(), v->initx);
			minBound.y() = std::min(minBound.y(), v->inity);
			minBound.z() = std::min(minBound.z(), v->initz);
			maxBound.x() = std::max(maxBound.x(), v->initx);
			maxBound.y() = std::max(maxBound.y(), v->inity);
			maxBound.z() = std::max(maxBound.z(), v->initz);
		}

		const float depth = maxBound.z() - minBound.z();
		const float backSliceZ = minBound.z() + depth * 0.12f; // 取最靠背面 12% 的区域
		Eigen::Vector3f backCentroid(0.0f, 0.0f, 0.0f);
		int backCount = 0;
		for (const auto* v : uniqueVertices) {
			if (v->initz <= backSliceZ) {
				backCentroid.x() += v->initx;
				backCentroid.y() += v->inity;
				backCentroid.z() += v->initz;
				++backCount;
			}
		}

		if (backCount > 0) {
			backCentroid /= static_cast<float>(backCount);
		}
		else {
			backCentroid = Eigen::Vector3f(
				0.5f * (minBound.x() + maxBound.x()),
				0.5f * (minBound.y() + maxBound.y()),
				minBound.z());
		}

		// 稍微往内部推一点，避免只作用在表面三角面
		backCentroid.z() = std::min(backCentroid.z() + depth * 0.02f, maxBound.z());

		const float anchorRadius = std::max(depth * 0.2f, 0.001f);
		object.fixRegion(backCentroid, anchorRadius);
	} else {
		std::cout << "No vertices collected for fixing region." << std::endl;
	}
	
	std::vector<int> topVertexLocalIndices;
	std::vector<int> bottomVertexLocalIndices;

	findTopAndBottomVertices(object.groups, topVertexLocalIndices, bottomVertexLocalIndices);
	int maxYIndex, minYIndex;
	findMaxAndMinYVertices(object.groups, maxYIndex, minYIndex);


	// Now topVertexLocalIndices and bottomVertexLocalIndices contain the local indices of the top and bottom vertices, respectively.

	
	//Fix by several vertices
	//float maxY = -std::numeric_limits<float>::infinity();
	//Vertex* vertexWithMaxY = nullptr;
	
	//for (Group& g : object.groups) {
	//	for (const auto& vertexPair : g.verticesMap) {
	//		Vertex* vertex = vertexPair.second;
	//		if (vertex->y > maxY) {
	//			maxY = vertex->y;
	//			vertexWithMaxY = vertex;
	//		}
	//	}
	//}
	
	//if (vertexWithMaxY != nullptr) {
	//	vertexWithMaxY->isFixed = true;
	//	
	//}
	/////////
	
#pragma omp parallel for
	for (int i = 0; i < object.groupNum; ++i) {
		object.groups[i].calMassMatrix(density);
		object.groups[i].calDampingMatrix();
		object.groups[i].calCenterofMass();
		object.groups[i].calInitCOM();//initial com
		object.groups[i].calLocalPos(); // initial local positions
		
		// Check if anisotropic parameters are used (youngs1 != youngs2)
		// Assuming youngs1, youngs2, youngs3 are global variables from params.h
		if (std::abs(youngs1 - youngs2) > 1e-1f || std::abs(youngs1 - youngs3) > 1e-1f) {
			object.groups[i].calGroupKAni(youngs1, youngs2, youngs3, poisson);
			if (i == 0) std::cout << "Using Anisotropic Stiffness Matrix (E1=" << youngs1 << ", E2=" << youngs2 << ", E3=" << youngs3 << ")\n";
		} else {
			object.groups[i].calGroupK(youngs, poisson);
			if (i == 0) std::cout << "Using Isotropic Stiffness Matrix (E=" << youngs << ")\n";
		}
		
		object.groups[i].setVertexMassesFromMassMatrix();//vertex mass
		object.groups[i].calMassGroup();
		object.groups[i].calMassDistributionMatrix();
		//object.groups[i].inverseTerm = (object.groups[i].massMatrix + object.groups[i].dampingMatrix * 0.01f).inverse(); 
		//object.groups[i].inverseTermSparse = object.groups[i].inverseTerm.sparseView();
		object.groups[i].calLHS();
	}

	//for calculate frame rate
	double lastTime = glfwGetTime();
	int nbFrames = 0;
	glfwSwapInterval(0);


	//------------------- save coordinates
	std::vector<Vertex*> objectUniqueVertices;

	// Optimization: Collect all vertices first, then sort and unique to avoid O(N^2) complexity
	size_t estimatedCount = 0;
	for (int i = 0; i < groupNum; ++i) {
		estimatedCount += object.getGroup(i).verticesMap.size();
	}
	objectUniqueVertices.reserve(estimatedCount);

	for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
		Group& group = object.getGroup(groupIdx);
		for (const auto& pair : group.verticesMap) {
			objectUniqueVertices.push_back(pair.second);
		}
	}

	// Sort by initial coordinates to identify duplicates
	std::sort(objectUniqueVertices.begin(), objectUniqueVertices.end(), [](const Vertex* a, const Vertex* b) {
		if (std::abs(a->initx - b->initx) > 1e-6f) return a->initx < b->initx;
		if (std::abs(a->inity - b->inity) > 1e-6f) return a->inity < b->inity;
		return a->initz < b->initz;
	});

	// Remove duplicates based on initial coordinates
	auto last = std::unique(objectUniqueVertices.begin(), objectUniqueVertices.end(), [](const Vertex* a, const Vertex* b) {
		return std::abs(a->initx - b->initx) <= 1e-6f &&
			   std::abs(a->inity - b->inity) <= 1e-6f &&
			   std::abs(a->initz - b->initz) <= 1e-6f;
	});
	objectUniqueVertices.erase(last, objectUniqueVertices.end());

	std::sort(objectUniqueVertices.begin(), objectUniqueVertices.end(), [](const Vertex* a, const Vertex* b) {
		return a->index < b->index;
		});//index from min to max

	// ------------------ Agent sphere ("finger") setup
	Eigen::Vector3f bboxMin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Eigen::Vector3f bboxMax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	for (const auto* v : objectUniqueVertices) {
		bboxMin.x() = std::min(bboxMin.x(), v->initx);
		bboxMin.y() = std::min(bboxMin.y(), v->inity);
		bboxMin.z() = std::min(bboxMin.z(), v->initz);
		bboxMax.x() = std::max(bboxMax.x(), v->initx);
		bboxMax.y() = std::max(bboxMax.y(), v->inity);
		bboxMax.z() = std::max(bboxMax.z(), v->initz);
	}
	const Eigen::Vector3f bboxCenter = 0.5f * (bboxMin + bboxMax);
	const float bboxDiag = (bboxMax - bboxMin).norm();

	AgentSphere agentSphere;
	agentSphere.enabled = agentEnabled;
	agentSphere.radius = std::max(1e-6f, agentRadiusBboxScale * bboxDiag);
	agentSphere.contactStiffness = agentContactStiffness;
	agentSphere.contactDamping = agentContactDamping;
	agentSphere.position = Eigen::Vector3f(bboxCenter.x(), bboxMax.y() + agentSphere.radius * 1.5f, bboxCenter.z());

	Eigen::Vector3f agentDevicePosition = agentSphere.position;
	Eigen::Vector3f agentDevicePrevPosition = agentDevicePosition;
	Eigen::Vector3f agentDeviceVelocity = Eigen::Vector3f::Zero();

	Eigen::Vector3f agentProxyPosition = agentSphere.position;
	Eigen::Vector3f agentProxyVelocity = Eigen::Vector3f::Zero();

	Eigen::Vector3f agentHomePosition = agentDevicePosition;

	Eigen::Vector3f agentLastDeviceForceN = Eigen::Vector3f::Zero();   // what you'd send to a haptic device
	Eigen::Vector3f agentLastContactForceN = Eigen::Vector3f::Zero();  // reaction on proxy from tissue
	Eigen::Vector3f agentLastCouplingForceN = Eigen::Vector3f::Zero(); // device<->proxy spring force (on proxy)
	int agentLastContactCount = 0;

	float objectMassKg = 0.0f;
	for (const auto* v : objectUniqueVertices) {
		objectMassKg += std::max(0.0f, v->vertexMass);
	}
	const float agentProxyMassKg = std::max(1e-6f, std::abs(agentProxyMassFracOfObject) * std::max(1e-6f, objectMassKg));
	const float invBboxDiag = 1.0f / std::max(1e-6f, bboxDiag);
	const float agentVcKLen = std::max(0.0f, agentVcStiffnessNPerBbox) * invBboxDiag;     // N per unit length
	const float agentVcCLen = std::max(0.0f, agentVcDampingNsPerBbox) * invBboxDiag;      // N*s per unit length
	const float agentProxyCorr = std::clamp(agentProxyPositionCorrection, 0.0f, 1.0f);

	// Choose contact vertices: surface vertices (TetGen trifaces) if available, otherwise all unique vertices.
	std::vector<Vertex*> agentContactVertices;
	std::vector<AgentTriangle> agentContactTriangles;
	agentContactVertices.reserve(objectUniqueVertices.size());
	{
		int maxIndex = 0;
		for (const auto* v : objectUniqueVertices) maxIndex = std::max(maxIndex, v->index);

		// Build an index->Vertex* lookup for triangle contact.
		std::vector<Vertex*> indexToVertex;
		if (maxIndex >= 0) {
			indexToVertex.assign(static_cast<size_t>(maxIndex + 1), nullptr);
			for (auto* v : objectUniqueVertices) {
				if (v && v->index >= 0 && v->index <= maxIndex) {
					indexToVertex[static_cast<size_t>(v->index)] = v;
				}
			}
		}

		// Prefer triangle-based contact using boundary faces extracted from the tet mesh.
		// This works in both STL meshing and direct-loading mode and deforms with the object.
		if (agentUseSurfaceTriangles && maxIndex >= 0 && !indexToVertex.empty()) {
			struct FaceKey {
				int i0, i1, i2;
				bool operator==(const FaceKey& o) const { return i0 == o.i0 && i1 == o.i1 && i2 == o.i2; }
			};
			struct FaceKeyHash {
				size_t operator()(const FaceKey& k) const noexcept {
					const auto h0 = std::hash<int>{}(k.i0);
					const auto h1 = std::hash<int>{}(k.i1);
					const auto h2 = std::hash<int>{}(k.i2);
					size_t h = h0;
					h ^= (h1 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
					h ^= (h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
					return h;
				}
			};
			struct FaceRec {
				int count = 0;
				int a = -1, b = -1, c = -1;   // face vertex indices (ordered)
				int opp = -1;                 // opposite vertex index (interior)
			};

			auto makeKey = [](int a, int b, int c) -> FaceKey {
				if (a > b) std::swap(a, b);
				if (b > c) std::swap(b, c);
				if (a > b) std::swap(a, b);
				return FaceKey{a, b, c};
			};

			size_t tetCount = 0;
			for (int gi = 0; gi < object.groupNum; ++gi) tetCount += object.groups[gi].tetrahedra.size();
			std::unordered_map<FaceKey, FaceRec, FaceKeyHash> faces;
			faces.reserve(std::max<size_t>(16, tetCount * 4));

			auto addFace = [&](int ia, int ib, int ic, int iopp) {
				const FaceKey key = makeKey(ia, ib, ic);
				auto it = faces.find(key);
				if (it == faces.end()) {
					FaceRec rec;
					rec.count = 1;
					rec.a = ia; rec.b = ib; rec.c = ic;
					rec.opp = iopp;
					faces.emplace(key, rec);
				} else {
					it->second.count += 1;
				}
			};

			for (int gi = 0; gi < object.groupNum; ++gi) {
				Group& g = object.groups[gi];
				for (Tetrahedron* t : g.tetrahedra) {
					if (!t) continue;
					int idx[4] = {-1, -1, -1, -1};
					for (int k = 0; k < 4; ++k) idx[k] = (t->vertices[k] ? t->vertices[k]->index : -1);
					if (idx[0] < 0 || idx[1] < 0 || idx[2] < 0 || idx[3] < 0) continue;
					addFace(idx[1], idx[2], idx[3], idx[0]);
					addFace(idx[0], idx[2], idx[3], idx[1]);
					addFace(idx[0], idx[1], idx[3], idx[2]);
					addFace(idx[0], idx[1], idx[2], idx[3]);
				}
			}

			agentContactTriangles.reserve(faces.size());
			for (const auto& kv : faces) {
				const FaceRec& rec = kv.second;
				if (rec.count != 1) continue; // interior face
				if (rec.a < 0 || rec.b < 0 || rec.c < 0 || rec.opp < 0) continue;
				if (rec.a > maxIndex || rec.b > maxIndex || rec.c > maxIndex || rec.opp > maxIndex) continue;
				Vertex* va = indexToVertex[static_cast<size_t>(rec.a)];
				Vertex* vb = indexToVertex[static_cast<size_t>(rec.b)];
				Vertex* vc = indexToVertex[static_cast<size_t>(rec.c)];
				Vertex* vopp = indexToVertex[static_cast<size_t>(rec.opp)];
				if (!va || !vb || !vc || !vopp) continue;
				agentContactTriangles.push_back(AgentTriangle{va, vb, vc, vopp});
			}
		}

		if (agentUseSurfaceVertices && out.numberoftrifaces > 0 && out.trifacelist && maxIndex >= 0) {
			const int indexOffset = out.firstnumber;
			std::vector<char> isSurface(static_cast<size_t>(maxIndex + 1), 0);
			for (int i = 0; i < out.numberoftrifaces * 3; ++i) {
				const int idx = out.trifacelist[i] - indexOffset;
				if (idx >= 0 && idx <= maxIndex) isSurface[static_cast<size_t>(idx)] = 1;
			}
			for (auto* v : objectUniqueVertices) {
				if (v->index >= 0 && v->index <= maxIndex && isSurface[static_cast<size_t>(v->index)]) {
					agentContactVertices.push_back(v);
				}
			}
			if (agentContactVertices.empty()) {
				agentContactVertices = objectUniqueVertices;
			}
		} else {
			agentContactVertices = objectUniqueVertices;
		}
	}

	// [REMOVED] The previous custom export logic was causing "key not found" errors 
	// because of vertex pointer mismatches after deduplication.
	// We now use TetGen's native save functions immediately after meshing (see above).
	
	/* 
	// Export a deterministic "latest" snapshot for XPBD/PBD to consume.
	// Also export a timestamped snapshot for bookkeeping.
	// (Skip when running in --export-tetgen mode; we'll export and exit later.)
	// Only save if autoSaveMesh is enabled (controlled by parameters.txt)
	if (!exportTetgenAndExit && autoSaveMesh) {
		try {
			const std::string exportDir = exportDirOverride.empty() ? "out/tetgenfem_exports" : exportDirOverride;
			const auto latest = exportTetgenNodeEleSnapshot(object, objectUniqueVertices, exportDir, "latest");
			const auto stamped = exportTetgenNodeEleSnapshot(object, objectUniqueVertices, exportDir, "snapshot_" + nowTimestampForFilename());
			std::cout << "[TetgenFEM] Exported TetGen mesh (.node/.ele)\n"
					  << "  latest:   " << latest.nodePathAbs << " | " << latest.elePathAbs << "\n"
					  << "  snapshot: " << stamped.nodePathAbs << " | " << stamped.elePathAbs << "\n";
		}
		catch (const std::exception& e) {
			std::cerr << "[TetgenFEM] Failed to export TetGen mesh: " << e.what() << "\n";
		}
	}
	*/

	Experiment3& experiment3 = Experiment3::instance();
	experiment3.init(&object, objectUniqueVertices);
	Experiment1& experiment1 = Experiment1::instance();
	experiment1.init(&object, objectUniqueVertices);
	Experiment2& experiment2 = Experiment2::instance();
	experiment2.init(&object, objectUniqueVertices);
	Experiment4& experiment4 = Experiment4::instance();

	DragState dragState;

	// Display states
	static bool showStressCloud = false;
	static bool showExplodedView = false;
	static bool showFiberFlow = false;
	static bool showGhostLinks = false;
	static bool showVolumePreservation = false; // Volume preservation visualization mode
	static int anisoDemoState = 0; // 0: Off, 1: Isotropic Demo, 2: Anisotropic Demo
	static Vertex* anisoDemoVertex = nullptr;
	static float anisoDemoForceMag = 2700.0f; 
	static float anisoDemoRadius = 0.35f;    
	static float explodedScale = 0.5f;
	static bool whiteBackground = false;
	static bool isPaused = false; // Pause physics simulation
	static float stressGain = 4.0f; // Added for interactive tuning (reduced to 2/3 of original 15.0)
	
	// Volume preservation state
	static float planeConstraintY = 0.0f; // Y coordinate of the constraint plane
	static float initialVolume = 0.0f; // Store initial volume when mode is activated
	static std::vector<Eigen::Vector3f> initialPositions; // Store initial positions for comparison

	int frame = 1;
	SimpleUI::Context ui;

	if (exportTetgenAndExit) {
		try {
			const std::string exportDir = exportDirOverride.empty() ? "out/tetgenfem_exports" : exportDirOverride;
			const auto latest = exportTetgenNodeEleSnapshot(object, objectUniqueVertices, exportDir, "latest");
			const auto stamped = exportTetgenNodeEleSnapshot(object, objectUniqueVertices, exportDir, "snapshot_" + nowTimestampForFilename());
			std::cout << "[TetgenFEM] Exported TetGen mesh (.node/.ele)\n"
					  << "  latest:   " << latest.nodePathAbs << " | " << latest.elePathAbs << "\n"
					  << "  snapshot: " << stamped.nodePathAbs << " | " << stamped.elePathAbs << "\n";
			return 0;
		}
		catch (const std::exception& e) {
			std::cerr << "[TetgenFEM] Failed to export TetGen mesh: " << e.what() << "\n";
			return 1;
		}
	}

	while (!glfwWindowShouldClose(window)) {
		ui.beginFrame(window);
		experiment3.update();
		experiment1.update();
		experiment2.update();
		experiment4.update();

		auto beginForceRecording = [&]() {
			isRecordingForce = true;
			recordedForces.clear();
			recordedTime.clear();
			recordStartTime = glfwGetTime();
			std::cout << ">>> Started recording force data..." << std::endl;
		};

		auto stopForceRecordingAndSave = [&](const std::string& filename) {
			if (!isRecordingForce) return;
			isRecordingForce = false;
			saveForceData(filename);
			std::cout << ">>> Stopped recording. Data saved." << std::endl;
		};

		auto beginAutoTest = [&](int axis) {
			isAutoTestActive = true;
			autoTestAxis = axis;
			autoTestStartTime = glfwGetTime();
			autoTestStartPos = Eigen::Vector3f(g_selectedVertex->x, g_selectedVertex->y, g_selectedVertex->z);

			dragState.active = true;
			dragState.target = g_selectedVertex;

			beginForceRecording();

			const char* axisName = (axis == 0) ? "X-AXIS" : "Y-AXIS";
			std::cout << ">>> STARTING AUTO TEST (" << axisName << ") on Vertex " << g_selectedVertex->index << std::endl;
		};

		// ------------------ Automated Experiment State Machine
		static int experimentState = 0; // 0: Idle, 1: Start X, 2: Wait X, 3: Start Y, 4: Wait Y, 5: Done
		static int framesWait = 0;

		// ------------------ UI layout (window coordinates, origin at top-left)
		const float uiMargin = 12.0f;
		const float uiW = 200.0f;
		const float uiH = 50.0f;
		const SimpleUI::Rect uiRunRect{ uiMargin, uiMargin, uiW, uiH };

		// ------------------ Agent sphere controls
		static KeyLatch agentToggleLatch;
		static KeyLatch agentHomeLatch;
		static KeyLatch agentPrintLatch;
		static KeyLatch agentVcLatch;

		if (agentToggleLatch.consume(window, GLFW_KEY_H)) {
			agentSphere.enabled = !agentSphere.enabled;
			std::cout << "[AgentSphere] " << (agentSphere.enabled ? "ENABLED" : "DISABLED") << "\n"
			          << "  Move: I/K (Y), J/L (X), U/O (Z) | Hold SHIFT for faster\n"
			          << "  VirtualCoupling: V | Home: T | Print force: G\n"
			          << "  Contact: triangles=" << agentContactTriangles.size()
			          << " vertices=" << agentContactVertices.size() << "\n";
		}
		static bool agentUseVC = agentVirtualCoupling;
		if (agentVcLatch.consume(window, GLFW_KEY_V)) {
			agentUseVC = !agentUseVC;
			std::cout << "[AgentSphere] VirtualCoupling " << (agentUseVC ? "ON" : "OFF") << "\n";
		}
		if (agentHomeLatch.consume(window, GLFW_KEY_T)) {
			agentDevicePosition = agentHomePosition;
			agentDevicePrevPosition = agentHomePosition;
			agentDeviceVelocity.setZero();
			agentProxyPosition = agentHomePosition;
			agentProxyVelocity.setZero();
		}

		// Continuous movement while key is held.
		{
			const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
			                   glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
			const float speed = agentMoveSpeedBboxPerSec * bboxDiag * (shift ? 4.0f : 1.0f);
			Eigen::Vector3f delta = Eigen::Vector3f::Zero();
			if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) delta.x() -= speed * timeStep;
			if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) delta.x() += speed * timeStep;
			if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) delta.y() += speed * timeStep;
			if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) delta.y() -= speed * timeStep;
			if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) delta.z() -= speed * timeStep;
			if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) delta.z() += speed * timeStep;
			agentDevicePosition += delta;
			agentDeviceVelocity = (agentDevicePosition - agentDevicePrevPosition) / std::max(1e-8f, timeStep);
			agentDevicePrevPosition = agentDevicePosition;
		}
		if (agentPrintLatch.consume(window, GLFW_KEY_G)) {
			float proxySurfDist = -1.0f;
			float proxySurfSN = 0.0f;
			float proxyPen = 0.0f;
			if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
				const AgentSurfaceQueryResult q = queryAgentSurface(agentProxyPosition, agentContactTriangles);
				if (q.found && q.outwardNormal.squaredNorm() > 0.0f) {
					proxySurfDist = q.distanceToSurface;
					proxySurfSN = q.outwardNormal.dot(agentProxyPosition - q.closestPoint);
					proxyPen = std::max(0.0f, agentSphere.radius - q.distanceToSurface);
				}
			}
			std::cout << "[AgentSphere] vc=" << (agentUseVC ? "1" : "0")
			          << " devicePos=(" << agentDevicePosition.x() << ", " << agentDevicePosition.y() << ", " << agentDevicePosition.z() << ")"
			          << " proxyPos=(" << agentProxyPosition.x() << ", " << agentProxyPosition.y() << ", " << agentProxyPosition.z() << ")"
			          << " deviceForceN=(" << agentLastDeviceForceN.x() << ", " << agentLastDeviceForceN.y() << ", " << agentLastDeviceForceN.z() << ")"
			          << " contactForceN=(" << agentLastContactForceN.x() << ", " << agentLastContactForceN.y() << ", " << agentLastContactForceN.z() << ")"
			          << " contacts=" << agentLastContactCount
			          << " proxySurfDist=" << proxySurfDist
			          << " proxySurfSN=" << proxySurfSN
			          << " proxyPen=" << proxyPen
			          << "\n";
		}

		// UI button triggers deterministic Experiment 3 (one-click).

		// ------------------ Manual right-click drag force (restores RMB "apply force")
		// Holding RMB drags the nearest vertex under the cursor and applies a spring-like force.
		static bool prevRightDown = false;
		const bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
		const bool rightPressed = rightDown && !prevRightDown;
		const bool rightReleased = !rightDown && prevRightDown;
		prevRightDown = rightDown;

		auto pointInRect = [](double x, double y, const SimpleUI::Rect& r) {
			return x >= r.x && x <= (r.x + r.w) && y >= r.y && y <= (r.y + r.h);
		};
		const bool cursorInUiButton = pointInRect(ui.state().mouseXWindow, ui.state().mouseYWindow, uiRunRect);

		// Press 'E' to export current (possibly deformed) tet mesh to .node/.ele for XPBD/PBD.
		static KeyLatch exportLatch;
		if (exportLatch.consume(window, GLFW_KEY_E)) {
			try {
				const std::string exportDir = exportDirOverride.empty() ? "out/tetgenfem_exports" : exportDirOverride;
				const auto paths = exportTetgenNodeEleSnapshot(object, objectUniqueVertices, exportDir, "latest");
				std::cout << "[TetgenFEM] Exported current TetGen mesh (E)\n"
						  << "  " << paths.nodePathAbs << "\n"
						  << "  " << paths.elePathAbs << "\n";
			}
			catch (const std::exception& e) {
				std::cerr << "[TetgenFEM] Export failed: " << e.what() << "\n";
			}
		}

		if (!isAutoTestActive && !experiment3.isActive() && !experiment1.isActive() && !experiment2.isActive() && !experiment4.isActive()) {
			if (rightReleased) {
				dragState.active = false;
				dragState.target = nullptr;
			}

			if (rightPressed && !cursorInUiButton) {
				Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
				model.block<3, 3>(0, 0) = rotation.toRotationMatrix();
				const Eigen::Matrix4f projection = buildProjectionMatrix();

				Vertex* picked = pickVertexAtCursor(
					objectUniqueVertices,
					ui.state().mouseXWindow,
					ui.state().mouseYWindow,
					model,
					projection,
					ui.state().windowWidth,
					ui.state().windowHeight);

				if (picked) {
					g_selectedVertex = picked;
					dragState.active = true;
					dragState.target = picked;
					dragState.lastX = ui.state().mouseXWindow;
					dragState.lastY = ui.state().mouseYWindow;

					const Eigen::Vector4f clip = projection * model *
						Eigen::Vector4f(picked->x, picked->y, picked->z, 1.0f);
					dragState.grabbedNdcZ = (std::abs(clip.w()) > 1e-8f) ? (clip.z() / clip.w()) : 0.0f;

					const Eigen::Matrix4f invProjectionModel = (projection * model).inverse();
					const Eigen::Vector3f cursorWorld = unprojectCursorToWorld(
						ui.state().mouseXFramebuffer,
						ui.state().mouseYFramebuffer,
						dragState.grabbedNdcZ,
						invProjectionModel,
						ui.state().framebufferWidth,
						ui.state().framebufferHeight);
					const Eigen::Vector3f targetPos(picked->x, picked->y, picked->z);
					dragState.grabOffset = targetPos - cursorWorld;
				}
			}
		}

		// Handle auto-start Experiment 3
		static bool exp3AutoStarted = false;
		static bool exp3AutoFinished = false;
		if (!exp3AutoStarted) {
			const char* env = std::getenv("TETGENFEM_AUTO_EXP3");
			if (env) {
				std::cout << "[TetgenFEM] Auto-starting Experiment 3...\n";
				experiment3.requestStart();
				exp3AutoStarted = true;
			}
		} else if (!exp3AutoFinished) {
			if (!experiment3.isActive()) {
				std::cout << "[TetgenFEM] Auto-Experiment 3 Finished. Exiting.\n";
				exp3AutoFinished = true;
				exit(0);
			}
		}

		// State Machine
		if (experimentState == 1) {
			beginAutoTest(0); // Start X Axis Test
			experimentState = 2;
		}
		else if (experimentState == 2) {
			if (!isAutoTestActive) { // Wait for finish
				framesWait++;
				if (framesWait > 30) { // Wait a bit between tests
					experimentState = 3;
					framesWait = 0;
				}
			}
		}
		else if (experimentState == 3) {
			beginAutoTest(1); // Start Y Axis Test
			experimentState = 4;
		}
		else if (experimentState == 4) {
			if (!isAutoTestActive) { // Wait for finish
				std::cout << ">>> ALL EXPERIMENTS COMPLETED. EXITING..." << std::endl;
				experimentState = 5;
				glfwSetWindowShouldClose(window, true);
			}
		}

		// ------------------ Interaction Logic (Optimized)
		static std::vector<Eigen::Vector3f> dragForces;
		if (dragForces.empty()) {
			int maxV = 0;
			for (auto* v : objectUniqueVertices) if (v->index > maxV) maxV = v->index;
			dragForces.resize(maxV + 1, Eigen::Vector3f::Zero());
		}
		
		// Reset forces efficiently
		#pragma omp parallel for
		for (int i = 0; i < (int)dragForces.size(); ++i) {
			dragForces[i] = Eigen::Vector3f::Zero();
		}

		// --- NEW: Anisotropy Demo Auto-Force (Region-based with Pulse & Diagonal) ---
		if (anisoDemoState > 0 && anisoDemoVertex != nullptr) {
			float rSq = anisoDemoRadius * anisoDemoRadius;
			Eigen::Vector3f centerPos(anisoDemoVertex->x, anisoDemoVertex->y, anisoDemoVertex->z);
			
			float pulse = 0.5f + 0.5f * std::sin(glfwGetTime() * 4.0f); 
			
			// Anisotropic mode uses 2x force for more visible deformation
			float forceMultiplier = (anisoDemoState == 2) ? 2.0f : 1.0f;
			float currentForce = anisoDemoForceMag * forceMultiplier;
			
			#pragma omp parallel for
			for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
				Group& group = object.getGroup(groupIdx);
				for (auto& pair : group.verticesMap) {
					Vertex* v = pair.second;
					Eigen::Vector3f vPos(v->x, v->y, v->z);
					float distSq = (vPos - centerPos).squaredNorm();
					if (distSq < rSq) {
						float weight = 1.0f - std::sqrt(distSq) / anisoDemoRadius;
						// Pull Diagonally (X and Y) to show directional bias
						#pragma omp atomic
						dragForces[v->index].x() += currentForce * weight * pulse;
						#pragma omp atomic
						dragForces[v->index].y() += currentForce * weight * pulse;
					}
				}
			}
		}

		// Experiment 1: deterministic constant load (independent of dragging).
		if (experiment1.isActive()) {
			experiment1.appendVertexForces(dragForces);
		}
		// Experiment 2: deterministic uniaxial stretch (independent of dragging).
		if (experiment2.isActive()) {
			experiment2.appendVertexForces(dragForces);
		}

		// Let Experiment 3 drive the drag target deterministically when active.
		if (experiment3.isActive() && !experiment3.wantsDrag()) {
			dragState.active = false;
			dragState.target = nullptr;
		}
		if (experiment3.wantsDrag()) {
			dragState.active = true;
			dragState.target = experiment3.targetVertex();
			if (dragState.target) {
				g_selectedVertex = dragState.target;
			}
		}
		// Experiment2 does not use the mouse-like drag pipeline, so always disable it while active.
		if (experiment2.isActive()) {
			dragState.active = false;
			dragState.target = nullptr;
		}
		if (experiment4.isActive()) {
			dragState.active = false;
			dragState.target = nullptr;
		}

		// Handle dragging physics (manual / auto-test / experiment3)
		if (dragState.active && dragState.target != nullptr) {
			Eigen::Vector3f desiredTargetPos = Eigen::Vector3f(dragState.target->x, dragState.target->y, dragState.target->z);
			bool processPhysics = true;

			if (experiment3.wantsDrag()) {
				desiredTargetPos = experiment3.desiredTargetPosition();
			}
			else if (isAutoTestActive) {
				double elapsed = glfwGetTime() - autoTestStartTime;
				if (elapsed > autoTestDuration) {
					// End Auto Test
					isAutoTestActive = false;
					isRecordingForce = false;
					dragState.active = false;
					processPhysics = false;
					std::string fname = (autoTestAxis == 0) ? "force_data_x.txt" : "force_data_y.txt";
					saveForceData(fname);
					std::cout << ">>> Auto Test Finished. Saved to " << fname << std::endl;
				}
				else {
					// Interpolate
					float t = static_cast<float>(elapsed / autoTestDuration);
					desiredTargetPos = autoTestStartPos;
					if (autoTestAxis == 0) desiredTargetPos.x() += autoTestDistance * t; // X Axis
					else desiredTargetPos.y() += autoTestDistance * t; // Y Axis
				}
			}
			else {
				const bool manualRightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
				if (!manualRightDown) {
					processPhysics = false;
					dragState.active = false;
					dragState.target = nullptr;
				}
				else {
					Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
					model.block<3, 3>(0, 0) = rotation.toRotationMatrix();
					const Eigen::Matrix4f projection = buildProjectionMatrix();
					const Eigen::Matrix4f invProjectionModel = (projection * model).inverse();
					desiredTargetPos = unprojectCursorToWorld(
						ui.state().mouseXFramebuffer,
						ui.state().mouseYFramebuffer,
						dragState.grabbedNdcZ,
						invProjectionModel,
						ui.state().framebufferWidth,
						ui.state().framebufferHeight) + dragState.grabOffset;
				}
			}

			if (processPhysics && dragState.active && !isPaused) {
				const float influenceRadius = dragInfluenceRadius;
				const float stiffness = dragStiffness;
				const float maxAccel = dragMaxAccel;
				Eigen::Vector3f targetPos(dragState.target->x, dragState.target->y, dragState.target->z);
				Eigen::Vector3f displacement = desiredTargetPos - targetPos;
				float displacementNorm = displacement.norm();
				if (displacementNorm > 1e-6f) {
					float maxDisp = dragMaxDisplacement;
					if (displacementNorm > maxDisp) {
						displacement *= (maxDisp / displacementNorm);
						displacementNorm = maxDisp;
					}
				}

				float currentFrameTotalForce = 0.0f;
				const std::vector<Vertex*>& verticesForForces =
					(experiment3.wantsDrag() ? experiment3.forceVertices() : objectUniqueVertices);

				#pragma omp parallel for reduction(+:currentFrameTotalForce)
				for (int i = 0; i < (int)verticesForForces.size(); ++i) {
					Vertex* vertex = verticesForForces[i];
					Eigen::Vector3f currentPos(vertex->x, vertex->y, vertex->z);
					float dist = (currentPos - targetPos).norm();
					if (dist <= influenceRadius) {
						float falloff = std::max(0.05f, 1.0f - dist / influenceRadius);
						if (vertex == dragState.target) {
							falloff *= 1.5f;
						}
						Eigen::Vector3f accel = displacement * (stiffness * falloff);
						float accelNorm = accel.norm();
						if (accelNorm > maxAccel) {
							accel *= (maxAccel / accelNorm);
						}
						dragForces[vertex->index] = accel;
						currentFrameTotalForce += accel.norm();
					}
				}

				float targetForce = 0.0f;
				if (dragState.target->index < (int)dragForces.size()) {
					targetForce = dragForces[dragState.target->index].norm();
				}
				if (experiment3.wantsDrag()) {
					experiment3.onDragForces(currentFrameTotalForce, targetForce);
				}

				if (isRecordingForce) {
					recordedForces.push_back(currentFrameTotalForce);
					recordedTime.push_back(glfwGetTime() - recordStartTime);
				}
			}
		}

		// Apply agent contact after any drag/experiment force contributions.
		if (agentSphere.enabled) {
			if (agentUseVC) {
			// Virtual coupling: device controls a target (agentDevicePosition), proxy interacts with tissue.
			// We use substeps to avoid "tunneling" and to get accurate proxy dynamics,
			// but we DON'T accumulate forces to the object in the substep loop.
			// Instead, we only apply contact force ONCE at the end based on final proxy state.
			const bool useTriangleContact = agentUseSurfaceTriangles && !agentContactTriangles.empty();
			const float deviceStep = agentDeviceVelocity.norm() * timeStep;
			const float maxStep = std::max(1e-6f, 0.25f * agentSphere.radius);
			const int substepsFromSpeed = static_cast<int>(std::ceil(deviceStep / maxStep));
			const int substeps = std::max(1, std::max(agentVcSubsteps, substepsFromSpeed));
			const float dtSub = timeStep / static_cast<float>(substeps);

			AgentContactResult lastContact{};
			Eigen::Vector3f lastCouplingForceN = Eigen::Vector3f::Zero();

			const float maxPenFrac = std::clamp(agentMaxPenetrationFrac, 0.0f, 0.95f);
			const float allowedPen = maxPenFrac * agentSphere.radius;
			const float projMargin = 1e-4f * bboxDiag;

			// Substep loop for proxy dynamics - DO NOT apply forces to object here
			for (int si = 0; si < substeps; ++si) {
				const Eigen::Vector3f couplingForceN =
					agentVcKLen * (agentDevicePosition - agentProxyPosition) +
					agentVcCLen * (agentDeviceVelocity - agentProxyVelocity);

				agentSphere.position = agentProxyPosition;
				agentSphere.velocity = agentProxyVelocity;

				// Use a temporary force buffer - don't touch dragForces in the substep loop!
				std::vector<Eigen::Vector3f> tempForces(dragForces.size(), Eigen::Vector3f::Zero());
				const AgentContactResult contact = useTriangleContact
					? applyAgentSphereTriangleContact(agentSphere, agentContactTriangles, dtSub, tempForces, 1.0f)
					: applyAgentSphereContact(agentSphere, agentContactVertices, dtSub, tempForces, 1.0f);

				// Keep penetration bounded (prevents "falling through" when stiffness/step is insufficient).
				// Note: contact.avgNormal points from proxy center to surface point (inward when outside),
				// so subtracting it moves the proxy outward.
				if (contact.contactVertexCount > 0 && contact.maxPenetration > 0.0f && contact.avgNormal.squaredNorm() > 0.0f) {
					const float overPen = std::max(0.0f, contact.maxPenetration - allowedPen);
					const float corr = std::clamp(agentProxyCorr, 0.0f, 1.0f);
					agentProxyPosition -= contact.avgNormal * (overPen * std::max(0.25f, corr) + projMargin);

					// Remove inward velocity component so we don't immediately re-penetrate.
					const float vnIn = agentProxyVelocity.dot(contact.avgNormal);
					if (vnIn > 0.0f) agentProxyVelocity -= contact.avgNormal * vnIn;
				}

				const Eigen::Vector3f proxyAcc = (couplingForceN + contact.reactionForceN) / agentProxyMassKg;
				agentProxyVelocity += proxyAcc * dtSub;
				agentProxyPosition += agentProxyVelocity * dtSub;

				// Recovery / constraint: keep proxy from ending up fully inside the closed surface.
				// If contact reports 0, it might still be inside (sphere completely contained => no boundary intersection),
				// so we clamp the proxy to a "shell" near the surface by projecting along the closest outward normal.
				if (useTriangleContact && contact.contactVertexCount == 0) {
					const AgentSurfaceQueryResult q = queryAgentSurface(agentProxyPosition, agentContactTriangles);
					if (q.found && q.outwardNormal.squaredNorm() > 0.0f) {
						const float sN = q.outwardNormal.dot(agentProxyPosition - q.closestPoint);
						const float targetSN = std::max(0.0f, agentSphere.radius - allowedPen);
						if (sN < targetSN) {
							agentProxyPosition += q.outwardNormal * ((targetSN - sN) + projMargin);
							const float vn = agentProxyVelocity.dot(q.outwardNormal);
							if (vn < 0.0f) agentProxyVelocity -= q.outwardNormal * vn;
						}
					}
				}

				lastCouplingForceN = couplingForceN;
				lastContact = contact;
			}

			// NOW apply contact force to object ONCE based on final proxy state
			// Use DISTRIBUTED force application like drag - affects all vertices within influence radius
			agentSphere.position = agentProxyPosition;
			agentSphere.velocity = agentProxyVelocity;
			
				// Query the closest surface point to determine penetration.
				// Note: Using only penetration here makes "hard press" feel weak in Virtual Coupling mode because
				// the proxy is intentionally kept near/outside the surface (penetration ~ 0). We therefore also
				// transfer the device<->proxy coupling force (normal component) to the tissue as a distributed load.
				const AgentSurfaceQueryResult surfQ = queryAgentSurface(agentProxyPosition, agentContactTriangles);
			
				AgentContactResult finalContact{};
				if (surfQ.found) {
					const Eigen::Vector3f contactPoint = surfQ.closestPoint;

					// Use a stable contact direction: from agent center to the closest surface point.
					// This is less jittery than per-vertex radial directions (especially on side contact).
					Eigen::Vector3f contactDir = (contactPoint - agentProxyPosition);
					const float contactDirLen = contactDir.norm();
					if (contactDirLen > 1e-6f) {
						contactDir /= contactDirLen;
					} else if (surfQ.outwardNormal.squaredNorm() > 0.0f) {
						// outwardNormal points out of the surface; contactDir should point toward the surface.
						contactDir = -surfQ.outwardNormal;
					} else {
						contactDir = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
					}

					// Penetration-based penalty term (kept for robustness).
					const float penetration = agentSphere.radius - surfQ.distanceToSurface;

						const float influenceRadius = std::max(1e-6f, agentSphere.radius * std::max(0.0f, agentInfluenceRadiusFrac));
						const float k = agentSphere.contactStiffness;
						const float c = agentSphere.contactDamping;

						// Transfer the (compressive) coupling force to tissue only when the proxy is near the surface.
						// Otherwise the closest-point query would cause "remote pressing" even when the proxy is far away.
						//
						// We also ramp this in over a small distance range so Virtual Coupling can still "press"
						// even when the proxy is clamped to stay slightly outside the surface.
						const float couplingMaxN = std::max(0.0f, lastCouplingForceN.dot(contactDir)); // Newton
						const float couplingRange = std::max(1e-6f, std::max(allowedPen, 0.25f * agentSphere.radius));
						float couplingWeight = 0.0f;
						if (surfQ.distanceToSurface <= agentSphere.radius + couplingRange) {
							couplingWeight = std::clamp(
								(agentSphere.radius + couplingRange - surfQ.distanceToSurface) / couplingRange, 0.0f, 1.0f);
						}
						const float couplingNormalForceN = couplingMaxN * couplingWeight;

					if (couplingNormalForceN > 0.0f || penetration > 0.0f) {
						// Compute a mass-weighted normalization so that sum(force) ~= couplingNormalForceN.
						float invWeightedMassSum = 0.0f;
						if (couplingNormalForceN > 0.0f) {
							float weightedMassSum = 0.0f;
							#pragma omp parallel for reduction(+:weightedMassSum)
							for (int i = 0; i < static_cast<int>(agentContactVertices.size()); ++i) {
								Vertex* v = agentContactVertices[i];
								if (!v || v->isFixed) continue;
								if (v->index < 0 || v->index >= static_cast<int>(dragForces.size())) continue;
								const Eigen::Vector3f vpos(v->x, v->y, v->z);
								const float distToContact = (vpos - contactPoint).norm();
								if (distToContact > influenceRadius) continue;
								const float t = distToContact / std::max(1e-6f, influenceRadius);
								const float oneMinusT = (1.0f - t);
								const float falloff2 = oneMinusT * oneMinusT;
								const float falloff = falloff2 * falloff2;
								weightedMassSum += std::max(0.0f, v->vertexMass) * falloff;
							}
							invWeightedMassSum = (weightedMassSum > 1e-12f) ? (1.0f / weightedMassSum) : 0.0f;
						}

						int contactCount = 0;
						Eigen::Vector3f totalReactionForce = Eigen::Vector3f::Zero();

						#pragma omp parallel for reduction(+:contactCount)
						for (int i = 0; i < static_cast<int>(agentContactVertices.size()); ++i) {
							Vertex* v = agentContactVertices[i];
							if (!v || v->isFixed) continue;
							if (v->index < 0 || v->index >= static_cast<int>(dragForces.size())) continue;

							const Eigen::Vector3f vpos(v->x, v->y, v->z);
							const float distToContact = (vpos - contactPoint).norm();
							if (distToContact > influenceRadius) continue;

							const float t = distToContact / std::max(1e-6f, influenceRadius);
							const float oneMinusT = (1.0f - t);
							const float falloff2 = oneMinusT * oneMinusT;
							const float falloff = falloff2 * falloff2;

							// Damping based on relative velocity along the contact direction (unilateral).
							const Eigen::Vector3f vtxVel(v->velx, v->vely, v->velz);
							const float relVn = (vtxVel - agentProxyVelocity).dot(contactDir);
							const float dampAccel = -c * std::min(0.0f, relVn);

							// Coupling force -> acceleration (distributed by mass and falloff).
							const float couplingAccel = couplingNormalForceN * invWeightedMassSum * falloff;

							// Penalty acceleration (only active when penetrating).
							const float penAccel = (penetration > 0.0f ? (k * penetration) : 0.0f) * falloff;

							const float accelMag = couplingAccel + penAccel + dampAccel * falloff;
							if (accelMag <= 0.0f) continue;

							const Eigen::Vector3f accel = contactDir * accelMag;

							#pragma omp atomic
							dragForces[v->index].x() += accel.x();
							#pragma omp atomic
							dragForces[v->index].y() += accel.y();
							#pragma omp atomic
							dragForces[v->index].z() += accel.z();

							++contactCount;

							const Eigen::Vector3f fOnVertex = accel * v->vertexMass;
							#pragma omp atomic
							totalReactionForce.x() -= fOnVertex.x();
							#pragma omp atomic
							totalReactionForce.y() -= fOnVertex.y();
							#pragma omp atomic
							totalReactionForce.z() -= fOnVertex.z();
						}

						finalContact.contactVertexCount = contactCount;
						finalContact.maxPenetration = std::max(0.0f, penetration);
						finalContact.avgNormal = contactDir; // from proxy center to closest surface point
						finalContact.reactionForceN = totalReactionForce;
					}
				}

			agentLastCouplingForceN = lastCouplingForceN;
			agentLastContactForceN = finalContact.reactionForceN;
			agentLastDeviceForceN = -lastCouplingForceN;
			agentLastContactCount = finalContact.contactVertexCount;
			} else {
			// Direct: the sphere is kinematic and contact force is reported directly.
			// Use DISTRIBUTED force application like VC mode
			agentSphere.position = agentDevicePosition;
			agentSphere.velocity = agentDeviceVelocity;
			const bool useTriangleContact = agentUseSurfaceTriangles && !agentContactTriangles.empty();
			
			AgentContactResult contact{};
			if (useTriangleContact) {
				const AgentSurfaceQueryResult surfQ = queryAgentSurface(agentDevicePosition, agentContactTriangles);
				if (surfQ.found) {
					const float penetration = agentSphere.radius - surfQ.distanceToSurface;
					if (penetration > 0.0f) {
						const Eigen::Vector3f contactPoint = surfQ.closestPoint;
						const Eigen::Vector3f contactNormal = surfQ.outwardNormal;
						const float influenceRadius = agentSphere.radius * agentInfluenceRadiusFrac;
						const float k = agentSphere.contactStiffness;
						const float c = agentSphere.contactDamping;
						float baseAccelMag = k * penetration;
						
						int contactCount = 0;
						Eigen::Vector3f totalReactionForce = Eigen::Vector3f::Zero();
						
						#pragma omp parallel for reduction(+:contactCount)
						for (int i = 0; i < static_cast<int>(agentContactVertices.size()); ++i) {
							Vertex* v = agentContactVertices[i];
							if (!v || v->isFixed) continue;
							if (v->index < 0 || v->index >= static_cast<int>(dragForces.size())) continue;
							
							const Eigen::Vector3f vpos(v->x, v->y, v->z);
							const float distToContact = (vpos - contactPoint).norm();
							
							if (distToContact <= influenceRadius) {
								const float t = distToContact / influenceRadius;
								const float falloff = (1.0f - t) * (1.0f - t);
								
								Eigen::Vector3f pushDir = (vpos - agentDevicePosition);
								const float pushDist = pushDir.norm();
								if (pushDist > 1e-6f) pushDir /= pushDist;
								else pushDir = contactNormal;
								
								const Eigen::Vector3f vtxVel(v->velx, v->vely, v->velz);
								const float relVn = (vtxVel - agentDeviceVelocity).dot(pushDir);
								const float dampMag = -c * std::min(0.0f, relVn);
								
								const float accelMag = (baseAccelMag + dampMag) * falloff;
								const Eigen::Vector3f accel = pushDir * accelMag;
								
								#pragma omp atomic
								dragForces[v->index].x() += accel.x();
								#pragma omp atomic
								dragForces[v->index].y() += accel.y();
								#pragma omp atomic
								dragForces[v->index].z() += accel.z();
								
								++contactCount;
								
								const Eigen::Vector3f fOnVertex = accel * v->vertexMass;
								#pragma omp atomic
								totalReactionForce.x() -= fOnVertex.x();
								#pragma omp atomic
								totalReactionForce.y() -= fOnVertex.y();
								#pragma omp atomic
								totalReactionForce.z() -= fOnVertex.z();
							}
						}
						contact.contactVertexCount = contactCount;
						contact.maxPenetration = penetration;
						contact.avgNormal = contactNormal;
						contact.reactionForceN = totalReactionForce;
					}
				}
			}
			
			agentLastCouplingForceN.setZero();
			agentLastContactForceN = contact.reactionForceN;
			agentLastDeviceForceN = contact.reactionForceN;
			agentLastContactCount = contact.contactVertexCount;
			agentProxyPosition = agentDevicePosition;
			agentProxyVelocity = agentDeviceVelocity;
		}

			if (agentWriteLiveFile) {
				static int liveFrame = 0;
				++liveFrame;
				if (liveFrame % std::max(1, agentLiveFileIntervalFrames) == 0) {
					try {
						std::filesystem::create_directories("out");
						std::ofstream f("out/agent_force_live.txt", std::ios::out | std::ios::trunc);
						if (f.is_open()) {
							f << "time " << glfwGetTime() << "\n";
							f << "vc " << (agentUseVC ? 1 : 0) << "\n";
							f << "devicePos " << agentDevicePosition.x() << " " << agentDevicePosition.y() << " " << agentDevicePosition.z() << "\n";
							f << "proxyPos " << agentProxyPosition.x() << " " << agentProxyPosition.y() << " " << agentProxyPosition.z() << "\n";
							f << "deviceForceN " << agentLastDeviceForceN.x() << " " << agentLastDeviceForceN.y() << " " << agentLastDeviceForceN.z() << "\n";
							f << "contactForceN " << agentLastContactForceN.x() << " " << agentLastContactForceN.y() << " " << agentLastContactForceN.z() << "\n";
							f << "contacts " << agentLastContactCount << "\n";
						}
					} catch (...) {
						// ignore live file errors
					}
				}
			}
		} else {
			agentLastDeviceForceN.setZero();
			agentLastContactForceN.setZero();
			agentLastCouplingForceN.setZero();
			agentLastContactCount = 0;
		}

		Eigen::Vector3f inputForce = Eigen::Vector3f::Zero(); // Placeholder for removed manual input


		static bool drawFaces = true;
		static bool drawEdges = true;
		
		// Physics update only when not paused
		if (!isPaused) {
#pragma omp parallel for
			for (int i = 0; i < groupNum; i++) {
				//object.groups[i].calGroupKFEM(youngs, poisson);
				object.groups[i].calPrimeVec(inputForce, dragForces);
				//object.groups[i].calPrimeVecS(topVertexLocalIndices, bottomVertexLocalIndices);
				//object.groups[i].calPrimeVec2(wKey);
				//object.groups[i].calPrimeVec(wKey);
				//object.groups[i].calPrimeVecT(wKey);
				/*object.groups[i].calLHSFEM();
				object.groups[i].calRHSFEM();
				object.groups[i].calDeltaXFEM();
				object.groups[i].calculateCurrentPositionsFEM();
				object.groups[i].updateVelocityFEM();
				object.groups[i].updatePositionFEM();*/

				object.groups[i].calRotationMatrix(frame);

			}
			/*for (int i = 0; i < groupNum; i++) {
				std::cout << "Group" << i << "Prime vector is" << std::endl << object.groups[i].primeVec;
			}*/


			static int defaultPbdIterations = 10;
			int pbdIterations = defaultPbdIterations;
			if (experiment1.isActive()) {
				pbdIterations = experiment1.pbdIterationsThisFrame(defaultPbdIterations);
			} else if (experiment2.isActive()) {
				pbdIterations = experiment2.pbdIterationsThisFrame(defaultPbdIterations);
			}
			object.PBDLOOP(pbdIterations);
			experiment2.onAfterPhysics();
			
			// Apply plane constraint for volume preservation visualization
			if (showVolumePreservation) {
				// Find vertices that penetrate the plane and project them back
				for (int i = 0; i < object.groupNum; ++i) {
					Group& group = object.groups[i];
					for (const auto& vertexPair : group.verticesMap) {
						Vertex* vertex = vertexPair.second;
						if (!vertex->isFixed && vertex->y > planeConstraintY) {
							// Project vertex to the plane
							vertex->y = planeConstraintY;
							// Also zero out velocity in Y direction to prevent bouncing
							vertex->vely = 0.0f;
						}
					}
				}
			}
		}

		// Update COM for all groups to ensure correct stress cloud visualization
		if (showStressCloud) {
			for (int i = 0; i < object.groupNum; ++i) {
				object.groups[i].calCenterofMass();
			}
		}
		
		// Calculate current volume for volume preservation visualization
		static float currentVolume = 0.0f;
		if (showVolumePreservation) {
			currentVolume = 0.0f;
			for (int i = 0; i < object.groupNum; ++i) {
				Group& group = object.groups[i];
				for (Tetrahedron* tet : group.tetrahedra) {
					currentVolume += tet->calVolumeTetra();
				}
			}
		}

		static KeyLatch cLatch;
		// Replaced SAVE button with Benchmark, so allow saving via Key C only or re-map
		if (cLatch.consume(window, GLFW_KEY_C)) {
			std::ofstream file("vbdcomp_our.txt", std::ios::out | std::ios::trunc);
			if (!file.is_open()) {
				std::cerr << "Failed to open file." << std::endl;
				return 0;
			}
			for (int i = 0; i < objectUniqueVertices.size(); i++) {
				file << i + 1 << " " << objectUniqueVertices[i]->x << " " << objectUniqueVertices[i]->y << " " << objectUniqueVertices[i]->z << std::endl;
			}
			file.close();
			std::cout << "Data has been written to the file." << std::endl;
		}

		


		// Handle stress gain tuning via keyboard
		if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) stressGain *= 1.02f; // '+' key
		if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) stressGain *= 0.98f; // '-' key

		// Handle pause/resume via keyboard (P key)
		static KeyLatch pauseLatch;
		if (pauseLatch.consume(window, GLFW_KEY_P)) {
			isPaused = !isPaused;
		}

		// Render here
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		static Eigen::Vector3f globalInitCOM = Eigen::Vector3f::Zero();
		static bool comCalculated = false;
		if (!comCalculated && object.groupNum > 0) {
			float totalMass = 0.0f;
			for (int i = 0; i < object.groupNum; ++i) {
				globalInitCOM += object.getGroup(i).initCOM * object.getGroup(i).groupMass;
				totalMass += object.getGroup(i).groupMass;
			}
			if (totalMass > 0) globalInitCOM /= totalMass;
			comCalculated = true;
		}
		//drawAxis1(0.3f, object.groups[0].rotate_matrix);
		
		drawAxis(0.3f);
		//std::cout << getRotationAngleZ(object.groups[0].rotate_matrix) << std::endl;;
		// Enable wireframe mode (unless in volume preservation mode which needs filled faces)
		if (!showVolumePreservation) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		}
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		mat = Eigen::Matrix4f::Identity();
		mat.block<3, 3>(0, 0) = rotation.toRotationMatrix();
		glMultMatrixf(mat.data());

		// Draw agent sphere ("finger") device/proxy.
		if (agentSphere.enabled) {
			glLineWidth(2.0f);
			// Device target (gray)
			if (whiteBackground) glColor3f(0.2f, 0.2f, 0.2f);
			else glColor3f(0.7f, 0.7f, 0.7f);
			drawWireSphereCircles(agentDevicePosition, agentSphere.radius, 36);

			// Proxy (orange/red)
			if (whiteBackground) glColor3f(0.8f, 0.2f, 0.2f);
			else glColor3f(1.0f, 0.4f, 0.2f);
			drawWireSphereCircles(agentProxyPosition, agentSphere.radius, 36);
		}

		// Draw constraint plane for volume preservation mode
		if (showVolumePreservation && planeConstraintY > 0.0f) {
			glDisable(GL_DEPTH_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glBegin(GL_QUADS);
			// Draw a semi-transparent plane
			if (whiteBackground) {
				glColor4f(0.3f, 0.3f, 0.3f, 0.5f);
			} else {
				glColor4f(0.8f, 0.8f, 0.8f, 0.5f);
			}
			// Draw a large plane (extend beyond liver bounds)
			float planeSize = 2.0f;
			glVertex3f(-planeSize, planeConstraintY, -planeSize);
			glVertex3f(planeSize, planeConstraintY, -planeSize);
			glVertex3f(planeSize, planeConstraintY, planeSize);
			glVertex3f(-planeSize, planeConstraintY, planeSize);
			glEnd();
			glEnable(GL_DEPTH_TEST);
		}
		
		// Draw vertices (skip in stress/volume preservation modes for cleaner visualization)
		if (!showStressCloud && !showVolumePreservation) {
			glPointSize(5.0f);

			if (whiteBackground) {
				glColor3f(0.1f, 0.1f, 0.1f);
			} else {
				glColor3f(1.0f, 1.0f, 1.0f);
			}
			glBegin(GL_POINTS);
			for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
				Group& group = object.getGroup(groupIdx);
				Eigen::Vector3f offset = Eigen::Vector3f::Zero();
				if (showExplodedView) {
					offset = (group.initCOM - globalInitCOM) * explodedScale;
				}
				std::vector<Vertex*> uniqueVertices = group.getUniqueVertices();
				for (Vertex* vertex : uniqueVertices) {
					glVertex3f(vertex->x + offset.x(), vertex->y + offset.y(), vertex->z + offset.z());
				}
			}
			glEnd();
		}

		// (Removed old debug text rendering code.)
		if (drawFaces) {
			// Pre-calculate smooth vertex stress if needed
			if (showVolumePreservation) {
				// Use filled mode for volume preservation visualization
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				// Enable blending for semi-transparent initial outline
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			} else if (showStressCloud) {
				// 1. Reset vertex accumulators (Parallelized)
				#pragma omp parallel for
				for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
					Group& group = object.getGroup(groupIdx);
					for (auto& pair : group.verticesMap) {
						pair.second->lastStress = 0.0f;
						pair.second->connectedTets = 0;
					}
				}

				// 2. Accumulate stress from tetrahedra (Parallelized)
				#pragma omp parallel for
				for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
					Group& group = object.getGroup(groupIdx);
					for (Tetrahedron* tet : group.tetrahedra) {
						Vertex* v0 = tet->vertices[0];
						Vertex* v1 = tet->vertices[1];
						Vertex* v2 = tet->vertices[2];
						Vertex* v3 = tet->vertices[3];

						Eigen::Matrix3f Ds;
						Ds << v1->x - v0->x, v2->x - v0->x, v3->x - v0->x,
							  v1->y - v0->y, v2->y - v0->y, v3->y - v0->y,
							  v1->z - v0->z, v2->z - v0->z, v3->z - v0->z;
						
						Eigen::Matrix3f F = Ds * tet->invDm;
						Eigen::Matrix3f E = 0.5f * (F.transpose() * F - Eigen::Matrix3f::Identity());
						float currentStress = E.norm(); 
						
						tet->lastStress = 0.05f * currentStress + 0.95f * tet->lastStress;

						for (int i = 0; i < 4; ++i) {
							#pragma omp atomic
							tet->vertices[i]->lastStress += tet->lastStress;
							#pragma omp atomic
							tet->vertices[i]->connectedTets++;
						}
					}
				}

				// 3. Spatial Laplacian Smoothing (Parallelized)
				for (int iter = 0; iter < 2; ++iter) { 
					#pragma omp parallel for
					for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
						Group& group = object.getGroup(groupIdx);
						for (Tetrahedron* tet : group.tetrahedra) {
							for (int i = 0; i < 4; ++i) {
								for (int j = i + 1; j < 4; ++j) {
									float avg = (tet->vertices[i]->lastStress / std::max(1, tet->vertices[i]->connectedTets) + 
												 tet->vertices[j]->lastStress / std::max(1, tet->vertices[j]->connectedTets)) * 0.5f;
									
									// Atomic updates to ensure thread safety
									float updateI = (avg * tet->vertices[i]->connectedTets - tet->vertices[i]->lastStress) * 0.1f;
									float updateJ = (avg * tet->vertices[j]->connectedTets - tet->vertices[j]->lastStress) * 0.1f;
									
									#pragma omp atomic
									tet->vertices[i]->lastStress += updateI;
									#pragma omp atomic
									tet->vertices[j]->lastStress += updateJ;
								}
							}
						}
					}
				}
			}

			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			if (showFiberFlow) {
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			}

			glBegin(GL_TRIANGLES);
			for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
				Group& group = object.getGroup(groupIdx);
				Eigen::Vector3f offset = Eigen::Vector3f::Zero();
				if (showExplodedView) {
					offset = (group.initCOM - globalInitCOM) * explodedScale;
				}
				for (Tetrahedron* tet : group.tetrahedra) {
					Vertex* v[4] = { tet->vertices[0], tet->vertices[1], tet->vertices[2], tet->vertices[3] };

					auto setVertexColor = [&](Vertex* vert) {
						float alpha = showFiberFlow ? 0.4f : 1.0f;
						if (showVolumePreservation) {
							// Color code based on volume preservation: green = good, red = volume loss
							float volumeRatio = (initialVolume > 1e-6f) ? (currentVolume / initialVolume) : 1.0f;
							// Map volume ratio to color: 1.0 = green, < 0.95 = red
							float r = std::max(0.0f, std::min(1.0f, 2.0f * (1.0f - volumeRatio)));
							float g = std::max(0.0f, std::min(1.0f, 2.0f * (volumeRatio - 0.5f)));
							float b = 0.2f;
							glColor4f(r, g, b, alpha);
						} else if (showStressCloud) {
							float avgStress = vert->connectedTets > 0 ? vert->lastStress / vert->connectedTets : 0.0f;
							float v = std::min(1.0f, avgStress * stressGain); 
							float r = std::max(0.0f, std::min(1.0f, 1.5f - std::abs(v * 4.0f - 3.0f)));
							float g = std::max(0.0f, std::min(1.0f, 1.5f - std::abs(v * 4.0f - 2.0f)));
							float b = std::max(0.0f, std::min(1.0f, 1.5f - std::abs(v * 4.0f - 1.0f)));
							glColor4f(r, g, b, alpha);
						} else if (showExplodedView) {
							float hue = (360.0f * groupIdx) / groupNum;
							float saturation = 0.45f; 
							float value = 0.95f;
							float red, green, blue;
							hsvToRgb(hue, saturation, value, red, green, blue);
							glColor4f(red, green, blue, alpha);
						} else {
							float hue = (360.0f * groupIdx) / groupNum;
							float saturation = 1.0f; 
							float value = 1.0f;
							float red, green, blue;
							hsvToRgb(hue, saturation, value, red, green, blue);
							glColor4f(red, green, blue, alpha);
						}
					};

					// Face 1
					setVertexColor(v[0]); glVertex3f(v[0]->x + offset.x(), v[0]->y + offset.y(), v[0]->z + offset.z());
					setVertexColor(v[1]); glVertex3f(v[1]->x + offset.x(), v[1]->y + offset.y(), v[1]->z + offset.z());
					setVertexColor(v[2]); glVertex3f(v[2]->x + offset.x(), v[2]->y + offset.y(), v[2]->z + offset.z());

					// Face 2
					setVertexColor(v[0]); glVertex3f(v[0]->x + offset.x(), v[0]->y + offset.y(), v[0]->z + offset.z());
					setVertexColor(v[1]); glVertex3f(v[1]->x + offset.x(), v[1]->y + offset.y(), v[1]->z + offset.z());
					setVertexColor(v[3]); glVertex3f(v[3]->x + offset.x(), v[3]->y + offset.y(), v[3]->z + offset.z());

					// Face 3
					setVertexColor(v[0]); glVertex3f(v[0]->x + offset.x(), v[0]->y + offset.y(), v[0]->z + offset.z());
					setVertexColor(v[2]); glVertex3f(v[2]->x + offset.x(), v[2]->y + offset.y(), v[2]->z + offset.z());
					setVertexColor(v[3]); glVertex3f(v[3]->x + offset.x(), v[3]->y + offset.y(), v[3]->z + offset.z());

					// Face 4
					setVertexColor(v[1]); glVertex3f(v[1]->x + offset.x(), v[1]->y + offset.y(), v[1]->z + offset.z());
					setVertexColor(v[2]); glVertex3f(v[2]->x + offset.x(), v[2]->y + offset.y(), v[2]->z + offset.z());
					setVertexColor(v[3]); glVertex3f(v[3]->x + offset.x(), v[3]->y + offset.y(), v[3]->z + offset.z());
				}
			}
			glEnd();
			
			// Draw initial outline for volume preservation comparison (wireframe of initial shape)
			if (showVolumePreservation && !initialPositions.empty()) {
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				glLineWidth(1.5f);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				if (whiteBackground) {
					glColor4f(0.3f, 0.3f, 0.3f, 0.4f); // Dark gray outline on white
				} else {
					glColor4f(0.7f, 0.7f, 0.7f, 0.4f); // Light gray outline on black
				}
				glBegin(GL_TRIANGLES);
				int posIdx = 0;
				for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
					Group& group = object.getGroup(groupIdx);
					for (Tetrahedron* tet : group.tetrahedra) {
						// Draw wireframe using initial positions (initx, inity, initz)
						Vertex* v[4] = {tet->vertices[0], tet->vertices[1], tet->vertices[2], tet->vertices[3]};
						
						// Face 1: vertices 0, 1, 2
						glVertex3f(v[0]->initx, v[0]->inity, v[0]->initz);
						glVertex3f(v[1]->initx, v[1]->inity, v[1]->initz);
						glVertex3f(v[2]->initx, v[2]->inity, v[2]->initz);
						
						// Face 2: vertices 0, 1, 3
						glVertex3f(v[0]->initx, v[0]->inity, v[0]->initz);
						glVertex3f(v[1]->initx, v[1]->inity, v[1]->initz);
						glVertex3f(v[3]->initx, v[3]->inity, v[3]->initz);
						
						// Face 3: vertices 0, 2, 3
						glVertex3f(v[0]->initx, v[0]->inity, v[0]->initz);
						glVertex3f(v[2]->initx, v[2]->inity, v[2]->initz);
						glVertex3f(v[3]->initx, v[3]->inity, v[3]->initz);
						
						// Face 4: vertices 1, 2, 3
						glVertex3f(v[1]->initx, v[1]->inity, v[1]->initz);
						glVertex3f(v[2]->initx, v[2]->inity, v[2]->initz);
						glVertex3f(v[3]->initx, v[3]->inity, v[3]->initz);
					}
				}
				glEnd();
				glDisable(GL_BLEND);
				// Restore filled mode for main rendering (edges will be drawn separately)
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
			
			if (showFiberFlow) {
				glDisable(GL_BLEND);
			}
		}
		
		// Restore wireframe mode after volume preservation rendering (if not in volume mode)
		if (!showVolumePreservation) {
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		}
		// Draw edges
		if (drawEdges) {
			glLineWidth(showStressCloud ? 1.0f : 2.5f);
			glBegin(GL_LINES);

			for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
				Group& group = object.getGroup(groupIdx);
				Eigen::Vector3f offset = Eigen::Vector3f::Zero();
				if (showExplodedView || showGhostLinks) {
					float currentExplodedScale = showExplodedView ? explodedScale : 0.15f; 
					offset = (group.initCOM - globalInitCOM) * currentExplodedScale;
				}
				for (Tetrahedron* tet : group.tetrahedra) {
					for (int edgeIdx = 0; edgeIdx < 6; ++edgeIdx) {
						Edge* edge = tet->edges[edgeIdx];
						Vertex* vertex1 = edge->vertices[0];
						Vertex* vertex2 = edge->vertices[1];
						bool isSurfaceEdge = edge->isBoundary;

						float red, green, blue;
						if (showStressCloud) {
							// For stress cloud, draw edges in a neutral color or hide them
							if (whiteBackground) {
								red = green = blue = 0.8f; // Light gray on white
							} else {
								red = green = blue = 0.2f; // Dark gray on black
							}
							if (!isSurfaceEdge) continue; // Only show surface edges in stress mode
						} else {
							float hue = (360.0f * groupIdx) / groupNum;
							float saturation = (showExplodedView || showGhostLinks) ? 0.45f : 1.0f;
							float value = (showExplodedView || showGhostLinks) ? 0.8f : 1.0f;
							hsvToRgb(hue, saturation, value, red, green, blue);

							if (isSurfaceEdge == false) {
								red = std::min(1.0f, red + 0.3f);
								green = std::min(1.0f, green + 0.3f);
								blue = std::min(1.0f, blue + 0.3f);
								float darkenFactor = 0.75f;
								red *= darkenFactor;
								green *= darkenFactor;
								blue *= darkenFactor;
							}
						}

						glColor3f(red, green, blue);
						glVertex3f(vertex1->x + offset.x(), vertex1->y + offset.y(), vertex1->z + offset.z());
						glVertex3f(vertex2->x + offset.x(), vertex2->y + offset.y(), vertex2->z + offset.z());
					}
				}
			}
			glEnd();
		}

		// Draw Fiber Flow (Anisotropic Fiber Directions)
		if (showFiberFlow) {
			glDisable(GL_DEPTH_TEST); // Ensure lines are visible through the mesh
			glLineWidth(2.0f);
			glBegin(GL_LINES);
			for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
				Group& group = object.getGroup(groupIdx);
				Eigen::Vector3f offset = Eigen::Vector3f::Zero();
				if (showExplodedView || showGhostLinks) {
					float currentExplodedScale = showExplodedView ? explodedScale : 0.15f;
					offset = (group.initCOM - globalInitCOM) * currentExplodedScale;
				}

				// Fiber direction in world space.
				// By default, E1 is the fiber direction and is aligned with the X-axis in the local frame.
				// We rotate it by the group's current rotation matrix.
				Eigen::Vector3f fiberDir = group.rotate_matrix * Eigen::Vector3f(1.0f, 0.0f, 0.0f);
				float lineLen = 0.015f; // Short line segment half-length

				for (Tetrahedron* tet : group.tetrahedra) {
					// Calculate tet center
					Eigen::Vector3f center(0, 0, 0);
					for (int i = 0; i < 4; ++i) {
						center += Eigen::Vector3f(tet->vertices[i]->x, tet->vertices[i]->y, tet->vertices[i]->z);
					}
					center /= 4.0f;
					center += offset; // Apply exploded view offset if any

					if (whiteBackground) glColor3f(0.2f, 0.5f, 0.2f); // Dark green on white
					else glColor3f(0.5f, 1.0f, 0.5f); // Bright green on black

					glVertex3f(center.x() - fiberDir.x() * lineLen, center.y() - fiberDir.y() * lineLen, center.z() - fiberDir.z() * lineLen);
					glVertex3f(center.x() + fiberDir.x() * lineLen, center.y() + fiberDir.y() * lineLen, center.z() + fiberDir.z() * lineLen);
				}
			}
			glEnd();
			glEnable(GL_DEPTH_TEST);
		}

		// --- NEW: Draw Single External Force Arrow (at demo center) ---
		if (anisoDemoState > 0 && anisoDemoVertex != nullptr) {
			glDisable(GL_DEPTH_TEST); // 让箭头始终可见，不被遮挡
			glLineWidth(4.0f);
			glBegin(GL_LINES);
			
			// Calculate offset for exploded view
			Eigen::Vector3f offset = Eigen::Vector3f::Zero();
			if (showExplodedView || showGhostLinks) {
				float currentExplodedScale = showExplodedView ? explodedScale : 0.15f;
				// Find the group containing the demo vertex
				for (int groupIdx = 0; groupIdx < groupNum; ++groupIdx) {
					Group& group = object.getGroup(groupIdx);
					if (group.verticesMap.find(anisoDemoVertex->index) != group.verticesMap.end()) {
						offset = (group.initCOM - globalInitCOM) * currentExplodedScale;
						break;
					}
				}
			}
			
			// Arrow start position (at demo vertex)
			Eigen::Vector3f start(anisoDemoVertex->x + offset.x(), 
			                      anisoDemoVertex->y + offset.y(), 
			                      anisoDemoVertex->z + offset.z());
			
			// Calculate force direction (diagonal: X + Y)
			float pulse = 0.5f + 0.5f * std::sin(glfwGetTime() * 4.0f);
			Eigen::Vector3f forceDir(1.0f, 1.0f, 0.0f);
			forceDir.normalize();
			
			// Anisotropic mode uses 2x force, so arrow should be 2x longer
			float forceMultiplier = (anisoDemoState == 2) ? 2.0f : 1.0f;
			
			// Longer arrow tail (increased scale)
			float fScale = 0.0003f; // 增加到原来的3倍
			float arrowLength = anisoDemoForceMag * forceMultiplier * pulse * fScale;
			Eigen::Vector3f end = start + forceDir * arrowLength;
			
			// Draw main arrow shaft (red)
			glColor3f(1.0f, 0.0f, 0.0f);
			glVertex3f(start.x(), start.y(), start.z());
			glVertex3f(end.x(), end.y(), end.z());
			
			// Draw arrowhead (V shape)
			Eigen::Vector3f dir = forceDir;
			Eigen::Vector3f side = dir.unitOrthogonal() * 0.03f; // Slightly larger arrowhead
			Eigen::Vector3f headBase = end - dir * 0.06f;
			
			glVertex3f(end.x(), end.y(), end.z());
			glVertex3f(headBase.x() + side.x(), headBase.y() + side.y(), headBase.z() + side.z());
			glVertex3f(end.x(), end.y(), end.z());
			glVertex3f(headBase.x() - side.x(), headBase.y() - side.y(), headBase.z() - side.z());
			
			glEnd();
			glEnable(GL_DEPTH_TEST);
		}

		// Draw Ghost Vertices (Connections between sub-groups - The "Coupling" visualization)
		if (showExplodedView || showGhostLinks) {
			float currentExplodedScale = showExplodedView ? explodedScale : 0.15f;
			glLineWidth(2.5f);
			glBegin(GL_LINES);
			// Yellowish/Cyan color for connections to stand out
			if (whiteBackground) glColor3f(0.0f, 0.5f, 0.7f); // Deep cyan on white
			else glColor3f(0.0f, 1.0f, 1.0f); // Bright cyan on black

			for (int i = 0; i < groupNum; ++i) {
				Group& g1 = object.getGroup(i);
				Eigen::Vector3f offset1 = (g1.initCOM - globalInitCOM) * currentExplodedScale;
				for (int dir = 0; dir < 6; ++dir) {
					int adjIdx = g1.adjacentGroupIDs[dir];
					if (adjIdx != -1 && adjIdx > i) { // Draw each pair once
						Group& g2 = object.getGroup(adjIdx);
						Eigen::Vector3f offset2 = (g2.initCOM - globalInitCOM) * currentExplodedScale;
						const auto& pairs = g1.commonVerticesInDirections[dir];
						for (size_t k = 0; k < pairs.first.size(); ++k) {
							Vertex* v1 = pairs.first[k];
							Vertex* v2 = pairs.second[k];
							glVertex3f(v1->x + offset1.x(), v1->y + offset1.y(), v1->z + offset1.z());
							glVertex3f(v2->x + offset2.x(), v2->y + offset2.y(), v2->z + offset2.z());
						}
					}
				}
			}
			glEnd();
		}

		
		//saveOBJ("43224.obj", object.groups);

		glPopMatrix();

		// ------------------ UI overlay (draw last)
		ui.beginDraw2D();
		// Left side buttons
		// ui.drawPanelBackground(uiPanelRect); // removed
		const bool canStartExp3 = !experiment3.isActive() && !experiment1.isActive() && !experiment2.isActive() && !experiment4.isActive();
		if (ui.button(uiRunRect, experiment3.buttonLabel(), canStartExp3)) {
			experiment3.requestStart();
		}
		const SimpleUI::Rect uiExp1Rect{ uiMargin, uiMargin + uiH + 8.0f, uiW, uiH };
		const bool canStartExp1 = !experiment3.isActive() && !experiment1.isActive() && !experiment2.isActive() && !experiment4.isActive();
		if (ui.button(uiExp1Rect, experiment1.buttonLabel(), canStartExp1)) {
			experiment1.requestStart();
		}
		const SimpleUI::Rect uiExp2Rect{ uiMargin, uiMargin + 2.0f * (uiH + 8.0f), uiW, uiH };
		const bool canStartExp2 = !experiment3.isActive() && !experiment1.isActive() && !experiment2.isActive() && !experiment4.isActive();
		if (ui.button(uiExp2Rect, experiment2.buttonLabel(), canStartExp2)) {
			experiment2.requestStart();
		}
		const SimpleUI::Rect uiExp4Rect{ uiMargin, uiMargin + 3.0f * (uiH + 8.0f), uiW, uiH };
		const bool canStartExp4 = !experiment3.isActive() && !experiment1.isActive() && !experiment2.isActive() && !experiment4.isActive();
		if (ui.button(uiExp4Rect, experiment4.buttonLabel(), canStartExp4)) {
			experiment4.requestStart();
		}

		// Right side buttons
		const float rightMargin = ui.state().windowWidth - uiW - uiMargin;
		const SimpleUI::Rect uiBgColorRect{ rightMargin, uiMargin, uiW, uiH };
		if (ui.button(uiBgColorRect, whiteBackground ? "Dark Background" : "White Background")) {
			whiteBackground = !whiteBackground;
		}

		const SimpleUI::Rect uiStressRect{ rightMargin, uiMargin + uiH + 8.0f, uiW, uiH };
		if (ui.button(uiStressRect, showStressCloud ? "Show Groups" : "Show Stress")) {
			showStressCloud = !showStressCloud;
			if (showStressCloud) {
				showFiberFlow = false;
				showGhostLinks = false;
				showExplodedView = false;
			}
		}

		const SimpleUI::Rect uiFiberRect{ rightMargin, uiMargin + 2.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiFiberRect, showFiberFlow ? "Hide Fiber" : "Show Fiber")) {
			showFiberFlow = !showFiberFlow;
			if (showFiberFlow) showStressCloud = false;
		}

		const SimpleUI::Rect uiGhostRect{ rightMargin, uiMargin + 3.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiGhostRect, showGhostLinks ? "Hide Coupling" : "Show Coupling")) {
			showGhostLinks = !showGhostLinks;
			if (showGhostLinks) showStressCloud = false;
		}

		const SimpleUI::Rect uiExplodedRect{ rightMargin, uiMargin + 4.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiExplodedRect, showExplodedView ? "Show Integrated" : "Exploded View")) {
			showExplodedView = !showExplodedView;
			if (showExplodedView) {
				showStressCloud = false;
				showVolumePreservation = false;
			}
		}

		const SimpleUI::Rect uiVolumePreservationRect{ rightMargin, uiMargin + 5.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiVolumePreservationRect, showVolumePreservation ? "Hide Volume Test" : "Volume Test")) {
			showVolumePreservation = !showVolumePreservation;
			if (showVolumePreservation) {
				// Disable other visualization modes
				showStressCloud = false;
				showExplodedView = false;
				showFiberFlow = false;
				showGhostLinks = false;
				
				// Calculate initial volume and store initial positions
				initialVolume = 0.0f;
				initialPositions.clear();
				for (int i = 0; i < object.groupNum; ++i) {
					Group& group = object.groups[i];
					for (Tetrahedron* tet : group.tetrahedra) {
						initialVolume += tet->calVolumeTetra();
					}
					for (const auto& vertexPair : group.verticesMap) {
						Vertex* vertex = vertexPair.second;
						initialPositions.push_back(Eigen::Vector3f(vertex->initx, vertex->inity, vertex->initz));
					}
				}
				
				// Find the top of the liver (max Y) and set plane constraint slightly below it
				float maxY = -std::numeric_limits<float>::max();
				for (const auto* v : objectUniqueVertices) {
					maxY = std::max(maxY, v->inity);
				}
				// Set plane at 80% of the height (to create visible compression)
				planeConstraintY = maxY * 0.8f;
				
				std::cout << "[Volume Preservation] Mode activated. Initial volume: " << initialVolume 
				          << ", Plane constraint Y: " << planeConstraintY << std::endl;
			} else {
				// Reset plane constraint when deactivated
				planeConstraintY = 0.0f;
			}
		}

		const SimpleUI::Rect uiPauseRect{ rightMargin, uiMargin + 6.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiPauseRect, isPaused ? "Resume(P)" : "Pause(P)")) {
			isPaused = !isPaused;
		}

		// --- NEW: Anisotropy Comparison Mode Button (3-State Cycle) ---
		const SimpleUI::Rect uiAnisoModeRect{ rightMargin, uiMargin + 7.0f * (uiH + 8.0f), uiW, uiH };
		const char* anisoLabel = "Demo: OFF";
		if (anisoDemoState == 1) anisoLabel = "Demo: Isotropic";
		else if (anisoDemoState == 2) anisoLabel = "Demo: Anisotropic";

		if (ui.button(uiAnisoModeRect, anisoLabel)) {
			anisoDemoState = (anisoDemoState + 1) % 3;

			if (anisoDemoState > 0) {
				// Initialize demo vertex if needed
				float maxX = -1e10f;
				for (int i = 0; i < object.groupNum; ++i) {
					for (auto& kv : object.groups[i].verticesMap) {
						Vertex* v = kv.second;
						if (!v->isFixed && v->initx > maxX) {
							maxX = v->initx;
							anisoDemoVertex = v;
						}
					}
				}

				if (anisoDemoState == 1) { // Isotropic
					youngs1 = 1000000.0f; 
					youngs2 = 1000000.0f;
					youngs3 = 1000000.0f;
					showFiberFlow = false;
				} else { // Anisotropic
					youngs1 = 20000000.0f; 
					youngs2 = 1000000.0f;
					youngs3 = 1000000.0f;
					showFiberFlow = true;
				}
			} else {
				anisoDemoVertex = nullptr;
				youngs1 = 1000000.0f; 
				youngs2 = 1000000.0f;
				youngs3 = 1000000.0f;
				showFiberFlow = false;
			}

			// Update Physics
			#pragma omp parallel for
			for (int i = 0; i < object.groupNum; ++i) {
				object.groups[i].calGroupKAni(youngs1, youngs2, youngs3, poisson);
				object.groups[i].calLHS();
			}
		}

		// Agent status label (digits/letters only for the tiny segment font).
		{
			const float sizePx = 10.0f;
			const float labelW = 420.0f;
			const float labelH = 28.0f;
			const float x = uiMargin;
			const float y = ui.state().windowHeight - uiMargin - labelH;
			const SimpleUI::Rect agentLabelRect{ x, y, labelW, labelH };

			if (agentSphere.enabled) {
				const std::string label =
					"AGENT ON VC " + std::string(agentUseVC ? "1" : "0") +
					" FX " + formatSignedInt(agentLastDeviceForceN.x()) +
					" FY " + formatSignedInt(agentLastDeviceForceN.y()) +
					" FZ " + formatSignedInt(agentLastDeviceForceN.z()) +
					" CNT " + std::to_string(agentLastContactCount);
				ui.drawLabel(agentLabelRect, label, sizePx);
			} else {
				ui.drawLabel(agentLabelRect, "AGENT OFF  PRESS H", sizePx);
			}
		}

		ui.endDraw2D();

		// Update background color
		if (whiteBackground) {
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		} else {
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		}

		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();

		//calculate frame rate
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0) { 
			printf("%d frames/sec\n", nbFrames);
			nbFrames = 0;
			lastTime += 1.0;
		}
		//printf("%d frame number\n", frame);
		frame++;
		//object.writeVerticesToFile("ourMethodResult.txt");
		/*object.bodyVolume = 0.0f;
		for (int i = 0; i < groupNum; i++)
		{
			object.groups[i].groupVolume = 0.0f;
		}
		for (int i = 0; i < groupNum; i++)
		{
			for (auto tets : object.groups[i].tetrahedra)
			{
				object.groups[i].groupVolume += tets->calVolumeTetra();
			}
			object.bodyVolume += object.groups[i].groupVolume;
		}*/
		
		//std::cout << object.bodyVolume << std::endl;
		
	
		/*double totalKE = 0.0;
		for (int i = 0; i < objectUniqueVertices.size(); i++) {
			double speedSquared = objectUniqueVertices[i]->velx * objectUniqueVertices[i]->velx + objectUniqueVertices[i]->vely * objectUniqueVertices[i]->vely + objectUniqueVertices[i]->velz * objectUniqueVertices[i]->velz;
			double kineticEnergy = 0.5 * objectUniqueVertices[i]->vertexMass * speedSquared;
			totalKE += kineticEnergy;
		}*/
		//double totalMass = 0.0;
		//double centerX = 0.0;
		//double centerY = 0.0;
		//double centerZ = 0.0;

		//for (int i = 0; i < objectUniqueVertices.size(); i++) {
		//	double vertexMass = objectUniqueVertices[i]->vertexMass;
		//	double vertexX = objectUniqueVertices[i]->x;
		//	double vertexY = objectUniqueVertices[i]->y;
		//	double vertexZ = objectUniqueVertices[i]->z;

		//	totalMass += vertexMass;
		//	centerX += vertexX * vertexMass;
		//	centerY += vertexY * vertexMass;
		//	centerZ += vertexZ * vertexMass;
		//}

		//if (totalMass != 0) {
		//	centerX /= totalMass;
		//	centerY /= totalMass;
		//	centerZ /= totalMass;
		//}
		//else {
		//	// Handle the case where totalMass is 0 to avoid division by zero
		//	centerX = 0.0;
		//	centerY = 0.0;
		//	centerZ = 0.0;
		//}

		// Output the center of mass
		//std::cout << "Center of Mass: (" << centerX << ", " << centerY << ", " << centerZ << ")" << std::endl;

	}
	
	
	glfwTerminate();
	return 0;
}
