#pragma once

#include <Eigen/Core>
#include <vector>

class Vertex;

struct AgentTriangle {
	Vertex* a = nullptr;
	Vertex* b = nullptr;
	Vertex* c = nullptr;
	// If provided, this vertex is on the interior side of the boundary face, used to orient an outward normal.
	Vertex* opp = nullptr;
};

struct AgentSphere {
	bool enabled = false;
	Eigen::Vector3f position = Eigen::Vector3f::Zero();
	Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
	float radius = 0.05f;

	// Contact is applied as an acceleration (same units as Gravity/dragForces in this codebase).
	float contactStiffness = 8000.0f; // 1/s^2
	float contactDamping = 50.0f;     // 1/s
};

struct AgentContactResult {
	Eigen::Vector3f reactionForceN = Eigen::Vector3f::Zero(); // force on the agent (Newton)
	int contactVertexCount = 0;
	float maxPenetration = 0.0f;
	Eigen::Vector3f avgNormal = Eigen::Vector3f::Zero();
};

// Applies penalty-based sphere contact to vertices. Writes per-vertex accelerations into `vertexAccels`
// (accumulated via +=). Returns total reaction force on the agent.
AgentContactResult applyAgentSphereContact(
	const AgentSphere& agent,
	const std::vector<Vertex*>& contactVertices,
	float timeStep,
	std::vector<Eigen::Vector3f>& vertexAccels);

// Triangle-based contact against a surface mesh (recommended). Writes per-vertex accelerations into `vertexAccels`
// (accumulated via +=). Returns total reaction force on the agent.
AgentContactResult applyAgentSphereTriangleContact(
	const AgentSphere& agent,
	const std::vector<AgentTriangle>& contactTriangles,
	float timeStep,
	std::vector<Eigen::Vector3f>& vertexAccels);

struct AgentSurfaceQueryResult {
	bool found = false;
	Eigen::Vector3f closestPoint = Eigen::Vector3f::Zero();
	Eigen::Vector3f outwardNormal = Eigen::Vector3f::Zero(); // unit-length if found
	float signedPlaneDistance = 0.0f; // along outwardNormal, positive means outside w.r.t. closest face plane
	float distanceToSurface = 0.0f;   // |p-closestPoint|
};

// Finds closest point on the (deforming) surface triangle mesh and an outward normal estimate (if opp is set).
AgentSurfaceQueryResult queryAgentSurface(
	const Eigen::Vector3f& p,
	const std::vector<AgentTriangle>& contactTriangles);
