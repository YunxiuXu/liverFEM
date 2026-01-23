#pragma once

#include <Eigen/Core>
#include <vector>

class Vertex;

struct AgentTriangle {
	Vertex* a = nullptr;
	Vertex* b = nullptr;
	Vertex* c = nullptr;
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
