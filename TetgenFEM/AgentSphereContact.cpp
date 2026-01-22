#include "AgentSphereContact.h"

#include <algorithm>
#include <cmath>

#include "Vertex.h"

AgentContactResult applyAgentSphereContact(
	const AgentSphere& agent,
	const std::vector<Vertex*>& contactVertices,
	float timeStep,
	std::vector<Eigen::Vector3f>& vertexAccels)
{
	AgentContactResult result{};
	if (!agent.enabled || contactVertices.empty() || vertexAccels.empty()) {
		return result;
	}

	(void)timeStep;
	const float r = std::max(0.0f, agent.radius);
	const float k = std::max(0.0f, agent.contactStiffness);
	const float c = std::max(0.0f, agent.contactDamping);

	float fx = 0.0f, fy = 0.0f, fz = 0.0f;
	int contacts = 0;

#pragma omp parallel for reduction(+:fx,fy,fz,contacts)
	for (int i = 0; i < static_cast<int>(contactVertices.size()); ++i) {
		Vertex* v = contactVertices[i];
		if (!v || v->isFixed) continue;
		if (v->index < 0 || v->index >= static_cast<int>(vertexAccels.size())) continue;

		const Eigen::Vector3f p(v->x, v->y, v->z);
		const Eigen::Vector3f d = p - agent.position;
		const float dist2 = d.squaredNorm();
		if (dist2 <= 1e-18f) continue;

		const float dist = std::sqrt(dist2);
		const float penetration = r - dist;
		if (penetration <= 0.0f) continue;

		const Eigen::Vector3f n = d / dist; // from agent center to vertex

		const Eigen::Vector3f vtxVel(v->velx, v->vely, v->velz);
		const float relVn = (vtxVel - agent.velocity).dot(n);
		const float dampMag = -c * std::min(0.0f, relVn); // only resist inward motion

		const float accelMag = k * penetration + dampMag;
		const Eigen::Vector3f accel = n * accelMag;

		vertexAccels[v->index] += accel;
		++contacts;

		const Eigen::Vector3f fOnVertex = accel * v->vertexMass;
		// Reaction force on agent is opposite.
		fx -= fOnVertex.x();
		fy -= fOnVertex.y();
		fz -= fOnVertex.z();
	}

	result.reactionForceN = Eigen::Vector3f(fx, fy, fz);
	result.contactVertexCount = contacts;
	return result;
}
