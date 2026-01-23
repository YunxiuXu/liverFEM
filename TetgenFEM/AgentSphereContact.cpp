#include "AgentSphereContact.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Geometry>

#include "Vertex.h"

namespace {
Eigen::Vector3f closestPointOnTriangle(
	const Eigen::Vector3f& p,
	const Eigen::Vector3f& a,
	const Eigen::Vector3f& b,
	const Eigen::Vector3f& c,
	Eigen::Vector3f* baryOut)
{
	// Real-Time Collision Detection (Christer Ericson), closest point on triangle.
	const Eigen::Vector3f ab = b - a;
	const Eigen::Vector3f ac = c - a;
	const Eigen::Vector3f ap = p - a;

	const float d1 = ab.dot(ap);
	const float d2 = ac.dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) {
		if (baryOut) *baryOut = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
		return a;
	}

	const Eigen::Vector3f bp = p - b;
	const float d3 = ab.dot(bp);
	const float d4 = ac.dot(bp);
	if (d3 >= 0.0f && d4 <= d3) {
		if (baryOut) *baryOut = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
		return b;
	}

	const float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		const float v = d1 / (d1 - d3);
		if (baryOut) *baryOut = Eigen::Vector3f(1.0f - v, v, 0.0f);
		return a + v * ab;
	}

	const Eigen::Vector3f cp = p - c;
	const float d5 = ab.dot(cp);
	const float d6 = ac.dot(cp);
	if (d6 >= 0.0f && d5 <= d6) {
		if (baryOut) *baryOut = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
		return c;
	}

	const float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		const float w = d2 / (d2 - d6);
		if (baryOut) *baryOut = Eigen::Vector3f(1.0f - w, 0.0f, w);
		return a + w * ac;
	}

	const float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		const Eigen::Vector3f bc = c - b;
		const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		if (baryOut) *baryOut = Eigen::Vector3f(0.0f, 1.0f - w, w);
		return b + w * bc;
	}

	const float denom = 1.0f / (va + vb + vc);
	const float v = vb * denom;
	const float w = vc * denom;
	if (baryOut) *baryOut = Eigen::Vector3f(1.0f - v - w, v, w);
	return a + ab * v + ac * w;
}

bool outwardNormalForTriangle(
	const AgentTriangle& tri,
	const Eigen::Vector3f& a,
	const Eigen::Vector3f& b,
	const Eigen::Vector3f& c,
	Eigen::Vector3f* outwardUnitOut)
{
	const Eigen::Vector3f nRaw = (b - a).cross(c - a);
	const float n2 = nRaw.squaredNorm();
	if (n2 <= 1e-24f) return false;

	// If we have an interior reference (opp vertex of the boundary face), orient the normal so that it
	// points away from the interior half-space. Do this without relying on the input winding.
	Eigen::Vector3f n = nRaw;
	if (tri.opp) {
		const Eigen::Vector3f opp(tri.opp->x, tri.opp->y, tri.opp->z);
		const float sOpp = nRaw.dot(opp - a);
		// If opp lies on the + side of nRaw, then outward is -nRaw; if opp lies on the - side, outward is +nRaw.
		// If sOpp is nearly zero (degenerate), keep the raw normal.
		if (std::abs(sOpp) > 1e-18f) {
			if (sOpp > 0.0f) n = -nRaw;
			else n = nRaw;
		}
	}
	*outwardUnitOut = n.normalized();
	return true;
}
} // namespace

AgentContactResult applyAgentSphereContact(
	const AgentSphere& agent,
	const std::vector<Vertex*>& contactVertices,
	float timeStep,
	std::vector<Eigen::Vector3f>& vertexAccels,
	float forceWeight)
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
	float nx = 0.0f, ny = 0.0f, nz = 0.0f;
	float maxPenetration = 0.0f;
	int contacts = 0;

#pragma omp parallel for reduction(+:fx,fy,fz,nx,ny,nz,contacts) reduction(max:maxPenetration)
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
		maxPenetration = std::max(maxPenetration, penetration);

		const Eigen::Vector3f n = d / dist; // from agent center to vertex

		const Eigen::Vector3f vtxVel(v->velx, v->vely, v->velz);
		const float relVn = (vtxVel - agent.velocity).dot(n);
		const float dampMag = -c * std::min(0.0f, relVn); // only resist inward motion

		// Add smooth extra stiffness for deep penetration to prevent tunneling
		const float penRatio = penetration / std::max(1e-6f, r);
		// Start extra penalty later (50%) and grow slower to allow softer deep press
		const float extraStiffness = (penRatio > 0.50f) ? (k * (penRatio - 0.50f) * 2.0f) : 0.0f;
		const float accelMag = k * penetration + dampMag + extraStiffness * penetration;
		const Eigen::Vector3f accel = n * (accelMag * forceWeight);

		vertexAccels[v->index] += accel;
		++contacts;

		const Eigen::Vector3f fOnVertex = n * accelMag * v->vertexMass;
		// Reaction force on agent is opposite.
		fx -= fOnVertex.x();
		fy -= fOnVertex.y();
		fz -= fOnVertex.z();

		// Track average normal direction (weighted by penetration).
		nx += n.x() * penetration;
		ny += n.y() * penetration;
		nz += n.z() * penetration;
	}

	result.reactionForceN = Eigen::Vector3f(fx, fy, fz);
	result.contactVertexCount = contacts;
	result.maxPenetration = maxPenetration;
	{
		const Eigen::Vector3f sumN(nx, ny, nz);
		const float nlen = sumN.norm();
		if (nlen > 1e-12f) result.avgNormal = sumN / nlen;
	}
	return result;
}

AgentContactResult applyAgentSphereTriangleContact(
	const AgentSphere& agent,
	const std::vector<AgentTriangle>& contactTriangles,
	float timeStep,
	std::vector<Eigen::Vector3f>& vertexAccels,
	float forceWeight)
{
	AgentContactResult result{};
	if (!agent.enabled || contactTriangles.empty() || vertexAccels.empty()) {
		return result;
	}

	(void)timeStep;
	const float r = std::max(0.0f, agent.radius);
	const float k = std::max(0.0f, agent.contactStiffness);
	const float c = std::max(0.0f, agent.contactDamping);

	float fx = 0.0f, fy = 0.0f, fz = 0.0f;
	float nx = 0.0f, ny = 0.0f, nz = 0.0f;
	float maxPenetration = 0.0f;
	int contacts = 0;

#pragma omp parallel for reduction(+:fx,fy,fz,nx,ny,nz,contacts) reduction(max:maxPenetration)
	for (int i = 0; i < static_cast<int>(contactTriangles.size()); ++i) {
		const AgentTriangle& tri = contactTriangles[i];
		if (!tri.a || !tri.b || !tri.c) continue;

		const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
		const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
		const Eigen::Vector3f cpos(tri.c->x, tri.c->y, tri.c->z);

		// Robust one-sided gating: skip faces for which the agent center is on the same side of the
		// triangle plane as the interior reference (opp). This avoids depending on face winding.
		Eigen::Vector3f outwardN = Eigen::Vector3f::Zero();
		if (!outwardNormalForTriangle(tri, a, b, cpos, &outwardN)) continue;
		if (tri.opp) {
			const Eigen::Vector3f opp(tri.opp->x, tri.opp->y, tri.opp->z);
			const Eigen::Vector3f nRaw = (b - a).cross(cpos - a);
			const float sOpp = nRaw.dot(opp - a);
			const float sP = nRaw.dot(agent.position - a);
			if (std::abs(sOpp) > 1e-18f && (sP * sOpp) > 0.0f) {
				// Agent is in the interior half-space of this boundary face.
				continue;
			}
		}

		Eigen::Vector3f bary(0.0f, 0.0f, 0.0f);
		const Eigen::Vector3f q = closestPointOnTriangle(agent.position, a, b, cpos, &bary);

		const Eigen::Vector3f d = q - agent.position;
		const float dist2 = d.squaredNorm();
		if (dist2 <= 1e-18f) continue;

		const float dist = std::sqrt(dist2);
		const float penetration = r - dist;
		if (penetration <= 0.0f) continue;
		maxPenetration = std::max(maxPenetration, penetration);

		const Eigen::Vector3f n = d / dist; // from agent center to surface point

		const Eigen::Vector3f va(tri.a->velx, tri.a->vely, tri.a->velz);
		const Eigen::Vector3f vb(tri.b->velx, tri.b->vely, tri.b->velz);
		const Eigen::Vector3f vc(tri.c->velx, tri.c->vely, tri.c->velz);
		const Eigen::Vector3f vq = va * bary.x() + vb * bary.y() + vc * bary.z();

		const float relVn = (vq - agent.velocity).dot(n);
		const float dampMag = -c * std::min(0.0f, relVn); // only resist inward motion

		// Add smooth extra stiffness for deep penetration to prevent tunneling
		const float penRatio = penetration / std::max(1e-6f, r);
		// Start extra penalty later (50%) and grow slower to allow softer deep press
		const float extraStiffness = (penRatio > 0.50f) ? (k * (penRatio - 0.50f) * 2.0f) : 0.0f;
		const float accelMag = k * penetration + dampMag + extraStiffness * penetration;
		const Eigen::Vector3f accel = n * (accelMag * forceWeight);

		float w0 = bary.x();
		float w1 = bary.y();
		float w2 = bary.z();

		// If some vertices are fixed, renormalize weights among the free ones.
		float wsum = 0.0f;
		if (tri.a && !tri.a->isFixed) wsum += w0;
		else w0 = 0.0f;
		if (tri.b && !tri.b->isFixed) wsum += w1;
		else w1 = 0.0f;
		if (tri.c && !tri.c->isFixed) wsum += w2;
		else w2 = 0.0f;
		if (wsum <= 1e-12f) continue;
		const float invWsum = 1.0f / wsum;
		w0 *= invWsum;
		w1 *= invWsum;
		w2 *= invWsum;

		auto applyToVertex = [&](Vertex* v, float w) {
			if (!v || w <= 0.0f || v->isFixed) return;
			if (v->index < 0 || v->index >= static_cast<int>(vertexAccels.size())) return;

			const Eigen::Vector3f delta = accel * w;

#pragma omp atomic
			vertexAccels[v->index].x() += delta.x();
#pragma omp atomic
			vertexAccels[v->index].y() += delta.y();
#pragma omp atomic
			vertexAccels[v->index].z() += delta.z();

			const Eigen::Vector3f fOnVertex = delta / forceWeight * v->vertexMass; // reaction force uses full magnitude
			fx -= fOnVertex.x();
			fy -= fOnVertex.y();
			fz -= fOnVertex.z();
		};

		applyToVertex(tri.a, w0);
		applyToVertex(tri.b, w1);
		applyToVertex(tri.c, w2);

		++contacts;

		nx += n.x() * penetration;
		ny += n.y() * penetration;
		nz += n.z() * penetration;
	}

	result.reactionForceN = Eigen::Vector3f(fx, fy, fz);
	result.contactVertexCount = contacts;
	result.maxPenetration = maxPenetration;
	{
		const Eigen::Vector3f sumN(nx, ny, nz);
		const float nlen = sumN.norm();
		if (nlen > 1e-12f) result.avgNormal = sumN / nlen;
	}
	return result;
}

AgentSurfaceQueryResult queryAgentSurface(
	const Eigen::Vector3f& p,
	const std::vector<AgentTriangle>& contactTriangles)
{
	AgentSurfaceQueryResult out{};
	if (contactTriangles.empty()) return out;

	float bestDist2 = std::numeric_limits<float>::infinity();
	Eigen::Vector3f bestQ = Eigen::Vector3f::Zero();
	Eigen::Vector3f bestOutwardN = Eigen::Vector3f::Zero();
	float bestSignedPlaneDist = 0.0f;

	for (const auto& tri : contactTriangles) {
		if (!tri.a || !tri.b || !tri.c) continue;
		const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
		const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
		const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);

		Eigen::Vector3f outwardN = Eigen::Vector3f::Zero();
		if (!outwardNormalForTriangle(tri, a, b, c, &outwardN)) continue;

		Eigen::Vector3f bary(0.0f, 0.0f, 0.0f);
		const Eigen::Vector3f q = closestPointOnTriangle(p, a, b, c, &bary);
		const float d2 = (q - p).squaredNorm();
		if (d2 < bestDist2) {
			bestDist2 = d2;
			bestQ = q;
			bestOutwardN = outwardN;
			bestSignedPlaneDist = outwardN.dot(p - a);
		}
	}

	if (!std::isfinite(bestDist2)) return out;
	out.found = true;
	out.closestPoint = bestQ;
	out.outwardNormal = bestOutwardN;
	out.signedPlaneDistance = bestSignedPlaneDist;
	out.distanceToSurface = std::sqrt(std::max(0.0f, bestDist2));
	return out;
}
