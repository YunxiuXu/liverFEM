
//#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <array>
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

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
#include "LeapC.h"
#endif



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

	#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
	class LeapCTracker {
	public:
		~LeapCTracker() { shutdown(); }

	bool init()
	{
		if (connection_) return true;
		eLeapRS r = LeapCreateConnection(nullptr, &connection_);
		if (r != eLeapRS_Success || !connection_) {
			std::cerr << "[LeapC] LeapCreateConnection failed (" << static_cast<int>(r) << ")\n";
			connection_ = nullptr;
			return false;
		}
		r = LeapOpenConnection(connection_);
		if (r != eLeapRS_Success) {
			std::cerr << "[LeapC] LeapOpenConnection failed (" << static_cast<int>(r) << ")\n";
			shutdown();
			return false;
		}
		return true;
	}

		void shutdown()
		{
			if (!connection_) return;
			LeapCloseConnection(connection_);
			LeapDestroyConnection(connection_);
			connection_ = nullptr;
			connected_ = false;
			hasHand_ = false;
			timeSec_ = -1.0;
		}

	void poll(double nowSec)
	{
		if (!connection_) return;
		LEAP_CONNECTION_MESSAGE msg;
		for (;;) {
			const eLeapRS r = LeapPollConnection(connection_, 0, &msg);
			if (r != eLeapRS_Success) break;

			switch (msg.type) {
			case eLeapEventType_Connection:
				connected_ = true;
				break;
			case eLeapEventType_ConnectionLost:
				connected_ = false;
				break;
			case eLeapEventType_DeviceLost:
				// keep connection but mark stale
				break;
			case eLeapEventType_Tracking:
				onTracking(msg.tracking_event, nowSec);
				break;
			default:
				break;
			}
		}
	}

	bool isConnected() const { return connected_; }

		bool getRightHandTipsMm(std::array<Eigen::Vector3f, 5>* outTipsMm, Eigen::Vector3f* outPalmMm, double* outTimeSec) const
		{
			if (!hasHand_) return false;
			if (outTipsMm) *outTipsMm = tipsMm_;
			if (outPalmMm) *outPalmMm = palmMm_;
			if (outTimeSec) *outTimeSec = timeSec_;
			return true;
		}

	private:
		void onTracking(const LEAP_TRACKING_EVENT* evt, double nowSec)
		{
			if (!evt) return;
			for (uint32_t i = 0; i < evt->nHands; ++i) {
				const LEAP_HAND* hand = &evt->pHands[i];
				if (!hand) continue;
				if (hand->type != eLeapHandType_Right) continue;

				const LEAP_VECTOR palm = hand->palm.position; // mm
				palmMm_ = Eigen::Vector3f(palm.x, palm.y, palm.z);

				// digits[0]=thumb, [1]=index, [2]=middle, [3]=ring, [4]=pinky
				for (int fi = 0; fi < 5; ++fi) {
					const LEAP_VECTOR tip = hand->digits[fi].distal.next_joint; // mm
					tipsMm_[static_cast<size_t>(fi)] = Eigen::Vector3f(tip.x, tip.y, tip.z);
				}

				timeSec_ = nowSec;
				hasHand_ = true;
				return;
			}
		}

		LEAP_CONNECTION connection_ = nullptr;
		bool connected_ = false;
		bool hasHand_ = false;
		Eigen::Vector3f palmMm_ = Eigen::Vector3f::Zero();
		std::array<Eigen::Vector3f, 5> tipsMm_{};
		double timeSec_ = -1.0;
	};
	#endif

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

	static float signedTetraVolume(
		const Eigen::Vector3f& p0,
		const Eigen::Vector3f& p1,
		const Eigen::Vector3f& p2,
		const Eigen::Vector3f& p3)
	{
		return (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0f;
	}

	struct AgentPhysKey {
		long long x = 0;
		long long y = 0;
		long long z = 0;
		bool operator==(const AgentPhysKey& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
		bool operator<(const AgentPhysKey& o) const noexcept
		{
			if (x != o.x) return x < o.x;
			if (y != o.y) return y < o.y;
			return z < o.z;
		}
	};

	struct AgentPhysKeyHash {
		size_t operator()(const AgentPhysKey& k) const noexcept
		{
			const size_t h0 = std::hash<long long>{}(k.x);
			const size_t h1 = std::hash<long long>{}(k.y);
			const size_t h2 = std::hash<long long>{}(k.z);
			size_t h = h0;
			h ^= (h1 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
			h ^= (h2 + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
			return h;
		}
	};

	static AgentPhysKey makePhysKey(const Vertex* v)
	{
		// Quantize init position to group-duplicate vertices into a single "physical" vertex id.
		// This is required because Object::updateIndices() assigns unique indices per group, so index-based
		// surface extraction would incorrectly include internal group interfaces.
		constexpr double kQuant = 1000000.0; // 1e-6 resolution
		const auto q = [](float x) -> long long { return static_cast<long long>(std::llround(static_cast<double>(x) * kQuant)); };
		if (!v) return AgentPhysKey{};
		return AgentPhysKey{q(v->initx), q(v->inity), q(v->initz)};
	}

static Eigen::Vector3f closestPointOnTriangle(
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

static bool segmentSphereFirstHitT(
	const Eigen::Vector3f& p0,
	const Eigen::Vector3f& p1,
	float r,
	float* outT)
{
	const Eigen::Vector3f d = p1 - p0;
	const float a = d.dot(d);
	if (a <= 1e-20f) return false;

	const float b = 2.0f * p0.dot(d);
	const float c = p0.dot(p0) - r * r;
	const float disc = b * b - 4.0f * a * c;
	if (disc <= 0.0f) return false;

	const float s = std::sqrt(disc);
	float t0 = (-b - s) / (2.0f * a);
	float t1 = (-b + s) / (2.0f * a);
	if (t0 > t1) std::swap(t0, t1);

	if (t1 < 0.0f || t0 > 1.0f) return false;
	const float t = (t0 >= 0.0f) ? t0 : t1;
	if (t < 0.0f || t > 1.0f) return false;
	if (outT) *outT = t;
	return true;
}

static bool rayIntersectsTriangle(
	const Eigen::Vector3f& rayOrigin,
	const Eigen::Vector3f& rayDir,
	const Eigen::Vector3f& v0,
	const Eigen::Vector3f& v1,
	const Eigen::Vector3f& v2,
	float* tOut)
{
	// Möller–Trumbore ray/triangle intersection (two-sided).
	const Eigen::Vector3f e1 = v1 - v0;
	const Eigen::Vector3f e2 = v2 - v0;
	const Eigen::Vector3f pvec = rayDir.cross(e2);
	const float det = e1.dot(pvec);
	if (std::abs(det) < 1e-9f) return false;
	const float invDet = 1.0f / det;

	const Eigen::Vector3f tvec = rayOrigin - v0;
	const float u = tvec.dot(pvec) * invDet;
	if (u < 0.0f || u > 1.0f) return false;

	const Eigen::Vector3f qvec = tvec.cross(e1);
	const float v = rayDir.dot(qvec) * invDet;
	if (v < 0.0f || (u + v) > 1.0f) return false;

	const float t = e2.dot(qvec) * invDet;
	if (t <= 1e-6f) return false;
	if (tOut) *tOut = t;
	return true;
}

static bool isPointInsideSurfaceRayCast(
	const Eigen::Vector3f& p,
	const std::vector<AgentTriangle>& tris,
	float eps)
{
	if (tris.empty()) return false;
	const Eigen::Vector3f dir = Eigen::Vector3f(1.0f, 0.2345f, 0.3456f).normalized();
	const Eigen::Vector3f origin = p + dir * std::max(1e-9f, eps);
	int hits = 0;
	for (const auto& tri : tris) {
		if (!tri.a || !tri.b || !tri.c) continue;
		const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
		const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
		const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
		float t = 0.0f;
		if (rayIntersectsTriangle(origin, dir, a, b, c, &t)) {
			++hits;
		}
	}
	return (hits % 2) == 1;
}

static bool isPointInsideSurfaceRayCastMulti(
	const Eigen::Vector3f& p,
	const std::vector<AgentTriangle>& tris,
	float eps)
{
	if (tris.empty()) return false;

	// Use multiple fixed ray directions and take a majority vote to reduce flicker near edges/vertices.
	static const std::array<Eigen::Vector3f, 3> dirs = {
		Eigen::Vector3f(1.0f, 0.2345f, 0.3456f).normalized(),
		Eigen::Vector3f(-0.143f, 1.0f, 0.311f).normalized(),
		Eigen::Vector3f(0.271f, -0.531f, 1.0f).normalized(),
	};

	const float e = std::max(1e-9f, eps);
	int insideVotes = 0;
	for (const Eigen::Vector3f& dir : dirs) {
		const Eigen::Vector3f origin = p + dir * e;
		int hits = 0;
		for (const auto& tri : tris) {
			if (!tri.a || !tri.b || !tri.c) continue;
			const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
			const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
			const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
			float t = 0.0f;
			if (rayIntersectsTriangle(origin, dir, a, b, c, &t)) {
				++hits;
			}
		}
		if ((hits % 2) == 1) ++insideVotes;
	}

	return insideVotes >= 2;
}

struct ForceGraphHistory {
	static constexpr int kCapacity = 240;
	std::array<Eigen::Vector4f, kCapacity> samples{};
	int head = 0;     // next write position
			bool filled = false;

		void push(const Eigen::Vector4f& s) {
			samples[static_cast<size_t>(head)] = s;
			head = (head + 1) % kCapacity;
			if (head == 0) filled = true;
		}

		int size() const { return filled ? kCapacity : head; }

		Eigen::Vector4f at(int i) const {
			const int n = size();
			if (n <= 0) return Eigen::Vector4f::Zero();
			const int idx = filled ? ((head + i) % kCapacity) : i;
			if (idx < 0 || idx >= kCapacity) return Eigen::Vector4f::Zero();
		return samples[static_cast<size_t>(idx)];
	}
};

static int applyAxisAlignedWallConstraints(
	const std::vector<std::vector<Vertex*>>& verticesByPhysId,
	float wallXMax,
	float wallYMin,
	float wallYMax,
	float dt,
	float restitution,
	float tangentialDamp)
{
	if (verticesByPhysId.empty()) return 0;

	const float invDt = 1.0f / std::max(1e-8f, dt);
	const float rest = std::clamp(restitution, 0.0f, 1.0f);
	const float tanD = std::clamp(tangentialDamp, 0.0f, 1.0f);

	int hitCount = 0;
	for (const auto& list : verticesByPhysId) {
		if (list.empty()) continue;

		bool anyFixed = false;
		for (Vertex* v : list) {
			if (v && v->isFixed) {
				anyFixed = true;
				break;
			}
		}
		if (anyFixed) continue;

		Eigen::Vector3f pAvg = Eigen::Vector3f::Zero();
		Eigen::Vector3f vAvg = Eigen::Vector3f::Zero();
		int n = 0;
		for (Vertex* v : list) {
			if (!v) continue;
			pAvg += Eigen::Vector3f(v->x, v->y, v->z);
			vAvg += Eigen::Vector3f(v->velx, v->vely, v->velz);
			++n;
		}
		if (n <= 0) continue;
		pAvg /= static_cast<float>(n);
		vAvg /= static_cast<float>(n);

		Eigen::Vector3f pNew = pAvg;
		bool hit = false;
		bool hitXMax = false;
		bool hitYMax = false;
		bool hitYMin = false;

		if (pNew.x() > wallXMax) {
			pNew.x() = wallXMax;
			hit = true;
			hitXMax = true;
		}
		if (pNew.y() > wallYMax) {
			pNew.y() = wallYMax;
			hit = true;
			hitYMax = true;
		}
		if (pNew.y() < wallYMin) {
			pNew.y() = wallYMin;
			hit = true;
			hitYMin = true;
		}

		if (!hit) continue;

		const Eigen::Vector3f dpAvg = pNew - pAvg;
		Eigen::Vector3f vNew = vAvg + dpAvg * invDt;

		auto applyPlaneResponse = [&](const Eigen::Vector3f& outwardN) {
			const float vn = vNew.dot(outwardN);
			if (vn > 0.0f) {
				vNew -= outwardN * ((1.0f + rest) * vn);
			}
			const Eigen::Vector3f vt = vNew - outwardN * vNew.dot(outwardN);
			vNew -= vt * tanD;
		};

		if (hitXMax) applyPlaneResponse(Eigen::Vector3f::UnitX());
		if (hitYMax) applyPlaneResponse(Eigen::Vector3f::UnitY());
		if (hitYMin) applyPlaneResponse(-Eigen::Vector3f::UnitY());

		for (Vertex* v : list) {
			if (!v || v->isFixed) continue;
			v->x = pNew.x();
			v->y = pNew.y();
			v->z = pNew.z();
			v->velx = vNew.x();
			v->vely = vNew.y();
			v->velz = vNew.z();
		}
		++hitCount;
	}
	return hitCount;
}

// Add helper function for normal consistency
static bool outwardNormalForTriangle(
	const AgentTriangle& tri,
	const Eigen::Vector3f& a,
	const Eigen::Vector3f& b,
	const Eigen::Vector3f& c,
	Eigen::Vector3f* outwardUnitOut)
{
	const Eigen::Vector3f nRaw = (b - a).cross(c - a);
	const float n2 = nRaw.squaredNorm();
	if (n2 <= 1e-24f) return false;

	Eigen::Vector3f n = nRaw;
	if (tri.opp) {
		const Eigen::Vector3f opp(tri.opp->x, tri.opp->y, tri.opp->z);
		const float sOpp = nRaw.dot(opp - a);
		if (std::abs(sOpp) > 1e-18f) {
			if (sOpp > 0.0f) n = -nRaw;
			else n = nRaw;
		}
	}
	*outwardUnitOut = n.normalized();
	return true;
}

				static AgentContactResult solveAgentSphereTriangleCollisionConstraint(
					Eigen::Vector3f& sphereCenter,
					Eigen::Vector3f& sphereVel,
					float sphereInvMass,
					float proxyInvMassScale,
				float contactVelocityRelaxation,
				float contactVelocityRelaxationMin,
				float contactNormalDamp,
				float sphereRadius,
					float allowedPenetration,
					float eps,
					float dt,
				float positionCorrection,
				float tangentialDamp,
				float frictionMu,
				int iterations,
					int manifoldTriangles,
					int* activeTriangleIndexInOut,
					const Eigen::Vector3f& sphereDriveForceN,
					const std::vector<AgentTriangle>& contactTriangles,
					const std::vector<std::array<int, 3>>& contactTrianglePhysIds,
					const std::vector<std::vector<int>>& contactTriangleNeighbors,
					const std::vector<std::vector<Vertex*>>& verticesByPhysId,
					const std::vector<float>& physMassSumKg)
			{
				AgentContactResult out{};
				if (contactTriangles.empty() || contactTriangles.size() != contactTrianglePhysIds.size() || verticesByPhysId.empty() || physMassSumKg.empty()) return out;
				if (!contactTriangleNeighbors.empty() && contactTriangleNeighbors.size() != contactTriangles.size()) return out;

				const float invMs = std::max(0.0f, sphereInvMass);
				const float invMsResp = invMs * std::clamp(proxyInvMassScale, 0.0f, 1.0f);
				const float r = std::max(1e-6f, sphereRadius);
				const float allowedPen = std::clamp(allowedPenetration, 0.0f, 0.95f * r);
				const float targetR = std::max(0.0f, r - allowedPen);
				const float shellR = targetR + std::max(0.0f, eps);
				const float shellR2 = shellR * shellR;
				const float invDt = 1.0f / std::max(1e-8f, dt);
				const float invDt2 = invDt * invDt;

				const float corr = std::clamp(positionCorrection, 0.0f, 1.0f);
				const float tanDamp = std::clamp(tangentialDamp, 0.0f, 1.0f);
			const float mu = std::clamp(frictionMu, 0.0f, 10.0f);
			const int iters = std::clamp(iterations, 1, 64);

		Eigen::Vector3f sumN = Eigen::Vector3f::Zero();
		float maxPen = 0.0f;
		Eigen::Vector3f maxPenNormal = Eigen::Vector3f::Zero();
		int movedVertexCount = 0;

			auto invMassOfPhysId = [&](int id) -> float {
				if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) return 0.0f;
				const auto& list = verticesByPhysId[static_cast<size_t>(id)];
				if (list.empty() || !list.front()) return 0.0f;
				if (list.front()->isFixed) return 0.0f;
				if (id < 0 || id >= static_cast<int>(physMassSumKg.size())) return 0.0f;
				const float m = physMassSumKg[static_cast<size_t>(id)];
				return (m > 1e-12f) ? (1.0f / m) : 0.0f;
			};

				const float velRelaxBase = std::clamp(contactVelocityRelaxation, 0.0f, 1.0f);
				const float velRelaxMin = std::clamp(contactVelocityRelaxationMin, 0.0f, velRelaxBase);
				const float nDamp = std::clamp(contactNormalDamp, 0.0f, 1.0f);

				auto applyDeltaToPhysId = [&](int id, const Eigen::Vector3f& dp, const Eigen::Vector3f& n) {
				if (dp.squaredNorm() <= 1e-24f) return;
				if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) return;
				const auto& list = verticesByPhysId[static_cast<size_t>(id)];
				for (Vertex* v : list) {
					if (!v || v->isFixed) continue;

				v->x += dp.x();
				v->y += dp.y();
				v->z += dp.z();

					Eigen::Vector3f vv(v->velx, v->vely, v->velz);
					// Apply velocity correction with relaxation to prevent oscillation.
					// Adaptive relaxation: if correction is large (fast impact), use LESS velocity feedback
					// to avoid adding huge energy to the system.
					float velocityRelaxation = velRelaxBase;
					const float corrLen = dp.norm();
					if (corrLen > 1e-4f) {
						// Reduce relaxation for large corrections, but less aggressively (100 -> 20)
						// to allow better penetration response without oscillation.
						velocityRelaxation = std::max(velRelaxMin, velRelaxBase / (1.0f + 20.0f * corrLen));
					}
					vv += dp * invDt * velocityRelaxation;

					Eigen::Vector3f relV = vv - sphereVel;
					float dvN = std::max(0.0f, dp.dot(n) * invDt);

					const float vnBefore = relV.dot(n);
					if (vnBefore < 0.0f) {
						// Remove inward normal velocity (non-penetration in velocity space).
						relV -= n * vnBefore;
						dvN += -vnBefore;
					}

					// Coulomb friction in velocity space (enables stick-slip without relying on penetration depth).
					if (mu > 0.0f && dvN > 0.0f) {
						const float vnAfter = relV.dot(n);
						const Eigen::Vector3f vt = relV - n * vnAfter;
						const float vtLen = vt.norm();
						if (vtLen > 1e-8f) {
							const float maxDvT = mu * dvN;
							const float dvT = std::min(vtLen, maxDvT);
							const Eigen::Vector3f dvt = -vt * (dvT / vtLen);
							relV += dvt;

							const float m = std::max(0.0f, v->vertexMass);
							out.reactionForceN -= dvt * (m * invDt);
						}
					}

					// Extra tangential damping (0..1): helps suppress chatter and improves "sticky drag".
					relV -= (relV - n * relV.dot(n)) * tanDamp;

					// Safety: never re-introduce inward normal velocity via friction corrections.
					const float vnAfterFriction = relV.dot(n);
					if (vnAfterFriction < 0.0f) {
						relV -= n * vnAfterFriction;
					}

					// Dampen outward normal relative velocity while PRESSING to prevent "buzzing" at contact.
					// Only apply when the proxy drive is into the surface (avoid sticky adhesion on release).
					if (nDamp > 0.0f) {
						float driveInto = 0.0f;
						if (sphereDriveForceN.squaredNorm() > 1e-12f) driveInto = sphereDriveForceN.dot(n);
						else driveInto = sphereVel.dot(n);
						if (driveInto > 0.0f) {
							const float vnRel = relV.dot(n);
							if (vnRel > 0.0f) relV -= n * (vnRel * nDamp);
						}
					}
					vv = sphereVel + relV;

					v->velx = vv.x();
					v->vely = vv.y();
					v->velz = vv.z();

					const float m = std::max(0.0f, v->vertexMass);
						out.reactionForceN -= dp * (m * invDt2);
						++movedVertexCount;
					}
				};

					// Contact manifold: solve a small set of nearest surface triangles.
					// Using >1 triangle near edges greatly reduces "falling into cracks" and contact buzzing.
					int preferredTi = -1;
					if (activeTriangleIndexInOut && *activeTriangleIndexInOut >= 0 &&
						*activeTriangleIndexInOut < static_cast<int>(contactTriangles.size())) {
						preferredTi = *activeTriangleIndexInOut;
					}

					const int manifoldK = std::clamp(manifoldTriangles, 1, 8);
					std::vector<int> activeTis;
					activeTis.reserve(static_cast<size_t>(manifoldK));

						auto triInShell = [&](int ti, float* dist2Out) -> bool {
							const auto& tri = contactTriangles[ti];
							if (!tri.a || !tri.b || !tri.c) return false;

						const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
						const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
						const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);

							// NOTE: Do NOT use one-sided gating here.
							// If the fingertip proxy sphere center crosses the surface (deep indentation),
							// one-sided gating would drop contact candidates entirely, causing "no force" and oscillation.

						Eigen::Vector3f bary(0.0f, 0.0f, 0.0f);
						const Eigen::Vector3f q = closestPointOnTriangle(sphereCenter, a, b, c, &bary);
						const float dist2 = (q - sphereCenter).squaredNorm();
						if (dist2 >= shellR2) return false;
						if (dist2Out) *dist2Out = dist2;
							return true;
						};

						// Distance^2 to a triangle (two-sided, for temporal tracking even when the proxy is far / inside).
						auto triDistance2 = [&](int ti) -> float {
							if (ti < 0 || ti >= static_cast<int>(contactTriangles.size())) return std::numeric_limits<float>::infinity();
							const auto& tri = contactTriangles[ti];
							if (!tri.a || !tri.b || !tri.c) return std::numeric_limits<float>::infinity();
							const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
							const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
							const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
							const Eigen::Vector3f q = closestPointOnTriangle(sphereCenter, a, b, c, nullptr);
							return (q - sphereCenter).squaredNorm();
						};

						auto findActiveSet = [&]() -> bool {
							activeTis.clear();
							std::vector<float> bestD2;
							bestD2.reserve(static_cast<size_t>(manifoldK));

						auto tryInsert = [&](int ti, float d2) {
							for (int existing : activeTis) {
								if (existing == ti) return;
							}

							if (static_cast<int>(activeTis.size()) < manifoldK) {
								activeTis.push_back(ti);
								bestD2.push_back(d2);
								return;
							}

							int worst = 0;
							for (int i = 1; i < static_cast<int>(bestD2.size()); ++i) {
								if (bestD2[i] > bestD2[worst]) worst = i;
							}
							if (d2 < bestD2[worst]) {
								activeTis[worst] = ti;
								bestD2[worst] = d2;
								}
							};

							// Temporal coherence: walk the preferred triangle along the surface (neighbor descent)
							// so that even after "in-air" motion we remain near the closest region without scanning
							// all triangles.
							if (preferredTi >= 0 && !contactTriangleNeighbors.empty()) {
								const int maxWalkSteps = 24;
								float best = triDistance2(preferredTi);
								for (int step = 0; step < maxWalkSteps; ++step) {
									int bestTi = preferredTi;
									float bestD = best;
									const auto& nbs = contactTriangleNeighbors[static_cast<size_t>(preferredTi)];
									for (int nb : nbs) {
										const float d = triDistance2(nb);
										if (d < bestD) {
											bestD = d;
											bestTi = nb;
										}
									}
									if (bestTi == preferredTi) break;
									preferredTi = bestTi;
									best = bestD;
								}
							}

							// Local manifold search around the preferred triangle.
							if (preferredTi >= 0 && !contactTriangleNeighbors.empty()) {
								const int maxVisited = 128;
								std::vector<int> queue;
								queue.reserve(static_cast<size_t>(maxVisited));

								auto pushUnique = [&](int ti) {
									if (ti < 0 || ti >= static_cast<int>(contactTriangles.size())) return;
									for (int v : queue) {
										if (v == ti) return;
									}
									queue.push_back(ti);
								};

								pushUnique(preferredTi);
								size_t qHead = 0;
								while (qHead < queue.size() && static_cast<int>(queue.size()) < maxVisited) {
									const int ti = queue[qHead++];
									float d2 = 0.0f;
									if (triInShell(ti, &d2)) tryInsert(ti, d2);

									const auto& nbs = contactTriangleNeighbors[static_cast<size_t>(ti)];
									for (int nb : nbs) {
										if (static_cast<int>(queue.size()) >= maxVisited) break;
										pushUnique(nb);
									}
								}
							} else {
								// Fallback: slow scan (should be rare; e.g., if neighbors not provided).
								if (preferredTi >= 0) {
									float d2 = 0.0f;
									if (triInShell(preferredTi, &d2)) tryInsert(preferredTi, d2);
								}
								for (int ti = 0; ti < static_cast<int>(contactTriangles.size()); ++ti) {
									if (ti == preferredTi) continue;
									float d2 = 0.0f;
									if (!triInShell(ti, &d2)) continue;
									tryInsert(ti, d2);
								}
							}

							if (activeTis.empty()) return false;

						// Sort by distance (K is small).
						for (int i = 0; i < static_cast<int>(activeTis.size()); ++i) {
							for (int j = i + 1; j < static_cast<int>(activeTis.size()); ++j) {
								if (bestD2[j] < bestD2[i]) {
									std::swap(bestD2[i], bestD2[j]);
									std::swap(activeTis[i], activeTis[j]);
								}
							}
						}

							preferredTi = activeTis.front();
							return true;
						};

					bool anyShellContact = false;
					for (int it = 0; it < iters; ++it) {
						if (activeTis.empty() && !findActiveSet()) break;

						bool anyPenetration = false;
						float bestD2ThisIter = std::numeric_limits<float>::infinity();
						int bestTiThisIter = -1;

						for (int ti : activeTis) {
							const auto& tri = contactTriangles[ti];
							const auto& ids = contactTrianglePhysIds[ti];
							if (!tri.a || !tri.b || !tri.c) continue;

							const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
							const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
							const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);

							// Re-check one-sided gating (positions may have moved).
							// FIXED: Disabled one-sided gating to allow deep penetration handling.
							// The original logic filtered out triangles if the sphere center was on the "inside" (opp side).
							// This caused collision loss when the finger pressed deep into the tissue.
							/*
							if (tri.opp) {
								const Eigen::Vector3f opp(tri.opp->x, tri.opp->y, tri.opp->z);
								const Eigen::Vector3f nRaw = (b - a).cross(c - a);
								const float n2 = nRaw.squaredNorm();
								if (n2 > 1e-24f) {
									const float sOpp = nRaw.dot(opp - a);
									const float sP = nRaw.dot(sphereCenter - a);
									if (std::abs(sOpp) > 1e-18f && (sP * sOpp) > 0.0f) continue;
								}
							}
							*/

							Eigen::Vector3f bary(0.0f, 0.0f, 0.0f);
							const Eigen::Vector3f q = closestPointOnTriangle(sphereCenter, a, b, c, &bary);
							const float dist2 = (q - sphereCenter).squaredNorm();
							if (dist2 >= shellR2) continue;
							anyShellContact = true;

							if (dist2 < bestD2ThisIter) {
								bestD2ThisIter = dist2;
								bestTiThisIter = ti;
							}

							// Check if sphere center is behind the triangle face (inside the object).
							Eigen::Vector3f outwardN;
							bool hasOutward = outwardNormalForTriangle(tri, a, b, c, &outwardN);
							bool isInside = false;
							
							Eigen::Vector3f d = q - sphereCenter;
							const float dist = std::sqrt(std::max(dist2, 1e-18f));
							
							if (hasOutward) {
								// d points from center to surface. If d aligns with outward normal, center is inside.
								// We use a small tolerance.
								if (d.dot(outwardN) > 1e-4f * dist) {
									isInside = true;
								}
							}

							float pen = 0.0f;
							Eigen::Vector3f n;

							if (isInside) {
								// Deep penetration handling:
								// If center is inside, we must push it out along the normal.
								// Continuity: at surface (dist=0), pen = shellR.
								// Inside (dist>0), pen should increase.
								pen = shellR + dist;
								
								// Cap huge penetrations to avoid explosion if we detect false positives far away
								if (pen > 3.0f * shellR) continue;

								// We need deltaQ (correction displacement) to be along -outwardN (inwards),
								// so that sphere update (-deltaQ) is along outwardN (outwards).
								n = -outwardN;
							} else {
								// Standard case: center is outside
								pen = shellR - dist;
								if (pen <= 0.0f) continue;

								n = (dist > 1e-6f) ? (d / dist) : Eigen::Vector3f(0.0f, 1.0f, 0.0f);
								
								// Fallback normal if d is degenerate
								if (dist <= 1e-6f && hasOutward) {
									n = -outwardN; // Point inward
								}
							}

							anyPenetration = true;

							// Robustness: deep penetration logic is now handled by isInside branch.
							// We keep maxPen updating logic.
							maxPen = std::max(maxPen, pen);
							if (pen >= maxPen) maxPenNormal = n;
							sumN += n * pen;

							const float invMa = invMassOfPhysId(ids[0]);
							const float invMb = invMassOfPhysId(ids[1]);
							const float invMc = invMassOfPhysId(ids[2]);

							const float w0 = bary.x();
							const float w1 = bary.y();
							const float w2 = bary.z();
							const float denom = (w0 * w0) * invMa + (w1 * w1) * invMb + (w2 * w2) * invMc + invMsResp;
							if (denom <= 1e-18f) continue;

							float correctionMag = pen * corr;
							const float maxCorrPerIter = 0.05f * shellR;
							if (correctionMag > maxCorrPerIter) correctionMag = maxCorrPerIter;

							const Eigen::Vector3f deltaQ = n * correctionMag;
							if (invMsResp > 0.0f) {
								const Eigen::Vector3f dpSphere = -deltaQ * (invMsResp / denom);
								sphereCenter += dpSphere;
								sphereVel += dpSphere * invDt;
							}
							applyDeltaToPhysId(ids[0], deltaQ * (w0 * invMa / denom), n);
							applyDeltaToPhysId(ids[1], deltaQ * (w1 * invMb / denom), n);
							applyDeltaToPhysId(ids[2], deltaQ * (w2 * invMc / denom), n);
						}

						// Update preferred triangle for temporal coherence.
						if (bestTiThisIter >= 0) preferredTi = bestTiThisIter;

						// If we didn't apply any correction this iteration, we're done.
						if (!anyPenetration) break;

						// Keep the active set only while at least one triangle remains in the shell.
						bool anyInShell = false;
						for (int ti : activeTis) {
							float d2 = 0.0f;
							if (triInShell(ti, &d2)) {
								anyInShell = true;
								break;
							}
						}
						if (!anyInShell) activeTis.clear();
					}

						if (activeTriangleIndexInOut) {
							// Keep the preferred triangle even when we're not in contact; this avoids expensive
							// re-search when hovering just outside the surface.
							if (preferredTi >= 0) *activeTriangleIndexInOut = preferredTi;
						}

			out.contactVertexCount = movedVertexCount;
			out.maxPenetration = maxPen;
			{
				const float nlen = sumN.norm();
				if (nlen > 1e-12f) out.avgNormal = sumN / nlen;
			}

			// Static friction pass (even when there is little/no penetration correction): use the temporally coherent
			// preferred surface triangle (no global scan) and the VC drive force as a normal-force estimate.
			if (mu > 0.0f && invMs > 0.0f && sphereDriveForceN.squaredNorm() > 1e-12f &&
				preferredTi >= 0 && preferredTi < static_cast<int>(contactTriangles.size())) {
				const auto& tri = contactTriangles[preferredTi];
				if (tri.a && tri.b && tri.c) {
					const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
					const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
					const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);

					// Same one-sided gating as the normal solver.
					bool frictionAllowed = true;
					if (tri.opp) {
						const Eigen::Vector3f opp(tri.opp->x, tri.opp->y, tri.opp->z);
						const Eigen::Vector3f nRaw = (b - a).cross(c - a);
						const float n2 = nRaw.squaredNorm();
						if (n2 > 1e-24f) {
							const float sOpp = nRaw.dot(opp - a);
							const float sP = nRaw.dot(sphereCenter - a);
							if (std::abs(sOpp) > 1e-18f && (sP * sOpp) > 0.0f) frictionAllowed = false;
						}
					}

					if (frictionAllowed) {
						Eigen::Vector3f bary(0.0f, 0.0f, 0.0f);
						const Eigen::Vector3f q = closestPointOnTriangle(sphereCenter, a, b, c, &bary);
						const float d2 = (q - sphereCenter).squaredNorm();
						const float dist = std::sqrt(std::max(d2, 0.0f));
						const float fricR = r + std::max(0.0f, eps);
						if (dist <= fricR && dist > 1e-8f) {
							const Eigen::Vector3f n = (q - sphereCenter) / dist; // from center to contact point
							const float normalForceFromDrive = std::max(0.0f, sphereDriveForceN.dot(n));
							const float normalForceFromContact = std::max(0.0f, (-out.reactionForceN).dot(n));
							const float normalForceMag = std::max(normalForceFromDrive, normalForceFromContact);
							const float Jn = normalForceMag * dt;
							if (Jn > 1e-8f) {
								const auto& ids = contactTrianglePhysIds[preferredTi];
								const float w0 = bary.x();
								const float w1 = bary.y();
								const float w2 = bary.z();

								// Average per-phys-id velocity for stability.
								auto physVel = [&](int id) -> Eigen::Vector3f {
									if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) return Eigen::Vector3f::Zero();
									const auto& list = verticesByPhysId[static_cast<size_t>(id)];
									Eigen::Vector3f v = Eigen::Vector3f::Zero();
									int n = 0;
									for (Vertex* vx : list) {
										if (!vx) continue;
										v += Eigen::Vector3f(vx->velx, vx->vely, vx->velz);
										++n;
									}
									if (n > 0) {
										v /= static_cast<float>(n);
									} else {
										v.setZero();
									}
									return v;
								};

								const Eigen::Vector3f va = physVel(ids[0]);
								const Eigen::Vector3f vb = physVel(ids[1]);
								const Eigen::Vector3f vc = physVel(ids[2]);
								const Eigen::Vector3f vq = va * w0 + vb * w1 + vc * w2;

								Eigen::Vector3f relV = vq - sphereVel;
								const float vn = relV.dot(n);
								const Eigen::Vector3f vt = relV - n * vn;
								const float vtLen = vt.norm();
								if (vtLen > 1e-6f) {
									const float invMa = invMassOfPhysId(ids[0]);
									const float invMb = invMassOfPhysId(ids[1]);
									const float invMc = invMassOfPhysId(ids[2]);
									const float invEff = (w0 * w0) * invMa + (w1 * w1) * invMb + (w2 * w2) * invMc + invMsResp;
									if (invEff > 1e-18f) {
										Eigen::Vector3f Jt = -vt / invEff;
										const float JtMax = mu * Jn;
										const float JtLen = Jt.norm();
										if (JtLen > JtMax) {
											Jt *= (JtMax / std::max(1e-12f, JtLen));
										}

										// Apply to sphere (dynamic proxy) and to all duplicates of each phys vertex.
										if (invMsResp > 0.0f) {
											sphereVel += (-Jt) * invMsResp;
										}

										auto applyDvToPhysId = [&](int id, const Eigen::Vector3f& dv) {
											if (dv.squaredNorm() <= 1e-20f) return;
											if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) return;
											const auto& list = verticesByPhysId[static_cast<size_t>(id)];
											for (Vertex* vx : list) {
												if (!vx || vx->isFixed) continue;
												vx->velx += dv.x();
												vx->vely += dv.y();
												vx->velz += dv.z();
											}
										};

										applyDvToPhysId(ids[0], (w0 * Jt) * invMa);
										applyDvToPhysId(ids[1], (w1 * Jt) * invMb);
										applyDvToPhysId(ids[2], (w2 * Jt) * invMc);
									}
								}
							}
						}
					}
				}
			}

				// Stabilize deep contact: after all iterations, remove any proxy velocity component that
				// would move further "into" the contact normal direction this frame (prevents chatter).
				if (movedVertexCount > 0) {
				Eigen::Vector3f n = Eigen::Vector3f::Zero();
				if (out.avgNormal.squaredNorm() > 1e-12f) {
					n = out.avgNormal;
				} else if (maxPenNormal.squaredNorm() > 1e-12f) {
					n = maxPenNormal.normalized();
				}
				if (n.squaredNorm() > 1e-12f) {
					const float vn = sphereVel.dot(n);
					if (vn > 0.0f) sphereVel -= n * vn;
				}

				// Additional damping for very deep indentation: suppresses the classic "penetrate / depenetrate"
				// oscillation when VC keeps driving the proxy into a highly compressed region.
				// (Shallow contact remains responsive.)
				const float penFrac = maxPen / std::max(1e-6f, r);
				if (penFrac > 0.25f) {
					const float t = std::clamp((penFrac - 0.25f) / 0.75f, 0.0f, 1.0f);
					sphereVel *= (1.0f - 0.8f * t);
				}
			}
		return out;
	}

			static AgentContactResult solveAgentSphereVertexCollisionConstraint(
				Eigen::Vector3f& sphereCenter,
				Eigen::Vector3f& sphereVel,
				float sphereInvMass,
				float proxyInvMassScale,
				float contactVelocityRelaxation,
				float contactVelocityRelaxationMin,
				float contactNormalDamp,
				float sphereRadius,
				float allowedPenetration,
				float eps,
				float dt,
			float positionCorrection,
			float tangentialDamp,
			float frictionMu,
			int iterations,
			const Eigen::Vector3f& sphereDriveForceN,
			const std::vector<int>& contactVertexPhysIds,
			const std::vector<std::vector<Vertex*>>& verticesByPhysId,
			const std::vector<float>& physMassSumKg)
		{
			AgentContactResult out{};
			if (contactVertexPhysIds.empty() || verticesByPhysId.empty() || physMassSumKg.empty()) return out;

				const float invMs = std::max(0.0f, sphereInvMass);
				const float invMsResp = invMs * std::clamp(proxyInvMassScale, 0.0f, 1.0f);
				const float r = std::max(1e-6f, sphereRadius);
				const float allowedPen = std::clamp(allowedPenetration, 0.0f, 0.95f * r);
				const float targetR = std::max(0.0f, r - allowedPen);
				const float shellR = targetR + std::max(0.0f, eps);
				const float shellR2 = shellR * shellR;
				const float invDt = 1.0f / std::max(1e-8f, dt);
			const float invDt2 = invDt * invDt;

			const float corr = std::clamp(positionCorrection, 0.0f, 1.0f);
			const float tanDamp = std::clamp(tangentialDamp, 0.0f, 1.0f);
			const float mu = std::clamp(frictionMu, 0.0f, 10.0f);
			const int iters = std::clamp(iterations, 1, 64);

		Eigen::Vector3f sumN = Eigen::Vector3f::Zero();
		float maxPen = 0.0f;
		Eigen::Vector3f maxPenNormal = Eigen::Vector3f::Zero();
		int movedVertexCount = 0;

			auto invMassOfPhysId = [&](int id) -> float {
				if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) return 0.0f;
				const auto& list = verticesByPhysId[static_cast<size_t>(id)];
				if (list.empty() || !list.front()) return 0.0f;
				if (list.front()->isFixed) return 0.0f;
				if (id < 0 || id >= static_cast<int>(physMassSumKg.size())) return 0.0f;
				const float m = physMassSumKg[static_cast<size_t>(id)];
				return (m > 1e-12f) ? (1.0f / m) : 0.0f;
			};

			const float velRelaxBase = std::clamp(contactVelocityRelaxation, 0.0f, 1.0f);
			const float velRelaxMin = std::clamp(contactVelocityRelaxationMin, 0.0f, velRelaxBase);
			const float nDamp = std::clamp(contactNormalDamp, 0.0f, 1.0f);

			auto applyDeltaToPhysId = [&](int id, const Eigen::Vector3f& dp, const Eigen::Vector3f& n) {
				if (dp.squaredNorm() <= 1e-24f) return;
				if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) return;
				const auto& list = verticesByPhysId[static_cast<size_t>(id)];
				for (Vertex* v : list) {
					if (!v || v->isFixed) continue;

				v->x += dp.x();
				v->y += dp.y();
				v->z += dp.z();

					Eigen::Vector3f vv(v->velx, v->vely, v->velz);
					// Apply velocity correction with relaxation to prevent oscillation.
					// Adaptive relaxation: if correction is large (fast impact), use LESS velocity feedback
					// to avoid adding huge energy to the system.
					float velocityRelaxation = velRelaxBase;
					const float corrLen = dp.norm();
					if (corrLen > 1e-4f) {
						// Reduce relaxation for large corrections, but less aggressively (100 -> 20)
						// to allow better penetration response without oscillation.
						velocityRelaxation = std::max(velRelaxMin, velRelaxBase / (1.0f + 20.0f * corrLen));
					}
					vv += dp * invDt * velocityRelaxation;

					Eigen::Vector3f relV = vv - sphereVel;
					float dvN = std::max(0.0f, dp.dot(n) * invDt);

					const float vnBefore = relV.dot(n);
					if (vnBefore < 0.0f) {
						relV -= n * vnBefore;
						dvN += -vnBefore;
					}

					if (mu > 0.0f && dvN > 0.0f) {
						const Eigen::Vector3f vt = relV - n * relV.dot(n);
						const float vtLen = vt.norm();
						if (vtLen > 1e-8f) {
							const float maxDvT = mu * dvN;
							const float dvT = std::min(vtLen, maxDvT);
							const Eigen::Vector3f vtNew = vt * ((vtLen - dvT) / vtLen);
							const Eigen::Vector3f dvt = vtNew - vt;
							relV += dvt;

							const float m = std::max(0.0f, v->vertexMass);
							out.reactionForceN -= dvt * (m * invDt);
						}
					}

					relV -= (relV - n * relV.dot(n)) * tanDamp;
					if (nDamp > 0.0f) {
						float driveInto = 0.0f;
						if (sphereDriveForceN.squaredNorm() > 1e-12f) driveInto = sphereDriveForceN.dot(n);
						else driveInto = sphereVel.dot(n);
						if (driveInto > 0.0f) {
							const float vnRel = relV.dot(n);
							if (vnRel > 0.0f) relV -= n * (vnRel * nDamp);
						}
					}
					vv = sphereVel + relV;

					v->velx = vv.x();
					v->vely = vv.y();
					v->velz = vv.z();

					const float m = std::max(0.0f, v->vertexMass);
						out.reactionForceN -= dp * (m * invDt2);
						++movedVertexCount;
					}
				};

			for (int it = 0; it < iters; ++it) {
				bool any = false;
				for (int id : contactVertexPhysIds) {
					if (id < 0 || id >= static_cast<int>(verticesByPhysId.size())) continue;
				const auto& list = verticesByPhysId[static_cast<size_t>(id)];
				if (list.empty()) continue;
				Vertex* vRef = list.front();
				if (!vRef) continue;

				const Eigen::Vector3f p(vRef->x, vRef->y, vRef->z);
				Eigen::Vector3f d = p - sphereCenter;
				const float dist2 = d.squaredNorm();
				if (dist2 >= shellR2) continue;

					const float dist = std::sqrt(std::max(dist2, 1e-18f));
					const Eigen::Vector3f n = (dist > 1e-6f) ? (d / dist) : Eigen::Vector3f(0.0f, 1.0f, 0.0f);
					const float pen = shellR - dist;
					if (pen <= 0.0f) continue;

				maxPen = std::max(maxPen, pen);
				if (pen >= maxPen) {
					maxPenNormal = n;
				}
				sumN += n * pen;

					const float invMv = invMassOfPhysId(id);
					const float denom = invMv + invMsResp;
					if (denom <= 1e-18f) continue;

					const Eigen::Vector3f dp = n * (pen * corr);
					if (invMsResp > 0.0f) {
						const Eigen::Vector3f dpSphere = -dp * (invMsResp / denom);
						sphereCenter += dpSphere;
						sphereVel += dpSphere * invDt;
					}
					applyDeltaToPhysId(id, dp * (invMv / denom), n);
					any = true;
				}
				if (!any) break;
			}

			out.contactVertexCount = movedVertexCount;
			out.maxPenetration = maxPen;
			{
				const float nlen = sumN.norm();
				if (nlen > 1e-12f) out.avgNormal = sumN / nlen;
		}

			// Same stabilization as triangle contact.
			if (movedVertexCount > 0) {
				Eigen::Vector3f n = Eigen::Vector3f::Zero();
				if (out.avgNormal.squaredNorm() > 1e-12f) {
					n = out.avgNormal;
				} else if (maxPenNormal.squaredNorm() > 1e-12f) {
					n = maxPenNormal.normalized();
				}
				if (n.squaredNorm() > 1e-12f) {
					const float vn = sphereVel.dot(n);
					if (vn > 0.0f) sphereVel -= n * vn;
				}

				const float penFrac = maxPen / std::max(1e-6f, r);
				if (penFrac > 0.25f) {
					const float t = std::clamp((penFrac - 0.25f) / 0.75f, 0.0f, 1.0f);
					sphereVel *= (1.0f - 0.8f * t);
				}
			}
		return out;
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
		// NOTE: The old code hard-fixed an internal "back slice" anchor region via Object::fixRegion(),
		// which prevents whole-body motion (you only feel "poking" instead of pushing/rolling/flipping).
		// Anchoring is now configurable (none / fixed / spring) and is initialized later in phys-id space.
	
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

		static constexpr int kFingerCount = 5;
		static constexpr int kIndexFinger = 1;
		const std::array<const char*, kFingerCount> kFingerNames = { "THUMB", "INDEX", "MIDDLE", "RING", "PINKY" };
	
		// Agent sphere ("finger") parameters (shared across all fingertips).
		AgentSphere agentSphere;
		agentSphere.enabled = agentEnabled;
		agentSphere.radius = std::max(1e-6f, agentRadiusBboxScale * bboxDiag);
		agentSphere.contactStiffness = agentContactStiffness;
		agentSphere.contactDamping = agentContactDamping;
		agentSphere.influenceRadiusFrac = agentInfluenceRadiusFrac;
	
		// Initialize a 5-finger "hand" above the model.
		const Eigen::Vector3f agentHandHomeAnchor(bboxCenter.x(), bboxMax.y() + agentSphere.radius * 1.5f, bboxCenter.z());
		agentSphere.position = agentHandHomeAnchor;
	
		const float rFinger = agentSphere.radius;
		const std::array<Eigen::Vector3f, kFingerCount> agentHandFingerOffsets = {
			Eigen::Vector3f(-2.2f * rFinger, 0.0f, -1.5f * rFinger), // thumb
			Eigen::Vector3f(-1.1f * rFinger, 0.0f, 0.0f),            // index
			Eigen::Vector3f(0.0f, 0.0f, 0.0f),                       // middle
			Eigen::Vector3f(1.1f * rFinger, 0.0f, 0.0f),             // ring
			Eigen::Vector3f(2.2f * rFinger, 0.0f, 0.0f)              // pinky
		};
	
		std::array<Eigen::Vector3f, kFingerCount> agentDevicePositions;
		std::array<Eigen::Vector3f, kFingerCount> agentDevicePrevPositions;
		std::array<Eigen::Vector3f, kFingerCount> agentDeviceVelocities;
		std::array<Eigen::Vector3f, kFingerCount> agentProxyPositions;
		std::array<Eigen::Vector3f, kFingerCount> agentProxyVelocities;
		std::array<Eigen::Vector3f, kFingerCount> agentHomePositions;
	
		for (int fi = 0; fi < kFingerCount; ++fi) {
			const Eigen::Vector3f p = agentHandHomeAnchor + agentHandFingerOffsets[static_cast<size_t>(fi)];
			agentHomePositions[static_cast<size_t>(fi)] = p;
			agentDevicePositions[static_cast<size_t>(fi)] = p;
			agentDevicePrevPositions[static_cast<size_t>(fi)] = p;
			agentDeviceVelocities[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
			agentProxyPositions[static_cast<size_t>(fi)] = p;
			agentProxyVelocities[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
		}
	
					std::array<Eigen::Vector3f, kFingerCount> agentLastDeviceForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentLastContactForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentLastCouplingForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentFilteredDeviceForcesN;
					std::array<int, kFingerCount> agentLastContactCounts{};
					std::array<float, kFingerCount> agentLastContactPenetrations{};
					std::array<int, kFingerCount> agentLastActiveContactTriangle{};
						agentLastDeviceForcesN.fill(Eigen::Vector3f::Zero());
						agentLastContactForcesN.fill(Eigen::Vector3f::Zero());
						agentLastCouplingForcesN.fill(Eigen::Vector3f::Zero());
						agentFilteredDeviceForcesN.fill(Eigen::Vector3f::Zero());
						agentLastActiveContactTriangle.fill(-1);
		
	#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
		LeapCTracker leapTracker;
		bool leapUseInput = leapEnabled;
		bool leapMappingCalibrated = false;
		Eigen::Vector3f leapCenterMm = Eigen::Vector3f::Zero();
		Eigen::Vector3f leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)];
		std::array<Eigen::Vector3f, kFingerCount> leapLatestTipsMm;
		leapLatestTipsMm.fill(Eigen::Vector3f::Zero());
		double leapLatestTimeSec = -1.0;
		if (leapUseInput && !leapTracker.init()) {
			std::cerr << "[LeapC] init failed; disabling Leap input.\n";
			leapUseInput = false;
		}
	#endif

		float objectMassKg = 0.0f;
		for (const auto* v : objectUniqueVertices) {
			objectMassKg += std::max(0.0f, v->vertexMass);
		}
	const float agentProxyMassKgTotal = std::max(1e-6f, std::abs(agentProxyMassFracOfObject) * std::max(1e-6f, objectMassKg));
	const float agentProxyMassKg = std::max(1e-6f, agentProxyMassKgTotal / static_cast<float>(kFingerCount));
	const float invBboxDiag = 1.0f / std::max(1e-6f, bboxDiag);
	const float agentVcKLen = std::max(0.0f, agentVcStiffnessNPerBbox) * invBboxDiag;     // N per unit length
	float agentVcCLenFree = 0.0f;
	float agentVcCLenContact = 0.0f;
	if (agentVcAutoDamping) {
		const float zFree = std::clamp(agentVcDampingRatioFree, 0.0f, 20.0f);
		const float zContact = std::clamp(agentVcDampingRatioContact, 0.0f, 20.0f);
		const float cCrit = (agentVcKLen > 0.0f && agentProxyMassKg > 1e-12f)
			? (2.0f * std::sqrt(agentVcKLen * agentProxyMassKg))
			: 0.0f;
		agentVcCLenFree = zFree * cCrit;
		agentVcCLenContact = zContact * cCrit;
	} else {
		agentVcCLenFree = std::max(0.0f, agentVcDampingNsPerBbox) * invBboxDiag;                 // N*s per unit length
		agentVcCLenContact = std::max(0.0f, agentVcDampingNsPerBboxInContact) * invBboxDiag;     // N*s per unit length
	}

		// Choose contact vertices/triangles on the OUTER surface.
		//
		// Important: Object::updateIndices() duplicates vertices across groups (unique indices per group),
		// so index-based face counting would incorrectly classify internal group interfaces as "surface".
		// We therefore build a "physical vertex id" by quantized init position and do surface extraction
		// in that space.
			std::vector<Vertex*> agentContactVertices;
			std::vector<AgentTriangle> agentContactTriangles;
			std::vector<std::array<int, 3>> agentContactTrianglePhysIds;
			std::vector<std::vector<int>> agentContactTriangleNeighbors;
			std::vector<int> agentContactVertexPhysIds;
			std::vector<std::vector<Vertex*>> agentVerticesByPhysId;
		agentContactVertices.reserve(objectUniqueVertices.size());
		agentContactTrianglePhysIds.reserve(objectUniqueVertices.size());
		agentContactVertexPhysIds.reserve(objectUniqueVertices.size());

		// Build physical id map from unique (deduped) init positions.
		std::unordered_map<AgentPhysKey, int, AgentPhysKeyHash> physKeyToId;
		physKeyToId.reserve(objectUniqueVertices.size() * 2);
		std::vector<Vertex*> physRep;
		physRep.reserve(objectUniqueVertices.size());
		for (Vertex* v : objectUniqueVertices) {
			if (!v) continue;
			const int id = static_cast<int>(physRep.size());
			physKeyToId.emplace(makePhysKey(v), id);
			physRep.push_back(v);
		}

			// Build physId -> all vertex copies (across groups).
			agentVerticesByPhysId.assign(physRep.size(), {});
				for (int gi = 0; gi < object.groupNum; ++gi) {
					Group& g = object.groups[gi];
				for (const auto& vertexPair : g.verticesMap) {
					Vertex* v = vertexPair.second;
					if (!v) continue;
					const auto it = physKeyToId.find(makePhysKey(v));
					if (it == physKeyToId.end()) continue;
						agentVerticesByPhysId[static_cast<size_t>(it->second)].push_back(v);
					}
				}

				// Previous-frame physical positions (used for conservative vertex-vs-sphere CCD).
				std::vector<Eigen::Vector3f> physPrevPositions(agentVerticesByPhysId.size(), Eigen::Vector3f::Zero());
				for (size_t id = 0; id < agentVerticesByPhysId.size(); ++id) {
					const auto& list = agentVerticesByPhysId[id];
					if (list.empty() || !list.front()) continue;
					const Vertex* v = list.front();
					physPrevPositions[id] = Eigen::Vector3f(v->x, v->y, v->z);
				}

				// ------------------ Optional: object anchoring (none / fixed / spring)
				//
				// This is critical for "whole organ manipulation": a hard-fixed anchor makes the organ immovable
			// (only local indentation). For interactive tasks like pushing/rolling/flipping, set anchor_mode=0
			// or use a soft spring anchor (anchor_mode=2).
			const int anchorModeClamped = std::clamp(anchorMode, 0, 2);
			Eigen::Vector3f anchorCenterRest = Eigen::Vector3f::Zero();
			float anchorRegionRadius = 0.0f;
			std::vector<int> anchorPhysIds;
			bool anchorRegionValid = false;

			if (anchorModeClamped != 0 && !physRep.empty()) {
				const float depth = bboxMax.z() - bboxMin.z();
				const float sliceFrac = std::clamp(anchorBackSliceFrac, 0.0f, 1.0f);
				const float backSliceZ = bboxMin.z() + depth * sliceFrac;

				Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
				int count = 0;
				for (Vertex* v : physRep) {
					if (!v) continue;
					if (v->initz <= backSliceZ) {
						centroid.x() += v->initx;
						centroid.y() += v->inity;
						centroid.z() += v->initz;
						++count;
					}
				}
				if (count > 0) {
					centroid /= static_cast<float>(count);
				} else {
					centroid = Eigen::Vector3f(
						0.5f * (bboxMin.x() + bboxMax.x()),
						0.5f * (bboxMin.y() + bboxMax.y()),
						bboxMin.z());
				}

				// Slight push towards +Z to avoid anchoring only a thin surface shell.
				const float pushFrac = std::clamp(anchorCenterPushFrac, 0.0f, 1.0f);
				centroid.z() = std::min(centroid.z() + depth * pushFrac, bboxMax.z());

				anchorCenterRest = centroid;
				anchorRegionRadius = std::max(1e-6f, depth * std::max(0.0f, anchorRadiusDepthFrac));

				const float r2 = anchorRegionRadius * anchorRegionRadius;
				anchorPhysIds.reserve(std::min<size_t>(physRep.size(), 2048));
				for (int id = 0; id < static_cast<int>(physRep.size()); ++id) {
					Vertex* rep = physRep[static_cast<size_t>(id)];
					if (!rep) continue;
					const Eigen::Vector3f p(rep->initx, rep->inity, rep->initz);
					if ((p - anchorCenterRest).squaredNorm() <= r2) {
						anchorPhysIds.push_back(id);
					}
				}
				anchorRegionValid = !anchorPhysIds.empty();

				if (anchorModeClamped == 1 && anchorRegionValid) {
					int fixedCount = 0;
					for (int id : anchorPhysIds) {
						if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
						for (Vertex* v : agentVerticesByPhysId[static_cast<size_t>(id)]) {
							if (!v) continue;
							if (!v->isFixed) {
								v->isFixed = true;
								++fixedCount;
							}
						}
					}
					std::cout << "[Anchor] mode=1 (fixed) center=" << anchorCenterRest.transpose()
							  << " radius=" << anchorRegionRadius
							  << " fixedVerts=" << fixedCount << "\n";
				} else if (anchorModeClamped == 2 && anchorRegionValid) {
					std::cout << "[Anchor] mode=2 (spring) center=" << anchorCenterRest.transpose()
							  << " radius=" << anchorRegionRadius
							  << " physIds=" << anchorPhysIds.size()
							  << " k=" << anchorSpringK
							  << " damp=" << anchorSpringDamping
							  << " maxA=" << anchorSpringMaxAccel << "\n";
				} else if (!anchorRegionValid) {
					std::cout << "[Anchor] mode=" << anchorModeClamped << " but region selection found 0 vertices.\n";
				}
			} else {
				std::cout << "[Anchor] mode=0 (none)\n";
			}

			// Extract outer boundary triangles by counting faces in physId space.
			if (agentUseSurfaceTriangles && !physRep.empty()) {
				struct FaceKey {
					int i0 = -1, i1 = -1, i2 = -1;
				bool operator==(const FaceKey& o) const noexcept { return i0 == o.i0 && i1 == o.i1 && i2 == o.i2; }
			};
			struct FaceKeyHash {
				size_t operator()(const FaceKey& k) const noexcept
				{
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
				Vertex* a = nullptr;
				Vertex* b = nullptr;
				Vertex* c = nullptr;
				Vertex* opp = nullptr;
				std::array<int, 3> ids{ {-1, -1, -1} };
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

			auto physIdOf = [&](Vertex* v) -> int {
				if (!v) return -1;
				const auto it = physKeyToId.find(makePhysKey(v));
				return (it == physKeyToId.end()) ? -1 : it->second;
			};

			auto addFace = [&](Vertex* a, Vertex* b, Vertex* c, Vertex* opp) {
				const int ia = physIdOf(a);
				const int ib = physIdOf(b);
				const int ic = physIdOf(c);
				if (ia < 0 || ib < 0 || ic < 0) return;
				const FaceKey key = makeKey(ia, ib, ic);
				auto it = faces.find(key);
				if (it == faces.end()) {
					FaceRec rec;
					rec.count = 1;
					rec.a = a;
					rec.b = b;
					rec.c = c;
					rec.opp = opp;
					rec.ids = {ia, ib, ic};
					faces.emplace(key, rec);
				} else {
					it->second.count += 1;
				}
			};

			for (int gi = 0; gi < object.groupNum; ++gi) {
				Group& g = object.groups[gi];
				for (Tetrahedron* t : g.tetrahedra) {
					if (!t) continue;
					Vertex* v0 = t->vertices[0];
					Vertex* v1 = t->vertices[1];
					Vertex* v2 = t->vertices[2];
					Vertex* v3 = t->vertices[3];
					if (!v0 || !v1 || !v2 || !v3) continue;
					addFace(v1, v2, v3, v0);
					addFace(v0, v2, v3, v1);
					addFace(v0, v1, v3, v2);
					addFace(v0, v1, v2, v3);
				}
			}

			std::vector<char> isSurfacePhys(physRep.size(), 0);
			agentContactTriangles.reserve(faces.size());
			agentContactTrianglePhysIds.reserve(faces.size());
				for (const auto& kv : faces) {
					const FaceRec& rec = kv.second;
					if (rec.count != 1) continue; // interior face (shared by two tets in phys space)
					if (!rec.a || !rec.b || !rec.c) continue;
					agentContactTriangles.push_back(AgentTriangle{rec.a, rec.b, rec.c, rec.opp});
					agentContactTrianglePhysIds.push_back(rec.ids);
					for (int k = 0; k < 3; ++k) {
						const int id = rec.ids[k];
						if (id >= 0 && id < static_cast<int>(isSurfacePhys.size())) {
							isSurfacePhys[static_cast<size_t>(id)] = 1;
						}
					}
				}

				// Build triangle adjacency (edge-sharing neighbors) in physical-id space for fast, stable contact manifold
				// search (avoids per-frame O(Ntri) scans).
				agentContactTriangleNeighbors.assign(agentContactTriangles.size(), {});
				if (!agentContactTrianglePhysIds.empty()) {
					auto edgeKey = [](int a, int b) -> uint64_t {
						const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
						const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
						return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
					};

					std::unordered_map<uint64_t, std::vector<int>> edgeToTris;
					edgeToTris.reserve(agentContactTrianglePhysIds.size() * 3);
					for (int ti = 0; ti < static_cast<int>(agentContactTrianglePhysIds.size()); ++ti) {
						const auto& ids = agentContactTrianglePhysIds[static_cast<size_t>(ti)];
						const int i0 = ids[0];
						const int i1 = ids[1];
						const int i2 = ids[2];
						if (i0 < 0 || i1 < 0 || i2 < 0) continue;
						edgeToTris[edgeKey(i0, i1)].push_back(ti);
						edgeToTris[edgeKey(i1, i2)].push_back(ti);
						edgeToTris[edgeKey(i2, i0)].push_back(ti);
					}

					auto addUnique = [](std::vector<int>& v, int x) {
						for (int e : v) {
							if (e == x) return;
						}
						v.push_back(x);
					};

					for (const auto& kv : edgeToTris) {
						const auto& list = kv.second;
						if (list.size() < 2) continue;
						for (size_t i = 0; i < list.size(); ++i) {
							for (size_t j = i + 1; j < list.size(); ++j) {
								const int a = list[i];
								const int b = list[j];
								if (a < 0 || b < 0) continue;
								if (a >= static_cast<int>(agentContactTriangleNeighbors.size()) ||
									b >= static_cast<int>(agentContactTriangleNeighbors.size())) continue;
								addUnique(agentContactTriangleNeighbors[static_cast<size_t>(a)], b);
								addUnique(agentContactTriangleNeighbors[static_cast<size_t>(b)], a);
							}
						}
					}
				}

				if (agentUseSurfaceVertices) {
					for (int id = 0; id < static_cast<int>(isSurfacePhys.size()); ++id) {
						if (!isSurfacePhys[static_cast<size_t>(id)]) continue;
						if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
					Vertex* rep = physRep[static_cast<size_t>(id)];
					if (!rep) continue;
					agentContactVertices.push_back(rep);
					agentContactVertexPhysIds.push_back(id);
				}
			}
		}

		// Fallback: if we couldn't build a clean surface vertex set, use all physical vertices.
				if (agentContactVertices.empty()) {
					agentContactVertices = objectUniqueVertices;
					agentContactVertexPhysIds.clear();
					agentContactVertexPhysIds.reserve(physRep.size());
					for (int id = 0; id < static_cast<int>(physRep.size()); ++id) {
						agentContactVertexPhysIds.push_back(id);
					}
				}

				// Seed per-finger preferred contact triangle (temporal coherence). This avoids a slow global
				// search on the first few frames after enabling the agent spheres.
				if (!agentContactTriangles.empty()) {
					for (int fi = 0; fi < kFingerCount; ++fi) {
						const Eigen::Vector3f p = agentProxyPositions[static_cast<size_t>(fi)];
						int bestTi = -1;
						float bestD2 = std::numeric_limits<float>::infinity();
						for (int ti = 0; ti < static_cast<int>(agentContactTriangles.size()); ++ti) {
							const auto& tri = agentContactTriangles[static_cast<size_t>(ti)];
							if (!tri.a || !tri.b || !tri.c) continue;
							const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
							const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
							const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
							const Eigen::Vector3f q = closestPointOnTriangle(p, a, b, c, nullptr);
							const float d2 = (q - p).squaredNorm();
							if (d2 < bestD2) {
								bestD2 = d2;
								bestTi = ti;
							}
						}
						if (bestTi >= 0) {
							agentLastActiveContactTriangle[static_cast<size_t>(fi)] = bestTi;
						}
					}
				}

				// Precompute per-physical-vertex mass (sum of duplicated group vertices) for stable constraints.
				std::vector<float> physMassSumKg;
				physMassSumKg.resize(agentVerticesByPhysId.size(), 0.0f);
			for (size_t id = 0; id < agentVerticesByPhysId.size(); ++id) {
				float m = 0.0f;
				for (Vertex* v : agentVerticesByPhysId[id]) {
					if (!v) continue;
					m += std::max(0.0f, v->vertexMass);
				}
				physMassSumKg[id] = m;
			}

			// Optional volumetric stabilization: keep tet volumes near rest to avoid "hollow" collapse.
			struct TetVolumeRec {
				std::array<int, 4> ids{ {-1, -1, -1, -1} };
				float restAbsVolume = 0.0f;
				float restSign = 1.0f; // +1 or -1
			};
			std::vector<TetVolumeRec> tetVolumeRecs;
			tetVolumeRecs.reserve(4096);
			for (int gi = 0; gi < object.groupNum; ++gi) {
				Group& g = object.groups[gi];
				for (Tetrahedron* t : g.tetrahedra) {
					if (!t) continue;
					if (!t->vertices[0] || !t->vertices[1] || !t->vertices[2] || !t->vertices[3]) continue;

					auto physId = [&](Vertex* v) -> int {
						if (!v) return -1;
						const auto it = physKeyToId.find(makePhysKey(v));
						return (it == physKeyToId.end()) ? -1 : it->second;
					};

					const int id0 = physId(t->vertices[0]);
					const int id1 = physId(t->vertices[1]);
					const int id2 = physId(t->vertices[2]);
					const int id3 = physId(t->vertices[3]);
					if (id0 < 0 || id1 < 0 || id2 < 0 || id3 < 0) continue;

					const Eigen::Vector3f p0(t->vertices[0]->initx, t->vertices[0]->inity, t->vertices[0]->initz);
					const Eigen::Vector3f p1(t->vertices[1]->initx, t->vertices[1]->inity, t->vertices[1]->initz);
					const Eigen::Vector3f p2(t->vertices[2]->initx, t->vertices[2]->inity, t->vertices[2]->initz);
					const Eigen::Vector3f p3(t->vertices[3]->initx, t->vertices[3]->inity, t->vertices[3]->initz);
					const float v0 = signedTetraVolume(p0, p1, p2, p3);
					const float av0 = std::abs(v0);
					if (!(av0 > 1e-12f)) continue;

					TetVolumeRec rec;
					rec.ids = { id0, id1, id2, id3 };
					rec.restAbsVolume = av0;
					rec.restSign = (v0 >= 0.0f) ? 1.0f : -1.0f;
					tetVolumeRecs.push_back(rec);
				}
			}
			std::vector<Eigen::Vector3f> tetVolumePhysDp(agentVerticesByPhysId.size(), Eigen::Vector3f::Zero());

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

	#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
				const double nowSec = glfwGetTime();
				if (leapUseInput) {
					leapTracker.poll(nowSec);
					std::array<Eigen::Vector3f, kFingerCount> tipsMm;
					double timeSec = -1.0;
					if (leapTracker.getRightHandTipsMm(&tipsMm, nullptr, &timeSec)) {
						leapLatestTipsMm = tipsMm;
						leapLatestTimeSec = timeSec;
					}
				}
	#endif

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
			#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
						static KeyLatch leapToggleLatch;
						static KeyLatch leapRecenterLatch;
						static KeyLatch leapGainDownLatch;
					static KeyLatch leapGainUpLatch;
	#endif

						if (agentToggleLatch.consume(window, GLFW_KEY_H)) {
							agentSphere.enabled = !agentSphere.enabled;
								std::cout << "[AgentSphere] " << (agentSphere.enabled ? "ENABLED" : "DISABLED") << "\n"
								          << "  Move: I/K (Y), J/L (X), U/O (Z) | Hold SHIFT for faster\n"
							          << "  VirtualCoupling: V | Home: T | Print force: G | Print COM: M\n"
			#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
								          << "  Leap: toggle=B | recenter=R | gain: '['/']'\n"
			#endif
							          << "  Contact: triangles=" << agentContactTriangles.size()
							          << " vertices=" << agentContactVertices.size()
							          << " fingers=" << kFingerCount << "\n";
						}
			static bool agentUseVC = agentVirtualCoupling;
				if (agentVcLatch.consume(window, GLFW_KEY_V)) {
					agentUseVC = !agentUseVC;
					std::cout << "[AgentSphere] VirtualCoupling " << (agentUseVC ? "ON" : "OFF") << "\n";
				}
						if (agentHomeLatch.consume(window, GLFW_KEY_T)) {
							for (int fi = 0; fi < kFingerCount; ++fi) {
								const Eigen::Vector3f p = agentHomePositions[static_cast<size_t>(fi)];
								agentDevicePositions[static_cast<size_t>(fi)] = p;
						agentDevicePrevPositions[static_cast<size_t>(fi)] = p;
						agentDeviceVelocities[static_cast<size_t>(fi)].setZero();
						agentProxyPositions[static_cast<size_t>(fi)] = p;
						agentProxyVelocities[static_cast<size_t>(fi)].setZero();
						agentLastDeviceForcesN[static_cast<size_t>(fi)].setZero();
						agentLastContactForcesN[static_cast<size_t>(fi)].setZero();
						agentLastCouplingForcesN[static_cast<size_t>(fi)].setZero();
						agentLastContactCounts[static_cast<size_t>(fi)] = 0;
					}
	#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
						leapMappingCalibrated = false;
						leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)];
	#endif
					}

	#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
				if (leapToggleLatch.consume(window, GLFW_KEY_B)) {
					leapUseInput = !leapUseInput;
					if (leapUseInput && !leapTracker.init()) {
						std::cerr << "[LeapC] init failed; Leap input remains disabled.\n";
						leapUseInput = false;
					}

						leapMappingCalibrated = false;
						leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)];
						std::cout << "[LeapC] Input " << (leapUseInput ? "ON" : "OFF")
					          << " | workspace(mm)=(" << leapWorkspaceXmm << "," << leapWorkspaceYmm << "," << leapWorkspaceZmm << ")"
					          << " worldMargin=" << leapWorldMargin
					          << " gain=" << leapGain
					          << " yOffset=" << leapYOffsetBboxFrac
					          << " spread=" << leapFingerSpreadGain
					          << " smoothing=" << leapSmoothingTime
					          << " flip=(" << (leapFlipX ? 1 : 0) << "," << (leapFlipY ? 1 : 0) << "," << (leapFlipZ ? 1 : 0) << ")"
					          << "\n";
				}
					if (leapRecenterLatch.consume(window, GLFW_KEY_R)) {
						leapMappingCalibrated = false;
						leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)];
						std::cout << "[LeapC] Recenter requested.\n";
					}
				if (leapGainDownLatch.consume(window, GLFW_KEY_LEFT_BRACKET)) {
					leapGain = std::max(0.0f, leapGain / 1.1f);
					std::cout << "[LeapC] gain=" << leapGain << "\n";
				}
				if (leapGainUpLatch.consume(window, GLFW_KEY_RIGHT_BRACKET)) {
					leapGain = std::max(0.0f, leapGain * 1.1f);
					std::cout << "[LeapC] gain=" << leapGain << "\n";
				}
	#endif

				// Agent device motion source: Leap (if enabled) else keyboard keys.
					{
						bool usedLeap = false;
	#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
						if (leapUseInput) {
							const double now = glfwGetTime();
							const double staleSec = 0.2;
							if (leapLatestTimeSec >= 0.0 && (now - leapLatestTimeSec) <= staleSec) {
								const Eigen::Vector3f extents = bboxMax - bboxMin;
								const float m = std::max(0.0f, leapWorldMargin);
								const Eigen::Vector3f worldRange = extents * (1.0f + 2.0f * m);

							const Eigen::Vector3f workspaceMm(
								std::max(1.0f, leapWorkspaceXmm),
								std::max(1.0f, leapWorkspaceYmm),
								std::max(1.0f, leapWorkspaceZmm));

								const float gain = std::max(0.0f, leapGain);
								const Eigen::Vector3f scale = gain * Eigen::Vector3f(
									(worldRange.x() > 1e-8f) ? (worldRange.x() / workspaceMm.x()) : (bboxDiag / workspaceMm.x()),
									(worldRange.y() > 1e-8f) ? (worldRange.y() / workspaceMm.y()) : (bboxDiag / workspaceMm.y()),
									(worldRange.z() > 1e-8f) ? (worldRange.z() / workspaceMm.z()) : (bboxDiag / workspaceMm.z()));

									const Eigen::Vector3f axisSign(
										leapFlipX ? -1.0f : 1.0f,
										leapFlipY ? -1.0f : 1.0f,
										leapFlipZ ? -1.0f : 1.0f);

									const float smooth = std::max(0.0f, leapSmoothingTime);
									const float alpha = (smooth > 1e-6f) ? (1.0f - std::exp(-timeStep / smooth)) : 1.0f;
									const float yOffset = leapYOffsetBboxFrac * extents.y();
									const Eigen::Vector3f margin = extents * m;

									const float spreadGain = std::max(0.0f, leapFingerSpreadGain);
									// Finger spread is mostly across X/Z; don't exaggerate vertical offsets by default.
									const Eigen::Vector3f spreadScale(spreadGain, 1.0f, spreadGain);

									const Eigen::Vector3f indexTipMm =
										leapLatestTipsMm[static_cast<size_t>(kIndexFinger)].cwiseProduct(axisSign);
									if (!leapMappingCalibrated) {
										// Keep the original 1-finger behavior: recenter on INDEX fingertip.
										leapCenterMm = indexTipMm;
										leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)];

										Eigen::Vector3f clampMin = bboxMin - margin;
										Eigen::Vector3f clampMax = bboxMax + margin;
										clampMin = clampMin.cwiseMin(leapAnchorWorld - margin);
										clampMax = clampMax.cwiseMax(leapAnchorWorld + margin);

										// Clamp only the hand base (index) to avoid all fingertips collapsing onto a clamp plane.
										Eigen::Vector3f base = leapAnchorWorld;
										base.y() += yOffset;
										base = base.cwiseMax(clampMin).cwiseMin(clampMax);

										const float maxRel = 0.5f * bboxDiag;
										const Eigen::Vector3f relClamp(maxRel, maxRel, maxRel);

										// Snap all fingers once so the hand appears immediately.
										for (int fi = 0; fi < kFingerCount; ++fi) {
											const Eigen::Vector3f tipMm = leapLatestTipsMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
											const Eigen::Vector3f relMm = tipMm - indexTipMm;
											Eigen::Vector3f relWorld = relMm.cwiseProduct(scale.cwiseProduct(spreadScale));
											relWorld = relWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
											const Eigen::Vector3f target = base + relWorld;

											agentDevicePositions[static_cast<size_t>(fi)] = target;
											agentDevicePrevPositions[static_cast<size_t>(fi)] = target;
											agentDeviceVelocities[static_cast<size_t>(fi)].setZero();
										}
										leapMappingCalibrated = true;
									} else {
										// Clamp to an expanded bbox to avoid "flying away".
										Eigen::Vector3f clampMin = bboxMin - margin;
										Eigen::Vector3f clampMax = bboxMax + margin;
										clampMin = clampMin.cwiseMin(leapAnchorWorld - margin);
										clampMax = clampMax.cwiseMax(leapAnchorWorld + margin);

										// Clamp only the hand base (index) to avoid all fingertips collapsing onto a clamp plane.
										Eigen::Vector3f base = leapAnchorWorld + (indexTipMm - leapCenterMm).cwiseProduct(scale);
										base.y() += yOffset;
										base = base.cwiseMax(clampMin).cwiseMin(clampMax);

										const float maxRel = 0.5f * bboxDiag;
										const Eigen::Vector3f relClamp(maxRel, maxRel, maxRel);

										for (int fi = 0; fi < kFingerCount; ++fi) {
											const Eigen::Vector3f tipMm = leapLatestTipsMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
											const Eigen::Vector3f relMm = tipMm - indexTipMm;
											Eigen::Vector3f relWorld = relMm.cwiseProduct(scale.cwiseProduct(spreadScale));
											relWorld = relWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
											const Eigen::Vector3f target = base + relWorld;

											auto& p = agentDevicePositions[static_cast<size_t>(fi)];
											auto& pPrev = agentDevicePrevPositions[static_cast<size_t>(fi)];
											auto& v = agentDeviceVelocities[static_cast<size_t>(fi)];

											p = p + alpha * (target - p);
											v = (p - pPrev) / std::max(1e-8f, timeStep);
											pPrev = p;
										}
									}
									usedLeap = true;
								} else {
									// If tracking is stale, let keyboard drive and re-calibrate on resume.
									leapMappingCalibrated = false;
									leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)];
								}
						}
	#endif
						if (!usedLeap) {
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
							for (int fi = 0; fi < kFingerCount; ++fi) {
								auto& p = agentDevicePositions[static_cast<size_t>(fi)];
								auto& pPrev = agentDevicePrevPositions[static_cast<size_t>(fi)];
								auto& v = agentDeviceVelocities[static_cast<size_t>(fi)];
								p += delta;
								v = (p - pPrev) / std::max(1e-8f, timeStep);
								pPrev = p;
							}
						}
				}
					if (agentPrintLatch.consume(window, GLFW_KEY_G)) {
						std::cout << "[AgentSphere] vc=" << (agentUseVC ? "1" : "0") << " fingers=" << kFingerCount << "\n";
						for (int fi = 0; fi < kFingerCount; ++fi) {
					const Eigen::Vector3f& devPos = agentDevicePositions[static_cast<size_t>(fi)];
					const Eigen::Vector3f& proxyPos = agentProxyPositions[static_cast<size_t>(fi)];
					const Eigen::Vector3f& devForceN = agentLastDeviceForcesN[static_cast<size_t>(fi)];
					const Eigen::Vector3f& contactForceN = agentLastContactForcesN[static_cast<size_t>(fi)];
					const int contacts = agentLastContactCounts[static_cast<size_t>(fi)];

					float proxySurfDist = -1.0f;
					float proxySurfSN = 0.0f;
					float proxyPen = 0.0f;
					if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
						const AgentSurfaceQueryResult q = queryAgentSurface(proxyPos, agentContactTriangles);
						if (q.found && q.outwardNormal.squaredNorm() > 0.0f) {
							proxySurfDist = q.distanceToSurface;
							proxySurfSN = q.outwardNormal.dot(proxyPos - q.closestPoint);
							proxyPen = std::max(0.0f, agentSphere.radius - q.distanceToSurface);
						}
					}

					std::cout << "  " << kFingerNames[static_cast<size_t>(fi)]
					          << " dev=(" << devPos.x() << "," << devPos.y() << "," << devPos.z() << ")"
					          << " proxy=(" << proxyPos.x() << "," << proxyPos.y() << "," << proxyPos.z() << ")"
					          << " devF=(" << devForceN.x() << "," << devForceN.y() << "," << devForceN.z() << ")"
					          << " contactF=(" << contactForceN.x() << "," << contactForceN.y() << "," << contactForceN.z() << ")"
					          << " cnt=" << contacts
					          << " surfDist=" << proxySurfDist
					          << " surfSN=" << proxySurfSN
						          << " pen=" << proxyPen
						          << "\n";
						}

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
					if (leapUseInput) {
						std::cout << "  [LeapC] tipMm (thumb..pinky)\n";
						for (int fi = 0; fi < kFingerCount; ++fi) {
							const Eigen::Vector3f& t = leapLatestTipsMm[static_cast<size_t>(fi)];
							std::cout << "    " << kFingerNames[static_cast<size_t>(fi)]
							          << " mm=(" << t.x() << "," << t.y() << "," << t.z() << ")\n";
						}
						const auto dist = [&](int a, int b) -> float {
							return (leapLatestTipsMm[static_cast<size_t>(a)] - leapLatestTipsMm[static_cast<size_t>(b)]).norm();
						};
						std::cout << "  [LeapC] tipDistMm TI " << dist(0, 1)
						          << " IM " << dist(1, 2)
						          << " MR " << dist(2, 3)
						          << " RP " << dist(3, 4)
						          << " TP " << dist(0, 4) << "\n";
						}
#endif
					}
					static KeyLatch comPrintLatch;
					if (comPrintLatch.consume(window, GLFW_KEY_M)) {
						Eigen::Vector3f com = Eigen::Vector3f::Zero();
						float totalMass = 0.0f;
						int fixedCount = 0;
						for (Vertex* v : objectUniqueVertices) {
							if (!v) continue;
							const float m = std::max(0.0f, v->vertexMass);
							totalMass += m;
							com += m * Eigen::Vector3f(v->x, v->y, v->z);
							if (v->isFixed) ++fixedCount;
						}
						if (totalMass > 1e-12f) com /= totalMass;
						std::cout << "[COM] (" << com.x() << "," << com.y() << "," << com.z() << ")"
						          << " fixed=" << fixedCount << "/" << objectUniqueVertices.size()
						          << " anchor_mode=" << anchorMode
						          << " wall=" << (wallEnabled ? 1 : 0)
						          << " tetVol=" << (tetVolumeConstraintEnabled ? 1 : 0)
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
				for (int gi = 0; gi < object.groupNum; ++gi) {
					for (const auto& pair : object.groups[gi].verticesMap) {
						Vertex* v = pair.second;
						if (v && v->index > maxV) maxV = v->index;
					}
				}
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

			// Optional: soft spring anchor (applied as per-vertex acceleration via dragForces).
			// This keeps the organ from drifting away while still allowing global translation/rotation.
			if (anchorModeClamped == 2 && anchorRegionValid && !anchorPhysIds.empty()) {
				const float k = std::max(0.0f, anchorSpringK);
				const float c = std::max(0.0f, anchorSpringDamping);
				const float maxA = anchorSpringMaxAccel;
				const float invR = (anchorRegionRadius > 1e-8f) ? (1.0f / anchorRegionRadius) : 0.0f;

				for (int id : anchorPhysIds) {
					if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
					const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
					if (list.empty()) continue;

					Vertex* rep = (id >= 0 && id < static_cast<int>(physRep.size())) ? physRep[static_cast<size_t>(id)] : nullptr;
					if (!rep) continue;
					const Eigen::Vector3f rest(rep->initx, rep->inity, rep->initz);

					Eigen::Vector3f pAvg = Eigen::Vector3f::Zero();
					Eigen::Vector3f vAvg = Eigen::Vector3f::Zero();
					int n = 0;
					for (Vertex* v : list) {
						if (!v) continue;
						if (v->isFixed) {
							n = 0;
							break;
						}
						pAvg += Eigen::Vector3f(v->x, v->y, v->z);
						vAvg += Eigen::Vector3f(v->velx, v->vely, v->velz);
						++n;
					}
					if (n <= 0) continue;
					pAvg /= static_cast<float>(n);
					vAvg /= static_cast<float>(n);

					Eigen::Vector3f accel = k * (rest - pAvg) - c * vAvg;
					if (invR > 0.0f) {
						const float t = std::min(1.0f, (pAvg - anchorCenterRest).norm() * invR);
						float w = 1.0f - t;
						w = w * w; // smooth falloff near boundary
						accel *= w;
					}

					const float aLen = accel.norm();
					if (maxA > 0.0f && aLen > maxA) accel *= (maxA / aLen);
					if (accel.squaredNorm() <= 1e-12f) continue;

					for (Vertex* v : list) {
						if (!v || v->isFixed) continue;
						const int idx = v->index;
						if (idx >= 0 && idx < static_cast<int>(dragForces.size())) {
							dragForces[static_cast<size_t>(idx)] += accel;
						}
					}
				}
				}

				// Remember previous proxy positions (used for collision substepping in direct/kinematic mode).
				std::array<Eigen::Vector3f, kFingerCount> agentProxyStartPositions = agentProxyPositions;

						// Update agent (finger) kinematics after any drag/experiment force contributions.
						// Tissue deformation + reaction force are solved as positional collision constraints after the PBD step.
						if (agentSphere.enabled) {
						if (agentUseVC) {
						// Virtual coupling: device drives a target; proxy is a smoothed sphere used for collision.
						const float maxVcDist = std::max(0.0f, agentVcMaxDistanceRadiusFrac) * agentSphere.radius;
						const float maxStep = std::max(1e-6f, 0.25f * agentSphere.radius);

						for (int fi = 0; fi < kFingerCount; ++fi) {
							const Eigen::Vector3f& devPos = agentDevicePositions[static_cast<size_t>(fi)];
							const Eigen::Vector3f& devVel = agentDeviceVelocities[static_cast<size_t>(fi)];
							Eigen::Vector3f& proxyPos = agentProxyPositions[static_cast<size_t>(fi)];
							Eigen::Vector3f& proxyVel = agentProxyVelocities[static_cast<size_t>(fi)];

							const float deviceStep = devVel.norm() * timeStep;
							const int substepsFromSpeed = static_cast<int>(std::ceil(deviceStep / maxStep));
							const float deviceGap = (devPos - proxyPos).norm();
							const int substepsFromGap = static_cast<int>(std::ceil(deviceGap / maxStep));
							const int baseSubsteps = std::max(agentVcSubsteps, std::max(substepsFromSpeed, substepsFromGap));
							const int substeps = std::clamp(baseSubsteps, 1, 64);
							const float dtSub = timeStep / static_cast<float>(substeps);

								Eigen::Vector3f lastCouplingForceN = Eigen::Vector3f::Zero();
								const bool wasContact = (agentLastContactCounts[static_cast<size_t>(fi)] > 0);
								const float vcCLen = wasContact ? agentVcCLenContact : agentVcCLenFree;
								
								// Adaptive VC stiffness based on previous frame penetration
								float vcStiffnessScale = 1.0f;
								if (wasContact) {
									const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
									const float penFrac = lastPen / std::max(1e-6f, agentSphere.radius);
									if (penFrac > 0.15f) {
										const float t = std::clamp((penFrac - 0.15f) / 0.25f, 0.0f, 1.0f);
										vcStiffnessScale = 1.0f - 0.95f * t;
									}
								}
								
								for (int si = 0; si < substeps; ++si) {
									Eigen::Vector3f disp = (devPos - proxyPos);
									const float dispLen = disp.norm();
									if (maxVcDist > 1e-6f && dispLen > maxVcDist) {
										disp *= (maxVcDist / dispLen);
									}
									const Eigen::Vector3f couplingForceN =
										(agentVcKLen * vcStiffnessScale) * disp +
										vcCLen * (devVel - proxyVel);

								const Eigen::Vector3f proxyAcc = couplingForceN / agentProxyMassKg;
								proxyVel += proxyAcc * dtSub;
								proxyPos += proxyVel * dtSub;

								lastCouplingForceN = couplingForceN;
							}

								agentLastCouplingForcesN[static_cast<size_t>(fi)] = lastCouplingForceN;
								// For a real haptic device this is the force you'd output.
								//
								// IMPORTANT: Do NOT feed proxyVel into the output damping term. proxyVel picks up
								// high-frequency noise from contact corrections and will show up as "buzzing" even
								// when the device is held still. Keep full damping for proxy dynamics, but output
								// only spring + device-velocity damping.
								Eigen::Vector3f dispOut = (devPos - proxyPos);
								const float dispLenOut = dispOut.norm();
								if (maxVcDist > 1e-6f && dispLenOut > maxVcDist) {
									dispOut *= (maxVcDist / dispLenOut);
								}
								const Eigen::Vector3f springForceN = (agentVcKLen * vcStiffnessScale) * dispOut;
								const Eigen::Vector3f deviceForceN = -(springForceN + vcCLen * devVel);
								agentLastDeviceForcesN[static_cast<size_t>(fi)] = deviceForceN;
						}
					} else {
						// Direct: kinematic sphere.
						for (int fi = 0; fi < kFingerCount; ++fi) {
							agentProxyPositions[static_cast<size_t>(fi)] = agentDevicePositions[static_cast<size_t>(fi)];
							agentProxyVelocities[static_cast<size_t>(fi)] = agentDeviceVelocities[static_cast<size_t>(fi)];
							agentLastCouplingForcesN[static_cast<size_t>(fi)].setZero();
							// Filled after the collision constraint solve.
							agentLastDeviceForcesN[static_cast<size_t>(fi)].setZero();
						}
					}
	
							} else {
							for (int fi = 0; fi < kFingerCount; ++fi) {
								agentLastDeviceForcesN[static_cast<size_t>(fi)].setZero();
							agentLastContactForcesN[static_cast<size_t>(fi)].setZero();
						agentLastCouplingForcesN[static_cast<size_t>(fi)].setZero();
						agentLastContactCounts[static_cast<size_t>(fi)] = 0;
					}
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

				auto applyTetVolumeStabilization = [&](int iterationsReq, float correctionReq) -> bool {
					if (!tetVolumeConstraintEnabled) return false;
					if (tetVolumeRecs.empty() || agentVerticesByPhysId.empty() || physMassSumKg.empty()) return false;

					const int iters = std::clamp(iterationsReq, 1, 8);
					const float corr = std::clamp(correctionReq, 0.0f, 1.0f);
					if (!(corr > 0.0f)) return false;

					const float invDt = 1.0f / std::max(1e-8f, timeStep);
					const float maxDp = 0.0025f * bboxDiag; // conservative safety clamp

					auto repOf = [&](int id) -> Vertex* {
						if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) return nullptr;
						const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
						return list.empty() ? nullptr : list.front();
					};
					auto invMassOf = [&](int id, Vertex* rep) -> float {
						if (!rep || rep->isFixed) return 0.0f;
						if (id < 0 || id >= static_cast<int>(physMassSumKg.size())) return 0.0f;
						const float m = physMassSumKg[static_cast<size_t>(id)];
						return (m > 1e-12f) ? (1.0f / m) : 0.0f;
					};

					bool anyVolumeCorrection = false;
					for (int it = 0; it < iters; ++it) {
						std::fill(tetVolumePhysDp.begin(), tetVolumePhysDp.end(), Eigen::Vector3f::Zero());

						for (const TetVolumeRec& rec : tetVolumeRecs) {
							const int id0 = rec.ids[0];
							const int id1 = rec.ids[1];
							const int id2 = rec.ids[2];
							const int id3 = rec.ids[3];

							Vertex* v0 = repOf(id0);
							Vertex* v1 = repOf(id1);
							Vertex* v2 = repOf(id2);
							Vertex* v3 = repOf(id3);
							if (!v0 || !v1 || !v2 || !v3) continue;

							const Eigen::Vector3f p0(v0->x, v0->y, v0->z);
							const Eigen::Vector3f p1(v1->x, v1->y, v1->z);
							const Eigen::Vector3f p2(v2->x, v2->y, v2->z);
							const Eigen::Vector3f p3(v3->x, v3->y, v3->z);

							const float V = signedTetraVolume(p0, p1, p2, p3);
							const float Vproj = rec.restSign * V;
							const float C = Vproj - rec.restAbsVolume;
							if (std::abs(C) <= std::max(1e-10f, 1e-6f * rec.restAbsVolume)) continue;

							const Eigen::Vector3f a = p1 - p0;
							const Eigen::Vector3f b = p2 - p0;
							const Eigen::Vector3f c = p3 - p0;

							Eigen::Vector3f g1 = b.cross(c) / 6.0f;
							Eigen::Vector3f g2 = c.cross(a) / 6.0f;
							Eigen::Vector3f g3 = a.cross(b) / 6.0f;
							Eigen::Vector3f g0 = -g1 - g2 - g3;
							g0 *= rec.restSign;
							g1 *= rec.restSign;
							g2 *= rec.restSign;
							g3 *= rec.restSign;

							const float w0 = invMassOf(id0, v0);
							const float w1 = invMassOf(id1, v1);
							const float w2 = invMassOf(id2, v2);
							const float w3 = invMassOf(id3, v3);
							const float denom =
								w0 * g0.squaredNorm() +
								w1 * g1.squaredNorm() +
								w2 * g2.squaredNorm() +
								w3 * g3.squaredNorm();
							if (!(denom > 1e-18f)) continue;

							const float s = (-C / denom) * corr;
							if (w0 > 0.0f) tetVolumePhysDp[static_cast<size_t>(id0)] += (w0 * s) * g0;
							if (w1 > 0.0f) tetVolumePhysDp[static_cast<size_t>(id1)] += (w1 * s) * g1;
							if (w2 > 0.0f) tetVolumePhysDp[static_cast<size_t>(id2)] += (w2 * s) * g2;
							if (w3 > 0.0f) tetVolumePhysDp[static_cast<size_t>(id3)] += (w3 * s) * g3;
						}

						int moved = 0;
						for (size_t id = 0; id < tetVolumePhysDp.size(); ++id) {
							Eigen::Vector3f dp = tetVolumePhysDp[id];
							const float len = dp.norm();
							if (len <= 1e-12f) continue;
							if (maxDp > 1e-8f && len > maxDp) dp *= (maxDp / len);

							const auto& list = agentVerticesByPhysId[id];
							for (Vertex* v : list) {
								if (!v || v->isFixed) continue;
								v->x += dp.x();
								v->y += dp.y();
								v->z += dp.z();
								v->velx += dp.x() * invDt;
								v->vely += dp.y() * invDt;
								v->velz += dp.z() * invDt;
							}
							++moved;
						}

						anyVolumeCorrection = anyVolumeCorrection || (moved > 0);
					}

					// Keep Group::groupVelocity consistent with Vertex::vel* for the next frame's primeVec.
					if (anyVolumeCorrection) {
						for (int gi = 0; gi < object.groupNum; ++gi) {
							Group& group = object.groups[gi];
							for (const auto& vertexPair : group.verticesMap) {
								Vertex* v = vertexPair.second;
								if (!v) continue;
								const int li = v->localIndex;
								if (li < 0 || (3 * li + 2) >= group.groupVelocity.size()) continue;
								if (v->isFixed) {
									group.groupVelocity.segment<3>(3 * li) = Eigen::Vector3f::Zero();
								} else {
									group.groupVelocity.segment<3>(3 * li) = Eigen::Vector3f(v->velx, v->vely, v->velz);
								}
							}
						}
					}

					return anyVolumeCorrection;
				};

				// Optional: volumetric stabilization (pre-contact). Helpful for global stability; deep-press
				// stability is additionally handled with a post-contact pass below.
				if (tetVolumeConstraintEnabled) {
					applyTetVolumeStabilization(tetVolumeConstraintIterations, tetVolumeConstraintCorrection);
				}

				// Conservative CCD: prevent surface vertices from tunneling through the proxy sphere between frames.
				int agentCcdHits = 0;
				if (agentSphere.enabled && !physPrevPositions.empty()) {
					const float rCcd = std::max(1e-6f, agentSphere.radius);
					const float r2Ccd = rCcd * rCcd;
					const float slop = std::max(1e-4f * bboxDiag, 0.01f * rCcd);
					const float invDt = 1.0f / std::max(1e-8f, timeStep);

					for (size_t id = 0; id < agentVerticesByPhysId.size(); ++id) {
						const auto& list = agentVerticesByPhysId[id];
						if (list.empty() || !list.front()) continue;
						Vertex* vRef = list.front();
						if (!vRef || vRef->isFixed) continue;

						const Eigen::Vector3f p0 = physPrevPositions[id];
						Eigen::Vector3f p1(vRef->x, vRef->y, vRef->z);
						const Eigen::Vector3f dpWorld = p1 - p0;
						if (dpWorld.squaredNorm() <= 1e-18f) continue;

						for (int fi = 0; fi < kFingerCount; ++fi) {
							const Eigen::Vector3f c0 = agentProxyStartPositions[static_cast<size_t>(fi)];
							const Eigen::Vector3f c1 = agentProxyPositions[static_cast<size_t>(fi)];

							const Eigen::Vector3f rel0 = p0 - c0;
							const Eigen::Vector3f rel1 = p1 - c1;
							if (rel0.squaredNorm() <= r2Ccd || rel1.squaredNorm() <= r2Ccd) continue;

							float tHit = 0.0f;
							if (!segmentSphereFirstHitT(rel0, rel1, rCcd, &tHit)) continue;

							const Eigen::Vector3f vHit = p0 + tHit * (p1 - p0);
							const Eigen::Vector3f cHit = c0 + tHit * (c1 - c0);
							Eigen::Vector3f n = vHit - cHit;
							const float nlen = n.norm();
							if (nlen <= 1e-8f) n = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
							else n /= nlen;

							const Eigen::Vector3f corrected = cHit + n * (rCcd + slop);
							const Eigen::Vector3f dp = corrected - p1;
							if (dp.squaredNorm() <= 1e-24f) break;

							for (Vertex* v : list) {
								if (!v || v->isFixed) continue;
								v->x += dp.x();
								v->y += dp.y();
								v->z += dp.z();
								v->velx += dp.x() * invDt;
								v->vely += dp.y() * invDt;
								v->velz += dp.z() * invDt;
							}
							p1 += dp;
							++agentCcdHits;
							break;
						}
					}
				}

							// Hard non-penetration against the agent sphere (prevents the tissue from "ghosting" through).
							// Use surface TRIANGLES (not all vertices) to avoid non-physical "fat contact" and jitter.
							if (agentSphere.enabled) {
								const bool useVC = agentUseVC;
								const float r = std::max(1e-6f, agentSphere.radius);
							const float maxPenFrac = std::clamp(agentMaxPenetrationFrac, 0.0f, 0.95f);
							const float allowedPen = maxPenFrac * r;
							// Small "contact shell" thickness improves stability and provides a basis for friction even
							// when the proxy is only lightly touching (PBD-style).
								const float eps = std::max(1e-4f * bboxDiag, 0.02f * r);
								const float corr = std::clamp(agentProxyPositionCorrection, 0.0f, 1.0f);
								const float tangentialDamp = std::clamp(agentCollisionTangentialDamp, 0.0f, 1.0f);
								const float contactProxyInvMassScale = std::clamp(agentContactProxyInvMassScale, 0.0f, 1.0f);
								const float contactVelRelax = std::clamp(agentContactVelocityRelaxation, 0.0f, 1.0f);
								const float contactVelRelaxMin = std::clamp(agentContactVelocityRelaxationMin, 0.0f, contactVelRelax);
								const float contactNormalDamp = std::clamp(agentContactNormalDamp, 0.0f, 1.0f);
								const int iters = std::clamp(agentCollisionIterations, 1, 64);
								const int manifoldK = std::clamp(agentContactManifoldTriangles, 1, 8);
								const float maxStep = std::max(1e-6f, 0.25f * r);
		
								bool anyContact = (agentCcdHits > 0);
								float maxPenetrationThisFrame = 0.0f;
								for (int fi = 0; fi < kFingerCount; ++fi) {
									AgentContactResult contact{};
									if (useVC) {
										Eigen::Vector3f& sphereCenter = agentProxyPositions[static_cast<size_t>(fi)];
										Eigen::Vector3f& sphereVel = agentProxyVelocities[static_cast<size_t>(fi)];
										const Eigen::Vector3f& devPos = agentDevicePositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f& devVel = agentDeviceVelocities[static_cast<size_t>(fi)];
										Eigen::Vector3f disp = devPos - sphereCenter;
										const float maxVcDist = std::max(0.0f, agentVcMaxDistanceRadiusFrac) * r;
										const float dispLen = disp.norm();
										if (maxVcDist > 1e-6f && dispLen > maxVcDist) {
											disp *= (maxVcDist / dispLen);
										}
											const bool wasContact = (agentLastContactCounts[static_cast<size_t>(fi)] > 0);
											const float vcCLen = wasContact ? agentVcCLenContact : agentVcCLenFree;
											
											// CRITICAL FIX: Adaptive VC stiffness based on penetration depth
											// When deeply penetrated, drastically reduce VC pulling force to prevent oscillation
											float vcStiffnessScale = 1.0f;
											if (wasContact) {
												const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
												const float penFrac = lastPen / std::max(1e-6f, r);
												// Start reducing stiffness at 15% penetration, down to 5% at 40% penetration
												if (penFrac > 0.15f) {
													const float t = std::clamp((penFrac - 0.15f) / 0.25f, 0.0f, 1.0f);
													vcStiffnessScale = 1.0f - 0.95f * t; // Reduce to 5% of original
												}
											}
											
											const Eigen::Vector3f driveForceN = (agentVcKLen * vcStiffnessScale) * disp + vcCLen * (devVel - sphereVel);
										const float sphereInvMass = 1.0f / agentProxyMassKg;
											if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
													contact = solveAgentSphereTriangleCollisionConstraint(
														sphereCenter,
														sphereVel,
														sphereInvMass,
														contactProxyInvMassScale,
														contactVelRelax,
														contactVelRelaxMin,
														contactNormalDamp,
														r,
														allowedPen,
														eps,
														timeStep,
														corr,
														tangentialDamp,
														agentFrictionMu,
														iters,
														manifoldK,
														&agentLastActiveContactTriangle[static_cast<size_t>(fi)],
														driveForceN,
														agentContactTriangles,
														agentContactTrianglePhysIds,
														agentContactTriangleNeighbors,
														agentVerticesByPhysId,
														physMassSumKg);
												} else if (!agentContactVertexPhysIds.empty()) {
											contact = solveAgentSphereVertexCollisionConstraint(
												sphereCenter,
												sphereVel,
												sphereInvMass,
												contactProxyInvMassScale,
												contactVelRelax,
												contactVelRelaxMin,
												contactNormalDamp,
												r,
												allowedPen,
												eps,
												timeStep,
												corr,
												tangentialDamp,
												agentFrictionMu,
												iters,
												driveForceN,
												agentContactVertexPhysIds,
												agentVerticesByPhysId,
												physMassSumKg);
										}
									} else {
										// Kinematic drive: substep along the proxy motion to avoid tunneling when the user moves fast.
										const Eigen::Vector3f p0 = agentProxyStartPositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f p1 = agentProxyPositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f dp = p1 - p0;
										const float stepLen = dp.norm();
										const int substeps = std::clamp(static_cast<int>(std::ceil(stepLen / maxStep)), 1, 64);
										const float dtSub = timeStep / static_cast<float>(substeps);
										const int itersSub = std::clamp(std::max(1, static_cast<int>(std::ceil(static_cast<float>(iters) / static_cast<float>(substeps)))), 1, 64);

										Eigen::Vector3f impulseNsec = Eigen::Vector3f::Zero();
										int contactVerts = 0;
										float maxPen = 0.0f;
										Eigen::Vector3f sumN = Eigen::Vector3f::Zero();

										Eigen::Vector3f prevP = p0;
										for (int si = 0; si < substeps; ++si) {
											const float a = static_cast<float>(si + 1) / static_cast<float>(substeps);
											Eigen::Vector3f p = p0 + dp * a;
											Eigen::Vector3f v = (p - prevP) / std::max(1e-8f, dtSub);

											AgentContactResult c{};
												if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
														c = solveAgentSphereTriangleCollisionConstraint(
															p,
															v,
															0.0f,
															contactProxyInvMassScale,
															contactVelRelax,
															contactVelRelaxMin,
															contactNormalDamp,
															r,
															allowedPen,
															eps,
															dtSub,
															corr,
															tangentialDamp,
															agentFrictionMu,
															itersSub,
															manifoldK,
															&agentLastActiveContactTriangle[static_cast<size_t>(fi)],
															Eigen::Vector3f::Zero(),
															agentContactTriangles,
															agentContactTrianglePhysIds,
															agentContactTriangleNeighbors,
															agentVerticesByPhysId,
															physMassSumKg);
													} else if (!agentContactVertexPhysIds.empty()) {
												c = solveAgentSphereVertexCollisionConstraint(
													p,
													v,
													0.0f,
													contactProxyInvMassScale,
													contactVelRelax,
													contactVelRelaxMin,
													contactNormalDamp,
													r,
													allowedPen,
													eps,
													dtSub,
													corr,
													tangentialDamp,
													agentFrictionMu,
													itersSub,
													Eigen::Vector3f::Zero(),
													agentContactVertexPhysIds,
													agentVerticesByPhysId,
													physMassSumKg);
											}

											impulseNsec += c.reactionForceN * dtSub;
											contactVerts += c.contactVertexCount;
											maxPen = std::max(maxPen, c.maxPenetration);
											sumN += c.avgNormal * std::max(0.0f, c.maxPenetration);

											prevP = p;
										}

										contact.reactionForceN = impulseNsec / std::max(1e-8f, timeStep);
										contact.contactVertexCount = contactVerts;
										contact.maxPenetration = maxPen;
										const float nlen = sumN.norm();
										if (nlen > 1e-12f) contact.avgNormal = sumN / nlen;
									}

									agentLastContactForcesN[static_cast<size_t>(fi)] = contact.reactionForceN;
									agentLastContactCounts[static_cast<size_t>(fi)] = contact.contactVertexCount;
									agentLastContactPenetrations[static_cast<size_t>(fi)] = contact.maxPenetration;
									if (!useVC) {
									agentLastDeviceForcesN[static_cast<size_t>(fi)] = contact.reactionForceN;
								}

								anyContact = anyContact || (contact.contactVertexCount > 0);
								maxPenetrationThisFrame = std::max(maxPenetrationThisFrame, contact.maxPenetration);
							}

								// Post-contact volumetric stabilization (deep press): prevents local collapse/inversion
								// which otherwise manifests as "penetrate / depenetrate" jitter at large indentations.
								// Apply once, then re-run a cheap contact pass to restore non-penetration.
								if (tetVolumeConstraintEnabled && anyContact && maxPenetrationThisFrame > 0.25f * r) {
									const bool didVolume = applyTetVolumeStabilization(
										tetVolumeConstraintIterations,
										0.5f * tetVolumeConstraintCorrection);
									if (didVolume && useVC) {
										const int iters2 = std::clamp(std::max(4, iters / 2), 1, 64);
										bool anyContact2 = false;
										float maxPen2 = 0.0f;
										for (int fi = 0; fi < kFingerCount; ++fi) {
											AgentContactResult contact2{};
											Eigen::Vector3f& sphereCenter = agentProxyPositions[static_cast<size_t>(fi)];
											Eigen::Vector3f& sphereVel = agentProxyVelocities[static_cast<size_t>(fi)];
											const Eigen::Vector3f& devPos = agentDevicePositions[static_cast<size_t>(fi)];
											const Eigen::Vector3f& devVel = agentDeviceVelocities[static_cast<size_t>(fi)];
											Eigen::Vector3f disp = devPos - sphereCenter;
											const float maxVcDist = std::max(0.0f, agentVcMaxDistanceRadiusFrac) * r;
											const float dispLen = disp.norm();
											if (maxVcDist > 1e-6f && dispLen > maxVcDist) {
												disp *= (maxVcDist / dispLen);
											}
											const bool wasContact = (agentLastContactCounts[static_cast<size_t>(fi)] > 0);
											const float vcCLen = wasContact ? agentVcCLenContact : agentVcCLenFree;
											
											// Adaptive VC stiffness (same as main pass)
											float vcStiffnessScale = 1.0f;
											if (wasContact) {
												const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
												const float penFrac = lastPen / std::max(1e-6f, r);
												if (penFrac > 0.15f) {
													const float t = std::clamp((penFrac - 0.15f) / 0.25f, 0.0f, 1.0f);
													vcStiffnessScale = 1.0f - 0.95f * t;
												}
											}
											
											const Eigen::Vector3f driveForceN = (agentVcKLen * vcStiffnessScale) * disp + vcCLen * (devVel - sphereVel);
											const float sphereInvMass = 1.0f / agentProxyMassKg;

												if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
														contact2 = solveAgentSphereTriangleCollisionConstraint(
															sphereCenter,
															sphereVel,
															sphereInvMass,
															contactProxyInvMassScale,
															contactVelRelax,
															contactVelRelaxMin,
															contactNormalDamp,
															r,
															allowedPen,
															eps,
															timeStep,
															corr,
															tangentialDamp,
															agentFrictionMu,
															iters2,
															manifoldK,
															&agentLastActiveContactTriangle[static_cast<size_t>(fi)],
															driveForceN,
															agentContactTriangles,
															agentContactTrianglePhysIds,
															agentContactTriangleNeighbors,
															agentVerticesByPhysId,
														physMassSumKg);
												} else if (!agentContactVertexPhysIds.empty()) {
												contact2 = solveAgentSphereVertexCollisionConstraint(
													sphereCenter,
													sphereVel,
													sphereInvMass,
													contactProxyInvMassScale,
													contactVelRelax,
													contactVelRelaxMin,
													contactNormalDamp,
													r,
													allowedPen,
													eps,
													timeStep,
													corr,
													tangentialDamp,
													agentFrictionMu,
													iters2,
													driveForceN,
													agentContactVertexPhysIds,
													agentVerticesByPhysId,
													physMassSumKg);
											}

											agentLastContactForcesN[static_cast<size_t>(fi)] = contact2.reactionForceN;
											agentLastContactCounts[static_cast<size_t>(fi)] = contact2.contactVertexCount;
											agentLastContactPenetrations[static_cast<size_t>(fi)] = contact2.maxPenetration;
											anyContact2 = anyContact2 || (contact2.contactVertexCount > 0);
											maxPen2 = std::max(maxPen2, contact2.maxPenetration);
										}
										anyContact = anyContact || anyContact2;
										maxPenetrationThisFrame = std::max(maxPenetrationThisFrame, maxPen2);
									}
								}

									// Proxy anti-tunneling (rare): if the proxy CENTER ended up fully inside the closed surface
									// *and* we had no contact constraints this frame, project it back to just outside the closest
									// surface point. This is intentionally gated to avoid per-frame ray casting (perf + haptics).
										if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
											const float rProxy = std::max(1e-6f, agentSphere.radius);
											const float slop = std::max(1e-4f * bboxDiag, 0.02f * rProxy);
											const float rayEps = 1e-5f * bboxDiag;

										// Cheap reject: if far away from the (initial) object bbox, we cannot be inside.
										// Keep the margin modest to avoid running expensive ray casts when the proxy is just "in the air".
										const Eigen::Vector3f margin = Eigen::Vector3f::Ones() * (0.10f * bboxDiag + rProxy + slop);
										const Eigen::Vector3f clampMin = bboxMin - margin;
										const Eigen::Vector3f clampMax = bboxMax + margin;

										for (int fi = 0; fi < kFingerCount; ++fi) {
											if (agentLastContactCounts[static_cast<size_t>(fi)] > 0) continue;
											Eigen::Vector3f p = agentProxyPositions[static_cast<size_t>(fi)];
											if (p.x() < clampMin.x() || p.y() < clampMin.y() || p.z() < clampMin.z() ||
												p.x() > clampMax.x() || p.y() > clampMax.y() || p.z() > clampMax.z()) {
												continue;
											}

											// Most frames are not tunnel events. Only do an expensive inside test when the proxy
											// is plausibly "fully inside" (farther than r+slop from its locally closest triangle),
											// and only at a low rate to avoid frame drops.
											const int preferredTi = agentLastActiveContactTriangle[static_cast<size_t>(fi)];
											if (preferredTi >= 0 && preferredTi < static_cast<int>(agentContactTriangles.size())) {
												const auto& tri = agentContactTriangles[static_cast<size_t>(preferredTi)];
												if (tri.a && tri.b && tri.c) {
													const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
													const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
													const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
													const Eigen::Vector3f q = closestPointOnTriangle(p, a, b, c, nullptr);
													const float dist = (q - p).norm();
													if (dist <= (rProxy + slop)) continue;
												}
											}
											constexpr int kInsideTestIntervalFrames = 4;
											if (((frame + fi) % kInsideTestIntervalFrames) != 0) continue;

											if (!isPointInsideSurfaceRayCastMulti(p, agentContactTriangles, rayEps)) continue;

											const AgentSurfaceQueryResult q = queryAgentSurface(p, agentContactTriangles);
											// Only correct the "fully inside with no intersection" case. If the sphere center is inside
											// but still within radius of the surface, that's a valid deep indentation state.
											if (q.found && q.outwardNormal.squaredNorm() > 1e-12f && q.distanceToSurface > (rProxy + slop)) {
												p = q.closestPoint + q.outwardNormal * slop;

												Eigen::Vector3f v = agentProxyVelocities[static_cast<size_t>(fi)];
												const float vn = v.dot(q.outwardNormal);
												if (vn < 0.0f) v -= q.outwardNormal * vn;

												agentProxyPositions[static_cast<size_t>(fi)] = p;
												agentProxyVelocities[static_cast<size_t>(fi)] = v;
											}
										}
									}

									// If using virtual coupling, update the device/coupling force after contact may have
									// pushed the proxy (so logs/plots reflect the final state this frame).
									if (useVC) {
									const float maxVcDist = std::max(0.0f, agentVcMaxDistanceRadiusFrac) * r;
									for (int fi = 0; fi < kFingerCount; ++fi) {
										const Eigen::Vector3f& devPos = agentDevicePositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f& devVel = agentDeviceVelocities[static_cast<size_t>(fi)];
										const Eigen::Vector3f& proxyPos = agentProxyPositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f& proxyVel = agentProxyVelocities[static_cast<size_t>(fi)];

										Eigen::Vector3f disp = devPos - proxyPos;
										const float dispLen = disp.norm();
											if (maxVcDist > 1e-6f && dispLen > maxVcDist) {
												disp *= (maxVcDist / dispLen);
											}

											// Adaptive VC stiffness for force feedback
											float vcStiffnessScale = 1.0f;
											const bool wasContact = (agentLastContactCounts[static_cast<size_t>(fi)] > 0);
											if (wasContact) {
												const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
												const float penFrac = lastPen / std::max(1e-6f, agentSphere.radius);
												if (penFrac > 0.15f) {
													const float t = std::clamp((penFrac - 0.15f) / 0.25f, 0.0f, 1.0f);
													vcStiffnessScale = 1.0f - 0.95f * t;
												}
											}
											
											const Eigen::Vector3f couplingForceN =
												(agentVcKLen * vcStiffnessScale) * disp +
												(wasContact ? agentVcCLenContact : agentVcCLenFree) * (devVel - proxyVel);
											agentLastCouplingForcesN[static_cast<size_t>(fi)] = couplingForceN;
											const float vcCLen = wasContact ? agentVcCLenContact : agentVcCLenFree;
											const Eigen::Vector3f springForceN = (agentVcKLen * vcStiffnessScale) * disp;
											agentLastDeviceForcesN[static_cast<size_t>(fi)] = -(springForceN + vcCLen * devVel);
										}
									}

									// Optional: low-pass filter the device force for haptic stability (does not affect motion).
									{
										const float tauBase = std::max(0.0f, agentDeviceForceFilterTauSec);
										if (tauBase > 0.0f) {
											const float v0 = 0.02f * bboxDiag;
											const float v1 = 0.10f * bboxDiag;
											const float tauContact = std::max(tauBase, 0.02f); // heavier smoothing only while pressing
											for (int fi = 0; fi < kFingerCount; ++fi) {
												const Eigen::Vector3f raw = agentLastDeviceForcesN[static_cast<size_t>(fi)];
												Eigen::Vector3f& f = agentFilteredDeviceForcesN[static_cast<size_t>(fi)];
												float tauEff = tauBase;
												if (agentLastContactCounts[static_cast<size_t>(fi)] > 0) {
													const float speed = agentDeviceVelocities[static_cast<size_t>(fi)].norm();
													const float t = (v1 > v0) ? std::clamp((speed - v0) / (v1 - v0), 0.0f, 1.0f) : 1.0f;
													tauEff = tauContact + (tauBase - tauContact) * t;
												}
												const float a = std::exp(-timeStep / std::max(1e-6f, tauEff));
												f = f * a + raw * (1.0f - a);
												agentLastDeviceForcesN[static_cast<size_t>(fi)] = f;
											}
										} else {
											for (int fi = 0; fi < kFingerCount; ++fi) {
												agentFilteredDeviceForcesN[static_cast<size_t>(fi)] =
													agentLastDeviceForcesN[static_cast<size_t>(fi)];
											}
										}
									}
	
							// Keep Group::groupVelocity consistent with Vertex::vel* for the next frame's primeVec.
							if (anyContact) {
								for (int gi = 0; gi < object.groupNum; ++gi) {
									Group& group = object.groups[gi];
									for (const auto& vertexPair : group.verticesMap) {
										Vertex* v = vertexPair.second;
									if (!v) continue;
									const int li = v->localIndex;
									if (li < 0 || (3 * li + 2) >= group.groupVelocity.size()) continue;
									if (v->isFixed) {
										group.groupVelocity.segment<3>(3 * li) = Eigen::Vector3f::Zero();
									} else {
										group.groupVelocity.segment<3>(3 * li) = Eigen::Vector3f(v->velx, v->vely, v->velz);
									}
								}
							}
						}

							// Optional: write live force/pose file (post-solve values).
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
											f << "fingers " << kFingerCount << "\n";
											for (int fi = 0; fi < kFingerCount; ++fi) {
												const std::string idx = std::to_string(fi);
												const Eigen::Vector3f& devPos = agentDevicePositions[static_cast<size_t>(fi)];
												const Eigen::Vector3f& proxyPos = agentProxyPositions[static_cast<size_t>(fi)];
												const Eigen::Vector3f& devForce = agentLastDeviceForcesN[static_cast<size_t>(fi)];
												const Eigen::Vector3f& contactForce = agentLastContactForcesN[static_cast<size_t>(fi)];
												const int contacts = agentLastContactCounts[static_cast<size_t>(fi)];

												f << "finger" << idx << "_name " << kFingerNames[static_cast<size_t>(fi)] << "\n";
												f << "finger" << idx << "_devicePos " << devPos.x() << " " << devPos.y() << " " << devPos.z() << "\n";
												f << "finger" << idx << "_proxyPos " << proxyPos.x() << " " << proxyPos.y() << " " << proxyPos.z() << "\n";
												f << "finger" << idx << "_deviceForceN " << devForce.x() << " " << devForce.y() << " " << devForce.z() << "\n";
												f << "finger" << idx << "_contactForceN " << contactForce.x() << " " << contactForce.y() << " " << contactForce.z() << "\n";
												f << "finger" << idx << "_contacts " << contacts << "\n";
											}
										}
									} catch (...) {
										// ignore live file errors
									}
							}
						}
					}

					// Axis-aligned "walls" (Y±, X+) tight to the initial bbox.
						if (wallEnabled && !agentVerticesByPhysId.empty()) {
						const Eigen::Vector3f extents = bboxMax - bboxMin;
						const Eigen::Vector3f margin = std::max(0.0f, wallMarginBboxScale) * extents;
						const float wallXMax = bboxMax.x() + margin.x();
						const float wallYMin = bboxMin.y() - margin.y();
						const float wallYMax = bboxMax.y() + margin.y();

						const int wallHits = applyAxisAlignedWallConstraints(
							agentVerticesByPhysId,
							wallXMax,
							wallYMin,
							wallYMax,
							timeStep,
							wallRestitution,
							wallTangentialDamp);

						// Keep Group::groupVelocity consistent with Vertex::vel* for the next frame's primeVec.
						if (wallHits > 0) {
							for (int gi = 0; gi < object.groupNum; ++gi) {
								Group& group = object.groups[gi];
								for (const auto& vertexPair : group.verticesMap) {
									Vertex* v = vertexPair.second;
									if (!v) continue;
									const int li = v->localIndex;
									if (li < 0 || (3 * li + 2) >= group.groupVelocity.size()) continue;
									if (v->isFixed) {
										group.groupVelocity.segment<3>(3 * li) = Eigen::Vector3f::Zero();
									} else {
										group.groupVelocity.segment<3>(3 * li) = Eigen::Vector3f(v->velx, v->vely, v->velz);
									}
								}
							}
							}
						}

						// Update previous physical positions for the next frame's CCD.
						if (!physPrevPositions.empty()) {
							for (size_t id = 0; id < agentVerticesByPhysId.size(); ++id) {
								const auto& list = agentVerticesByPhysId[id];
								if (list.empty() || !list.front()) continue;
								const Vertex* v = list.front();
								physPrevPositions[id] = Eigen::Vector3f(v->x, v->y, v->z);
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

					// Proxies (colored by finger).
					const std::array<Eigen::Vector3f, kFingerCount> proxyColors = {
						Eigen::Vector3f(0.92f, 0.18f, 0.18f), // thumb
					Eigen::Vector3f(1.00f, 0.55f, 0.20f), // index
					Eigen::Vector3f(0.95f, 0.90f, 0.20f), // middle
					Eigen::Vector3f(0.25f, 0.90f, 0.35f), // ring
					Eigen::Vector3f(0.30f, 0.60f, 0.95f)  // pinky
				};
				for (int fi = 0; fi < kFingerCount; ++fi) {
					Eigen::Vector3f c = proxyColors[static_cast<size_t>(fi)];
					if (whiteBackground) c *= 0.75f;
						glColor3f(c.x(), c.y(), c.z());
						drawWireSphereCircles(agentProxyPositions[static_cast<size_t>(fi)], agentSphere.radius, 36);
					}
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

			// Draw axis-aligned walls (Y±, X+) around the initial bbox.
			if (wallEnabled) {
				const Eigen::Vector3f extents = bboxMax - bboxMin;
				const Eigen::Vector3f margin = std::max(0.0f, wallMarginBboxScale) * extents;
				const float x0 = bboxMin.x() - margin.x();
				const float x1 = bboxMax.x() + margin.x();
				const float y0 = bboxMin.y() - margin.y();
				const float y1 = bboxMax.y() + margin.y();
				const float z0 = bboxMin.z() - margin.z();
				const float z1 = bboxMax.z() + margin.z();

				const bool blendWasEnabled = (glIsEnabled(GL_BLEND) == GL_TRUE);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				glDepthMask(GL_FALSE);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

				if (whiteBackground) {
					glColor4f(0.1f, 0.2f, 0.6f, 0.18f);
				} else {
					glColor4f(0.6f, 0.7f, 1.0f, 0.18f);
				}

				glBegin(GL_QUADS);
				// +Y wall
				glVertex3f(x0, y1, z0);
				glVertex3f(x1, y1, z0);
				glVertex3f(x1, y1, z1);
				glVertex3f(x0, y1, z1);
				// -Y wall
				glVertex3f(x0, y0, z0);
				glVertex3f(x1, y0, z0);
				glVertex3f(x1, y0, z1);
				glVertex3f(x0, y0, z1);
				// +X wall
				glVertex3f(x1, y0, z0);
				glVertex3f(x1, y1, z0);
				glVertex3f(x1, y1, z1);
				glVertex3f(x1, y0, z1);
				glEnd();

				glDepthMask(GL_TRUE);
				if (!blendWasEnabled) glDisable(GL_BLEND);
				if (!showVolumePreservation) {
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				}
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
			// Record force history for plotting.
			// Requested: show only the INDEX fingertip proxy force (not the sum across fingers).
			static constexpr int kForceGraphFingerIndex = 1; // 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
			static ForceGraphHistory agentForceHistory;
			static float agentForceGraphScaleN = 1.0f;
			if (agentSphere.enabled && !isPaused) {
				const Eigen::Vector3f proxyForceN = agentLastContactForcesN[static_cast<size_t>(kForceGraphFingerIndex)];
				agentForceHistory.push(Eigen::Vector4f(proxyForceN.x(), proxyForceN.y(), proxyForceN.z(), proxyForceN.norm()));
			}

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

			// Agent force mini graph (bottom-left, above the status label).
			if (agentSphere.enabled) {
				const float labelH = 28.0f;
				const float graphW = 420.0f;
				const float graphH = 120.0f;
				const float graphX = uiMargin;
				const float graphY = ui.state().windowHeight - uiMargin - labelH - 8.0f - graphH;

				// Panel background in framebuffer coordinates.
				const float sx = ui.state().scaleX;
				const float sy = ui.state().scaleY;
				const float x = graphX * sx;
				const float y = graphY * sy;
				const float w = graphW * sx;
				const float h = graphH * sy;

				glColor4f(0.05f, 0.05f, 0.06f, 0.72f);
				glBegin(GL_QUADS);
				glVertex2f(x, y);
				glVertex2f(x + w, y);
				glVertex2f(x + w, y + h);
				glVertex2f(x, y + h);
				glEnd();

				glColor4f(1.0f, 1.0f, 1.0f, 0.18f);
				glBegin(GL_LINE_LOOP);
				glVertex2f(x, y);
				glVertex2f(x + w, y);
				glVertex2f(x + w, y + h);
				glVertex2f(x, y + h);
				glEnd();

				const float padWin = 10.0f;
				const float legendHWin = 16.0f;
				const float padX = padWin * sx;
				const float padY = padWin * sy;
				const float legendH = legendHWin * sy;

				const float plotX = x + padX;
				const float plotY = y + padY + legendH;
				const float plotW = std::max(1.0f, w - 2.0f * padX);
				const float plotH = std::max(1.0f, h - 2.0f * padY - legendH);
				const float yMid = plotY + plotH * 0.5f;

				// Auto-scale (slow decay).
				float maxAbs = 1e-3f;
				const int n = agentForceHistory.size();
				for (int i = 0; i < n; ++i) {
					const Eigen::Vector4f s = agentForceHistory.at(i);
					maxAbs = std::max(maxAbs, std::abs(s.x()));
					maxAbs = std::max(maxAbs, std::abs(s.y()));
					maxAbs = std::max(maxAbs, std::abs(s.z()));
					maxAbs = std::max(maxAbs, s.w());
				}
				const float targetScale = std::max(1e-3f, maxAbs * 1.10f);
				if (targetScale > agentForceGraphScaleN) agentForceGraphScaleN = targetScale;
				else agentForceGraphScaleN = std::max(targetScale, agentForceGraphScaleN * 0.985f);

				// Zero line.
				glColor4f(1.0f, 1.0f, 1.0f, 0.18f);
				glBegin(GL_LINES);
				glVertex2f(plotX, yMid);
				glVertex2f(plotX + plotW, yMid);
				glEnd();

				auto mapY = [&](float v) -> float {
					const float a = (plotH * 0.48f) / std::max(1e-6f, agentForceGraphScaleN);
					return yMid - v * a;
				};

				auto drawSeries = [&](int comp, float r, float g, float b, float a) {
					if (n <= 0) return;
					glColor4f(r, g, b, a);
					glLineWidth(2.0f);
					glBegin(GL_LINE_STRIP);
					for (int i = 0; i < n; ++i) {
						const float t = (n > 1) ? (static_cast<float>(i) / static_cast<float>(n - 1)) : 0.0f;
						const float px = plotX + t * plotW;
						const float v = agentForceHistory.at(i)[comp];
						glVertex2f(px, mapY(v));
					}
					glEnd();
				};

				// Colors: X=red, Y=green, Z=blue, |F|=yellow.
				drawSeries(0, 0.85f, 0.25f, 0.25f, 0.95f);
				drawSeries(1, 0.25f, 0.85f, 0.25f, 0.95f);
				drawSeries(2, 0.30f, 0.55f, 0.95f, 0.95f);
				drawSeries(3, 0.95f, 0.85f, 0.20f, 0.95f);

				// Title/scale label (segment font supports digits + letters only).
				{
					const SimpleUI::Rect graphLabelRect{ graphX + 8.0f, graphY + 2.0f, graphW - 16.0f, legendHWin };
					const std::string label =
						"INDEX PROXY FORCE N  SCALE " + formatSignedInt(agentForceGraphScaleN) +
						"  FX FY FZ MAG";
					ui.drawLabel(graphLabelRect, label, 9.0f);
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
					const Eigen::Vector3f proxyForceN = agentLastContactForcesN[static_cast<size_t>(kForceGraphFingerIndex)];
					const int contacts = agentLastContactCounts[static_cast<size_t>(kForceGraphFingerIndex)];
					const std::string label =
						"AGENT ON VC " + std::string(agentUseVC ? "1" : "0") +
						" IDX FX " + formatSignedInt(proxyForceN.x()) +
						" FY " + formatSignedInt(proxyForceN.y()) +
						" FZ " + formatSignedInt(proxyForceN.z()) +
						" CNT " + std::to_string(contacts);
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
