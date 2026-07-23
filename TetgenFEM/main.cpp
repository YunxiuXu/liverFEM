
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
#include <sstream>
#include <iomanip>
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
#include "HapticInterface.h"

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
			hasRightHand_ = false;
			hasLeftHand_ = false;
			rightTimeSec_ = -1.0;
			leftTimeSec_ = -1.0;
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
			if (!hasRightHand_) return false;
			if (outTipsMm) *outTipsMm = rightTipsMm_;
			if (outPalmMm) *outPalmMm = rightPalmMm_;
			if (outTimeSec) *outTimeSec = rightTimeSec_;
			return true;
		}

		bool getRightHandDistalPrevMm(std::array<Eigen::Vector3f, 5>* outPrevMm) const
		{
			if (!hasRightHand_) return false;
			if (outPrevMm) *outPrevMm = rightDistalPrevMm_;
			return true;
		}

		bool getRightHandRotations(std::array<Eigen::Quaternionf, 5>* outRotations) const
		{
			if (!hasRightHand_) return false;
			if (outRotations) *outRotations = rightTipsRot_;
			return true;
		}

		bool getLeftHandTipsMm(std::array<Eigen::Vector3f, 5>* outTipsMm, Eigen::Vector3f* outPalmMm, double* outTimeSec) const
		{
			if (!hasLeftHand_) return false;
			if (outTipsMm) *outTipsMm = leftTipsMm_;
			if (outPalmMm) *outPalmMm = leftPalmMm_;
			if (outTimeSec) *outTimeSec = leftTimeSec_;
			return true;
		}

		bool getLeftHandDistalPrevMm(std::array<Eigen::Vector3f, 5>* outPrevMm) const
		{
			if (!hasLeftHand_) return false;
			if (outPrevMm) *outPrevMm = leftDistalPrevMm_;
			return true;
		}

		bool getLeftHandRotations(std::array<Eigen::Quaternionf, 5>* outRotations) const
		{
			if (!hasLeftHand_) return false;
			if (outRotations) *outRotations = leftTipsRot_;
			return true;
		}

	private:
		void onTracking(const LEAP_TRACKING_EVENT* evt, double nowSec)
		{
			if (!evt) return;
			bool sawRight = false;
			bool sawLeft = false;
			for (uint32_t i = 0; i < evt->nHands; ++i) {
				const LEAP_HAND* hand = &evt->pHands[i];
				if (!hand) continue;
				const bool isRight = (hand->type == eLeapHandType_Right);
				const bool isLeft = (hand->type == eLeapHandType_Left);
				if (!isRight && !isLeft) continue;

				const LEAP_VECTOR palm = hand->palm.position; // mm
				Eigen::Vector3f palmMm(palm.x, palm.y, palm.z);

				std::array<Eigen::Vector3f, 5> tipsMm;
				std::array<Eigen::Quaternionf, 5> tipsRot;
				std::array<Eigen::Vector3f, 5> prevMm;
				// digits[0]=thumb, [1]=index, [2]=middle, [3]=ring, [4]=pinky
				for (int fi = 0; fi < 5; ++fi) {
					const LEAP_VECTOR tip = hand->digits[fi].distal.next_joint; // mm
					tipsMm[static_cast<size_t>(fi)] = Eigen::Vector3f(tip.x, tip.y, tip.z);

					const LEAP_VECTOR prev = hand->digits[fi].distal.prev_joint; // mm
					prevMm[static_cast<size_t>(fi)] = Eigen::Vector3f(prev.x, prev.y, prev.z);

					const LEAP_QUATERNION rot = hand->digits[fi].distal.rotation;
					tipsRot[static_cast<size_t>(fi)] = Eigen::Quaternionf(rot.w, rot.x, rot.y, rot.z);
				}

				if (isRight) {
					rightPalmMm_ = palmMm;
					rightTipsMm_ = tipsMm;
					rightDistalPrevMm_ = prevMm;
					rightTipsRot_ = tipsRot;
					rightTimeSec_ = nowSec;
					sawRight = true;
				}
				if (isLeft) {
					leftPalmMm_ = palmMm;
					leftTipsMm_ = tipsMm;
					leftDistalPrevMm_ = prevMm;
					leftTipsRot_ = tipsRot;
					leftTimeSec_ = nowSec;
					sawLeft = true;
				}
			}
			hasRightHand_ = sawRight;
			hasLeftHand_ = sawLeft;
		}

		LEAP_CONNECTION connection_ = nullptr;
		bool connected_ = false;
		bool hasRightHand_ = false;
		bool hasLeftHand_ = false;
		Eigen::Vector3f rightPalmMm_ = Eigen::Vector3f::Zero();
		Eigen::Vector3f leftPalmMm_ = Eigen::Vector3f::Zero();
		std::array<Eigen::Vector3f, 5> rightTipsMm_{};
		std::array<Eigen::Vector3f, 5> leftTipsMm_{};
		std::array<Eigen::Vector3f, 5> rightDistalPrevMm_{};
		std::array<Eigen::Vector3f, 5> leftDistalPrevMm_{};
		std::array<Eigen::Quaternionf, 5> rightTipsRot_{};
		std::array<Eigen::Quaternionf, 5> leftTipsRot_{};
		double rightTimeSec_ = -1.0;
		double leftTimeSec_ = -1.0;
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

static Eigen::Vector3f closestPointOnTriangleRTCD(
	const Eigen::Vector3f& p,
	const Eigen::Vector3f& a,
	const Eigen::Vector3f& b,
	const Eigen::Vector3f& c)
{
	// Real-Time Collision Detection (Christer Ericson), closest point on triangle.
	const Eigen::Vector3f ab = b - a;
	const Eigen::Vector3f ac = c - a;
	const Eigen::Vector3f ap = p - a;
	const float d1 = ab.dot(ap);
	const float d2 = ac.dot(ap);
	if (d1 <= 0.0f && d2 <= 0.0f) return a;

	const Eigen::Vector3f bp = p - b;
	const float d3 = ab.dot(bp);
	const float d4 = ac.dot(bp);
	if (d3 >= 0.0f && d4 <= d3) return b;

	const float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		const float v = d1 / (d1 - d3);
		return a + v * ab;
	}

	const Eigen::Vector3f cp = p - c;
	const float d5 = ab.dot(cp);
	const float d6 = ac.dot(cp);
	if (d6 >= 0.0f && d5 <= d6) return c;

	const float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		const float w = d2 / (d2 - d6);
		return a + w * ac;
	}

	const float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b);
	}

	const float denom = 1.0f / (va + vb + vc);
	const float v = vb * denom;
	const float w = vc * denom;
	return a + ab * v + ac * w;
}

static bool outwardNormalForTriangleInit(
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
		const Eigen::Vector3f opp(tri.opp->initx, tri.opp->inity, tri.opp->initz);
		const float sOpp = nRaw.dot(opp - a);
		if (std::abs(sOpp) > 1e-18f) {
			if (sOpp > 0.0f) n = -nRaw;
			else n = nRaw;
		}
	}
	*outwardUnitOut = n.normalized();
	return true;
}

static int applyLiverCavityConstraints(
	const std::vector<std::vector<Vertex*>>& verticesByPhysId,
	const std::vector<AgentTriangle>& surfaceTriangles,
	const std::vector<std::vector<int>>& surfaceTriangleNeighbors,
	const std::vector<char>& cavityTriangleEnabled,
	const std::vector<int>& surfacePhysIds,
	std::vector<int>& activeTriangleByPhysIdInOut,
	float cavityGap,
	int openAxis,
	float openLo,
	float openHi,
	float dt,
	float restitution,
	float tangentialDamp)
{
	if (verticesByPhysId.empty() || surfaceTriangles.empty() || surfacePhysIds.empty()) return 0;
	if (!surfaceTriangleNeighbors.empty() && surfaceTriangleNeighbors.size() != surfaceTriangles.size()) return 0;
	if (!cavityTriangleEnabled.empty() && cavityTriangleEnabled.size() != surfaceTriangles.size()) return 0;

	const float gap = std::max(0.0f, cavityGap);
	const int axis = std::clamp(openAxis, 0, 2);
	const float lo = std::min(openLo, openHi);
	const float hi = std::max(openLo, openHi);
	const float invDt = 1.0f / std::max(1e-8f, dt);
	const float rest = std::clamp(restitution, 0.0f, 1.0f);
	const float tanD = std::clamp(tangentialDamp, 0.0f, 1.0f);

	auto triIsEnabled = [&](int ti) -> bool {
		if (ti < 0 || ti >= static_cast<int>(surfaceTriangles.size())) return false;
		if (cavityTriangleEnabled.empty()) return true;
		return cavityTriangleEnabled[static_cast<size_t>(ti)] != 0;
	};

	auto firstEnabledTri = [&]() -> int {
		for (int ti = 0; ti < static_cast<int>(surfaceTriangles.size()); ++ti) {
			if (triIsEnabled(ti)) return ti;
		}
		return -1;
	};

	const int defaultTri = firstEnabledTri();
	if (defaultTri < 0) return 0;

	auto dist2ToTri = [&](int ti, const Eigen::Vector3f& p, Eigen::Vector3f* outwardNOut, Eigen::Vector3f* aOut, float* signedPlaneOut) -> float {
		if (ti < 0 || ti >= static_cast<int>(surfaceTriangles.size())) return std::numeric_limits<float>::infinity();
		if (!triIsEnabled(ti)) return std::numeric_limits<float>::infinity();
		const AgentTriangle& tri = surfaceTriangles[static_cast<size_t>(ti)];
		if (!tri.a || !tri.b || !tri.c) return std::numeric_limits<float>::infinity();

		const Eigen::Vector3f a(tri.a->initx, tri.a->inity, tri.a->initz);
		const Eigen::Vector3f b(tri.b->initx, tri.b->inity, tri.b->initz);
		const Eigen::Vector3f c(tri.c->initx, tri.c->inity, tri.c->initz);
		Eigen::Vector3f outwardN = Eigen::Vector3f::Zero();
		if (!outwardNormalForTriangleInit(tri, a, b, c, &outwardN)) return std::numeric_limits<float>::infinity();

		const Eigen::Vector3f q = closestPointOnTriangleRTCD(p, a, b, c);
		if (outwardNOut) *outwardNOut = outwardN;
		if (aOut) *aOut = a;
		if (signedPlaneOut) *signedPlaneOut = outwardN.dot(p - a);
		return (q - p).squaredNorm();
	};

	int hitCount = 0;
	const int maxWalkSteps = 12;

	for (int pid : surfacePhysIds) {
		if (pid < 0 || pid >= static_cast<int>(verticesByPhysId.size())) continue;
		const auto& list = verticesByPhysId[static_cast<size_t>(pid)];
		if (list.empty()) continue;

		bool anyFixed = false;
		for (Vertex* v : list) {
			if (v && v->isFixed) { anyFixed = true; break; }
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

		// Exposed side (surgeon access): leave the selected side open.
		if (pAvg[axis] >= lo && pAvg[axis] <= hi) continue;

		int ti = defaultTri;
		if (pid >= 0 && pid < static_cast<int>(activeTriangleByPhysIdInOut.size())) {
			ti = activeTriangleByPhysIdInOut[static_cast<size_t>(pid)];
			if (!triIsEnabled(ti)) ti = defaultTri;
		}

		// Local neighbor-walk: find a nearby closer triangle without scanning all triangles.
		float bestD2 = std::numeric_limits<float>::infinity();
		Eigen::Vector3f bestN = Eigen::Vector3f::Zero();
		Eigen::Vector3f bestA = Eigen::Vector3f::Zero();
		float bestSignedPlane = 0.0f;
		for (int step = 0; step < maxWalkSteps; ++step) {
			int bestTi = ti;
			Eigen::Vector3f nCand, aCand;
			float signedCand = 0.0f;
			const float d2Here = dist2ToTri(ti, pAvg, &nCand, &aCand, &signedCand);
			bestD2 = d2Here;
			bestN = nCand;
			bestA = aCand;
			bestSignedPlane = signedCand;

			if (!surfaceTriangleNeighbors.empty()) {
				const auto& nbrs = surfaceTriangleNeighbors[static_cast<size_t>(ti)];
				for (int nb : nbrs) {
					Eigen::Vector3f n2, a2;
					float s2 = 0.0f;
					const float d2 = dist2ToTri(nb, pAvg, &n2, &a2, &s2);
					if (d2 < bestD2) {
						bestD2 = d2;
						bestTi = nb;
						bestN = n2;
						bestA = a2;
						bestSignedPlane = s2;
					}
				}
			}

			if (bestTi == ti) break;
			ti = bestTi;
		}

		if (!std::isfinite(bestD2) || bestN.squaredNorm() < 1e-12f) continue;
		if (pid >= 0 && pid < static_cast<int>(activeTriangleByPhysIdInOut.size())) {
			activeTriangleByPhysIdInOut[static_cast<size_t>(pid)] = ti;
		}

		// For an outward-facing surface, plane distance > gap means we've pushed outside the inflated cavity.
		const float penetration = bestSignedPlane - gap;
		if (penetration <= 0.0f) continue;

		// Limit per-step correction to avoid startup "ghost pull" when initial pose slightly intersects
		// the cavity (e.g., after model reorientation / discretization mismatch).
		// Keep cavity push-back gentle and bounded to avoid startup/occasional ghost-force snaps.
		const float maxCorrection = std::max(0.001f, 0.15f * std::max(0.0f, gap));
		const float usedPenetration = std::min(penetration, maxCorrection);
		const Eigen::Vector3f pNew = pAvg - bestN * usedPenetration;
		Eigen::Vector3f vNew = vAvg + (pNew - pAvg) * invDt;

		// Velocity response (like a plane collision), using the cavity outward normal.
		const float vn = vNew.dot(bestN);
		if (vn > 0.0f) vNew -= bestN * ((1.0f + rest) * vn);
		const Eigen::Vector3f vt = vNew - bestN * vNew.dot(bestN);
		vNew -= vt * tanD;

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
				bool injectVelocityFromPositionCorrection,
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
				const bool injectVel = injectVelocityFromPositionCorrection;

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
					if (injectVel) {
						float velocityRelaxation = velRelaxBase;
						const float corrLen = dp.norm();
						if (corrLen > 1e-4f) {
							velocityRelaxation = std::max(velRelaxMin, velRelaxBase / (1.0f + 20.0f * corrLen));
						}
						vv += dp * invDt * velocityRelaxation;
					}

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
						if (dist2Out) *dist2Out = dist2;
						return dist2 < shellR2;
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

							// Manifold normal-consistency: prevent contact "fighting" between opposite sides of
							// thin features (common source of local twitching + force buzzing).
							// Allow fairly sharp edges, but reject near-opposite faces.
							const float manifoldNormalCosMin = -0.2f;
							Eigen::Vector3f preferredOutwardN = Eigen::Vector3f::Zero();
							bool hasPreferredOutwardN = false;

							auto triOutwardNormal = [&](int ti, Eigen::Vector3f* nOut) -> bool {
								if (!nOut) return false;
								if (ti < 0 || ti >= static_cast<int>(contactTriangles.size())) return false;
								const auto& tri = contactTriangles[ti];
								if (!tri.a || !tri.b || !tri.c) return false;
								const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
								const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
								const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
								return outwardNormalForTriangle(tri, a, b, c, nOut);
							};

						auto tryInsert = [&](int ti, float d2, bool enforceNormalConsistency) {
							for (int existing : activeTis) {
								if (existing == ti) return;
							}

							if (enforceNormalConsistency && hasPreferredOutwardN) {
								Eigen::Vector3f outN = Eigen::Vector3f::Zero();
								if (!triOutwardNormal(ti, &outN)) return;
								if (outN.dot(preferredOutwardN) < manifoldNormalCosMin) return;
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

							// Establish a reference outward normal from the preferred triangle (if possible).
							if (preferredTi >= 0) {
								hasPreferredOutwardN = triOutwardNormal(preferredTi, &preferredOutwardN);
							}

							// Local manifold search around the preferred triangle.
							if (preferredTi >= 0 && !contactTriangleNeighbors.empty()) {
								const int maxVisited = 128;
								std::vector<int> queue;
								queue.reserve(static_cast<size_t>(maxVisited));
								std::vector<std::pair<int, float>> visitedD2;
								visitedD2.reserve(static_cast<size_t>(maxVisited));

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
									float d2 = std::numeric_limits<float>::infinity();
									const bool inShell = triInShell(ti, &d2);
									visitedD2.emplace_back(ti, d2);
									if (inShell) tryInsert(ti, d2, /*enforceNormalConsistency=*/true);

									const auto& nbs = contactTriangleNeighbors[static_cast<size_t>(ti)];
									for (int nb : nbs) {
										if (static_cast<int>(queue.size()) >= maxVisited) break;
										pushUnique(nb);
									}
								}

								// If we are deep inside (no triangles are within the narrow shell radius),
								// fall back to the closest triangles in the visited neighborhood so we still
								// produce a consistent push-out direction and avoid "no force" oscillation.
								if (activeTis.empty()) {
									for (const auto& it : visitedD2) {
										if (!std::isfinite(it.second)) continue;
										// Relax normal-consistency when we have no candidates at all (deep inside case).
										tryInsert(it.first, it.second, /*enforceNormalConsistency=*/false);
									}
								}
							} else {
								// Fallback: slow scan (should be rare; e.g., if neighbors not provided).
								if (preferredTi >= 0) {
									float d2 = 0.0f;
									if (triInShell(preferredTi, &d2)) tryInsert(preferredTi, d2, /*enforceNormalConsistency=*/true);
								}
								for (int ti = 0; ti < static_cast<int>(contactTriangles.size()); ++ti) {
									if (ti == preferredTi) continue;
									float d2 = 0.0f;
									if (!triInShell(ti, &d2)) continue;
									tryInsert(ti, d2, /*enforceNormalConsistency=*/true);
								}

								// Deep-inside fallback: keep at least the preferred triangle so we can push out.
								if (activeTis.empty() && preferredTi >= 0) {
									const float d2 = triDistance2(preferredTi);
									if (std::isfinite(d2)) tryInsert(preferredTi, d2, /*enforceNormalConsistency=*/false);
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

							if (dist2 < bestD2ThisIter) {
								bestD2ThisIter = dist2;
								bestTiThisIter = ti;
							}

							Eigen::Vector3f outwardN = Eigen::Vector3f::Zero();
							const bool hasOutward = outwardNormalForTriangle(tri, a, b, c, &outwardN);

							const Eigen::Vector3f d = q - sphereCenter;
							const float dist = std::sqrt(std::max(dist2, 1e-18f));
							const float pen = shellR - dist;
							if (pen <= 0.0f) continue;

							Eigen::Vector3f n = (dist > 1e-6f) ? (d / dist) : Eigen::Vector3f(0.0f, 1.0f, 0.0f);
							if (dist <= 1e-6f && hasOutward) {
								// Fallback normal if d is degenerate (use inward direction).
								n = -outwardN;
							}

							anyPenetration = true;

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
							// Per-iteration clamp: avoids explosive corrections, but must still be strong enough
							// to prevent "tunnel / depenetrate" chatter under fast motion.
							//
							// NOTE: kinematic mode substeps reduce solver iterations per substep; if this clamp is
							// too conservative, penetration accumulates and shows up as buzzing/ghosting. Prefer a
							// slightly larger clamp and rely on velocity relaxation + normal damping for stability.
							float maxCorrPerIter = 0.10f * shellR;
							if (shellR > 1e-12f) {
								// When deeply inside, allow a larger correction step so the proxy can't "sink"
								// many frames before being pushed back out.
								const float penOver = pen / shellR; // ~[0..3] after clamping
								const float tDeep = std::clamp((penOver - 1.0f) / 2.0f, 0.0f, 1.0f);
								maxCorrPerIter = shellR * (0.10f + 0.20f * tDeep); // 10%..30% of shellR
							}
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

struct PipelineProfiler {
	bool active = false;
	int warmupFrames = 60;
	int measureFrames = 240;
	int frameCount = 0;
	int sampleCount = 0;
	double sumTotalMs = 0.0;
	double sumPreSimMs = 0.0;
	double sumPhysicsMs = 0.0;
	double sumSimMs = 0.0;
	double sumRenderMs = 0.0;
	double sumHapticTxMs = 0.0;
	double maxTotalMs = 0.0;

	void record(double totalMs, double preSimMs, double physicsMs, double simMs, double renderMs, double hapticTxMs) {
		++frameCount;
		if (frameCount <= warmupFrames) return;
		if (sampleCount >= measureFrames) return;
		++sampleCount;
		sumTotalMs += totalMs;
		sumPreSimMs += preSimMs;
		sumPhysicsMs += physicsMs;
		sumSimMs += simMs;
		sumRenderMs += renderMs;
		sumHapticTxMs += hapticTxMs;
		maxTotalMs = std::max(maxTotalMs, totalMs);
	}

	bool finished() const { return sampleCount >= measureFrames; }

	void printSummary() const {
		if (sampleCount <= 0) return;
		const double inv = 1.0 / static_cast<double>(sampleCount);
		const double meanTotal = sumTotalMs * inv;
		const double meanPreSim = sumPreSimMs * inv;
		const double meanPhysics = sumPhysicsMs * inv;
		const double meanSim = sumSimMs * inv;
		const double meanRender = sumRenderMs * inv;
		const double meanHapticTx = sumHapticTxMs * inv;
		const double meanFps = (meanTotal > 1e-9) ? (1000.0 / meanTotal) : 0.0;
		std::cout << std::fixed << std::setprecision(3)
		          << "[PipelineProfile] samples=" << sampleCount
		          << " warmup=" << warmupFrames << "\n"
		          << "  mean_frame_ms=" << meanTotal
		          << " p99_frame_ms~=" << maxTotalMs
		          << " mean_fps=" << meanFps << "\n"
		          << "  mean_pre_sim_ms=" << meanPreSim
		          << " (Leap poll + hand mapping)\n"
		          << "  mean_physics_ms=" << meanPhysics
		          << " (GB-cFEM prime + PBD)\n"
		          << "  mean_full_sim_ms=" << meanSim
		          << " (physics + contact + force mapping/filter)\n"
		          << "  mean_render_ms=" << meanRender << "\n"
		          << "  mean_haptic_tx_ms=" << meanHapticTx
		          << " (force-to-serial; 0 if UART disabled)\n"
		          << "  software_latency_ms~=" << meanTotal
		          << " (same-frame hand input -> force command)\n";
	}
};

int main(int argc, char** argv) {

	bool exportTetgenAndExit = false;
	std::string exportDirOverride;
	PipelineProfiler pipelineProfiler;
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
		if (std::string(argv[i]) == "--profile-pipeline-frames" && i + 1 < argc) {
			pipelineProfiler.active = true;
			pipelineProfiler.measureFrames = std::max(1, std::atoi(argv[++i]));
			continue;
		}
		if (std::string(argv[i]) == "--profile-pipeline-warmup" && i + 1 < argc) {
			pipelineProfiler.warmupFrames = std::max(0, std::atoi(argv[++i]));
			continue;
		}
	}

	if (pipelineProfiler.active) {
		// Reproducible software-side timing: synthetic finger motion, no UART dependency.
		haptic_uart_enabled = false;
		leapEnabled = false;
		std::cout << "[PipelineProfile] enabled: warmup=" << pipelineProfiler.warmupFrames
		          << " measure=" << pipelineProfiler.measureFrames << "\n";
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
	
	// Optional: rotate the loaded TetGen mesh (about its bbox center).
	// This is applied before group division so material/group mapping stays consistent.
	if ((std::abs(model_rotateX_deg) > 1e-6f || std::abs(model_rotateY_deg) > 1e-6f || std::abs(model_rotateZ_deg) > 1e-6f) && out.pointlist && out.numberofpoints > 0) {
		double minx = std::numeric_limits<double>::infinity();
		double miny = std::numeric_limits<double>::infinity();
		double minz = std::numeric_limits<double>::infinity();
		double maxx = -std::numeric_limits<double>::infinity();
		double maxy = -std::numeric_limits<double>::infinity();
		double maxz = -std::numeric_limits<double>::infinity();
		for (int i = 0; i < out.numberofpoints; ++i) {
			const double x = static_cast<double>(out.pointlist[3 * i + 0]);
			const double y = static_cast<double>(out.pointlist[3 * i + 1]);
			const double z = static_cast<double>(out.pointlist[3 * i + 2]);
			minx = std::min(minx, x); miny = std::min(miny, y); minz = std::min(minz, z);
			maxx = std::max(maxx, x); maxy = std::max(maxy, y); maxz = std::max(maxz, z);
		}
		const double cx = 0.5 * (minx + maxx);
		const double cz = 0.5 * (minz + maxz);
		const double cy = 0.5 * (miny + maxy);
		const double radX = static_cast<double>(model_rotateX_deg) * (3.14159265358979323846 / 180.0);
		const double radY = static_cast<double>(model_rotateY_deg) * (3.14159265358979323846 / 180.0);
		const double radZ = static_cast<double>(model_rotateZ_deg) * (3.14159265358979323846 / 180.0);
		const double cX = std::cos(radX), sX = std::sin(radX);
		const double cY = std::cos(radY), sY = std::sin(radY);
		const double cZ = std::cos(radZ), sZ = std::sin(radZ);
		for (int i = 0; i < out.numberofpoints; ++i) {
			double x = static_cast<double>(out.pointlist[3 * i + 0]) - cx;
			double y = static_cast<double>(out.pointlist[3 * i + 1]) - cy;
			double z = static_cast<double>(out.pointlist[3 * i + 2]) - cz;

			// X rotation
			if (std::abs(model_rotateX_deg) > 1e-6f) {
				const double y1 = cX * y - sX * z;
				const double z1 = sX * y + cX * z;
				y = y1;
				z = z1;
			}
			// Y rotation
			if (std::abs(model_rotateY_deg) > 1e-6f) {
				const double x1 = cY * x + sY * z;
				const double z1 = -sY * x + cY * z;
				x = x1;
				z = z1;
			}
			// Z rotation
			if (std::abs(model_rotateZ_deg) > 1e-6f) {
				const double x1 = cZ * x - sZ * y;
				const double y1 = sZ * x + cZ * y;
				x = x1;
				y = y1;
			}

			out.pointlist[3 * i + 0] = static_cast<REAL>(x + cx);
			out.pointlist[3 * i + 1] = static_cast<REAL>(y + cy);
			out.pointlist[3 * i + 2] = static_cast<REAL>(z + cz);
		}
		std::cout << "[Model] rotateX(deg)=" << model_rotateX_deg
		          << " rotateY(deg)=" << model_rotateY_deg
		          << " rotateZ(deg)=" << model_rotateZ_deg
		          << " applied (about bbox center).\n";
	}
	



	Object object;
	groupNum = groupNumX * groupNumY * groupNumZ;
	object.groupNum = groupNum;
	object.groupNumX = groupNumX;
	object.groupNumY = groupNumY;
	object.groupNumZ = groupNumZ;

	if (halfYoungsEnabled) {
		int count = 0;
		for (int gi = 0; gi < groupNum; ++gi) {
			if (std::abs(effectiveYoungsForGroup(gi, youngs) - youngs) > 1e-3f) ++count;
		}
		std::cout << "[Material] half_youngs_enabled=true"
		          << ", value=" << halfYoungsValue
		          << ", axis=" << halfYoungsAxis
		          << ", side=" << halfYoungsSide
		          << ", groups=" << count << "/" << groupNum << "\n";
	}
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

	// ====================================
	// Initialize Haptic UART Interface
	// ====================================
	HapticInterface haptic;
	if (haptic_uart_enabled) {
		std::cout << "\n========================================" << std::endl;
		std::cout << "Initializing Haptic UART Interface..." << std::endl;
		
		// 1. Scan for ports
		std::vector<std::string> ports;
		try {
			if (std::filesystem::exists("/dev")) {
				for (const auto& entry : std::filesystem::directory_iterator("/dev")) {
					std::string name = entry.path().filename().string();
					// Look for macOS usbserial devices
					if (name.rfind("cu.usbserial", 0) == 0 || name.rfind("tty.usbserial", 0) == 0 || name.rfind("cu.usbmodem", 0) == 0) {
						ports.push_back(entry.path().string());
					}
				}
			}
		} catch (const std::filesystem::filesystem_error& ex) {
			std::cerr << "Error scanning /dev: " << ex.what() << std::endl;
		}
		std::sort(ports.begin(), ports.end());

		std::string selectedPort = haptic_uart_port;

		// 2. List ports and ask user
		if (ports.empty()) {
			std::cout << "No USB serial ports found (matching cu.usbserial* or cu.usbmodem*)." << std::endl;
			std::cout << "Using configured port: " << selectedPort << std::endl;
		} else {
			std::cout << "\nAvailable Serial Ports:" << std::endl;
			for (size_t i = 0; i < ports.size(); ++i) {
				std::cout << "  [" << i << "] " << ports[i] << std::endl;
			}
			std::cout << "\nSelect port number (default " << selectedPort << "): ";
			
			std::string input;
			// Use getline to handle empty Enter key
			// Clear any previous newline char left in buffer? No, initialization is early.
			if (std::getline(std::cin, input)) {
				if (!input.empty()) {
					try {
						int idx = std::stoi(input);
						if (idx >= 0 && idx < static_cast<int>(ports.size())) {
							selectedPort = ports[idx];
						} else {
							std::cout << "Invalid index, using default." << std::endl;
						}
					} catch (...) {
						std::cout << "Invalid input, using default." << std::endl;
					}
				}
			}
			std::cout << "Target Port: " << selectedPort << std::endl;
		}
		
		std::cout << "  Motor ID (Index): " << haptic_uart_motor_id << std::endl;
		std::cout << "  Motor ID (Thumb): " << haptic_uart_thumb_motor_id << std::endl;
		std::cout << "  Motor ID (Middle): " << haptic_uart_middle_motor_id << std::endl;
		std::cout << "  Motor ID (Ring): " << haptic_uart_ring_motor_id << std::endl;
		std::cout << "  Force range: " << haptic_min_force_input << " - " << haptic_max_force_input << " N" << std::endl;
		std::cout << "  PWM range: " << haptic_min_pwm_output << " - " << haptic_max_pwm_output << std::endl;
		std::cout << "  Gamma: " << haptic_gamma << " (Power Law Mapping)" << std::endl;
		std::cout << "  SlewLimiter: enabled=" << (haptic_slew_enabled ? 1 : 0)
		          << " upPWM/s=" << haptic_slew_up_pwm_per_sec
		          << " downPWM/s=" << haptic_slew_down_pwm_per_sec
		          << std::endl;
		
		haptic.setParameters(haptic_min_force_input, haptic_max_force_input, haptic_min_pwm_output, haptic_max_pwm_output, haptic_gamma);
		haptic.setSlewLimiter(haptic_slew_enabled, haptic_slew_up_pwm_per_sec, haptic_slew_down_pwm_per_sec);
		if (haptic.init(selectedPort)) {
			std::cout << "✓ Haptic interface initialized successfully!" << std::endl;
		} else {
			std::cout << "✗ Failed to initialize haptic interface" << std::endl;
		}
		std::cout << "========================================\n" << std::endl;
	}

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
			const float scale = effectiveYoungsScaleForGroup(i);
			object.groups[i].calGroupKAni(youngs1 * scale, youngs2 * scale, youngs3 * scale, poisson);
			if (i == 0) std::cout << "Using Anisotropic Stiffness Matrix (E1=" << youngs1 << ", E2=" << youngs2 << ", E3=" << youngs3 << ")\n";
		} else {
			const float E = effectiveYoungsForGroup(i, youngs);
			object.groups[i].calGroupK(E, poisson);
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
	const Eigen::Vector3f bboxExtents = bboxMax - bboxMin;
	const std::array<Eigen::Vector3f, 3> tumorPresetInits = {
		Eigen::Vector3f(-0.0228788f, 0.2075f, 0.714083f),
		Eigen::Vector3f(-0.676255f, 0.302103f, 1.0287f),
		Eigen::Vector3f(-0.629456f, 0.573538f, 0.531963f)
	};
	int tumorModeIndex = 0; // 0=OFF, 1..3 = preset index
	Eigen::Vector3f tumorPickedInit = tumorPresetInits[0];
	const Eigen::Vector3f wallMargin0 = std::max(0.0f, wallMarginBboxScale) * bboxExtents;
	const float wallXMax0 = bboxMax.x() + wallMargin0.x();
	const float wallYMin0 = bboxMin.y() - wallMargin0.y();
	const float wallYMax0 = bboxMax.y() + wallMargin0.y();
	const float wallZMin0 = bboxMin.z() - wallMargin0.z();
	const float wallZMax0 = bboxMax.z() + wallMargin0.z();

		// Material stiffness mapping in world space (for haptics + visualization).
		//
		// IMPORTANT: group division uses *adaptive quantile edges* (see divideIntoGroups in Object.cpp),
		// not uniform voxel spacing. If we use uniform bins here, the "white tumor overlay" will not
		// match where haptics thinks the hard region is.
		const int groupNx = std::max(1, groupNumX);
		const int groupNy = std::max(1, groupNumY);
		const int groupNz = std::max(1, groupNumZ);

		auto makeEdges = [](std::vector<float>& values, int bins, float minVal, float maxVal) {
			std::vector<float> edges;
			edges.reserve(static_cast<size_t>(bins) + 1);
			if (values.empty() || bins <= 0) return edges;
			std::sort(values.begin(), values.end());

			edges.push_back(minVal - 1e-4f);
			for (int i = 1; i < bins; ++i) {
				int idx = static_cast<int>(std::round(static_cast<float>(i) * static_cast<float>(values.size()) / static_cast<float>(bins)));
				idx = std::max(0, std::min(static_cast<int>(values.size()) - 1, idx));
				edges.push_back(values[static_cast<size_t>(idx)]);
			}
			edges.push_back(maxVal + 1e-4f);

			// Ensure strictly increasing to avoid upper_bound degeneracy.
			const float step = std::max(1e-5f, (maxVal - minVal) * 1e-5f);
			for (size_t i = 1; i < edges.size(); ++i) {
				if (edges[i] <= edges[i - 1]) edges[i] = edges[i - 1] + step;
			}
			return edges;
		};

		auto findBin = [](float v, const std::vector<float>& edges, int fallbackBins) {
			if (edges.size() < 2) return std::max(0, fallbackBins - 1);
			auto it = std::upper_bound(edges.begin(), edges.end(), v);
			int idx = static_cast<int>(it - edges.begin()) - 1;
			idx = std::max(0, std::min(static_cast<int>(edges.size()) - 2, idx));
			return idx;
		};

		// Recompute the adaptive edges from tetra centroids (rest/initial coordinates).
		std::vector<float> xVals, yVals, zVals;
		xVals.reserve(static_cast<size_t>(out.numberoftetrahedra));
		yVals.reserve(static_cast<size_t>(out.numberoftetrahedra));
		zVals.reserve(static_cast<size_t>(out.numberoftetrahedra));
		for (int gi = 0; gi < groupNum; ++gi) {
			Group& g = object.getGroup(gi);
			for (Tetrahedron* tet : g.tetrahedra) {
				const float cx = (tet->vertices[0]->initx + tet->vertices[1]->initx + tet->vertices[2]->initx + tet->vertices[3]->initx) * 0.25f;
				const float cy = (tet->vertices[0]->inity + tet->vertices[1]->inity + tet->vertices[2]->inity + tet->vertices[3]->inity) * 0.25f;
				const float cz = (tet->vertices[0]->initz + tet->vertices[1]->initz + tet->vertices[2]->initz + tet->vertices[3]->initz) * 0.25f;
				xVals.push_back(cx);
				yVals.push_back(cy);
				zVals.push_back(cz);
			}
		}
		const std::vector<float> groupXEdges = makeEdges(xVals, groupNx, bboxMin.x(), bboxMax.x());
		const std::vector<float> groupYEdges = makeEdges(yVals, groupNy, bboxMin.y(), bboxMax.y());
		const std::vector<float> groupZEdges = makeEdges(zVals, groupNz, bboxMin.z(), bboxMax.z());

		auto groupIndexFromWorldPoint = [&](const Eigen::Vector3f& p) -> int {
			const int gx = findBin(p.x(), groupXEdges, groupNx);
			const int gy = findBin(p.y(), groupYEdges, groupNy);
			const int gz = findBin(p.z(), groupZEdges, groupNz);
			return gz * groupNx * groupNy + gy * groupNx + gx;
		};
		auto applyTumorMode = [&](int modeIdx, bool printLog) {
			if (tumorPresetInits.empty()) return;
			const int modeCount = static_cast<int>(tumorPresetInits.size()) + 1; // + OFF
			tumorModeIndex = ((modeIdx % modeCount) + modeCount) % modeCount;
			tumorYoungsEnabled = (tumorModeIndex != 0);
			if (tumorModeIndex == 0) {
				if (printLog) std::cout << "[TumorPreset] mode=OFF\n";
				return;
			}
			const int presetIdx = tumorModeIndex - 1;
			tumorPickedInit = tumorPresetInits[static_cast<size_t>(presetIdx)];

			const float invX = 1.0f / std::max(1e-8f, bboxExtents.x());
			const float invY = 1.0f / std::max(1e-8f, bboxExtents.y());
			const float invZ = 1.0f / std::max(1e-8f, bboxExtents.z());
			tumorCenterXFrac = std::clamp((tumorPickedInit.x() - bboxMin.x()) * invX, 0.0f, 1.0f);
			tumorCenterYFrac = std::clamp((tumorPickedInit.y() - bboxMin.y()) * invY, 0.0f, 1.0f);
			tumorCenterZFrac = std::clamp((tumorPickedInit.z() - bboxMin.z()) * invZ, 0.0f, 1.0f);

			tumorCenterGroupOverrideEnabled = true;
			tumorCenterGroupX = findBin(tumorPickedInit.x(), groupXEdges, groupNx);
			tumorCenterGroupY = findBin(tumorPickedInit.y(), groupYEdges, groupNy);
			tumorCenterGroupZ = findBin(tumorPickedInit.z(), groupZEdges, groupNz);

			if (printLog) {
				std::cout << "[TumorPreset] mode=" << tumorModeIndex
				          << " idx=" << presetIdx
				          << " init=(" << tumorPickedInit.x() << "," << tumorPickedInit.y() << "," << tumorPickedInit.z() << ")"
				          << " frac=(" << tumorCenterXFrac << "," << tumorCenterYFrac << "," << tumorCenterZFrac << ")"
				          << " overrideGroup=(" << tumorCenterGroupX << "," << tumorCenterGroupY << "," << tumorCenterGroupZ << ")\n";
			}
		};
		applyTumorMode(0, true); // startup default: OFF

		auto materialScaleAtWorldPoint = [&](const Eigen::Vector3f& p) -> float {
			const float base = youngs;
			if (std::abs(base) < 1e-6f) return 1.0f;
			const int gi = groupIndexFromWorldPoint(p);
			const float eff = effectiveYoungsForGroup(gi, base);
			return eff / base;
		};
		{
			static bool printedTumorMapping = false;
			if (!printedTumorMapping && tumorYoungsEnabled) {
				const Eigen::Vector3f tumorInit(
					bboxMin.x() + tumorCenterXFrac * bboxExtents.x(),
					bboxMin.y() + tumorCenterYFrac * bboxExtents.y(),
					bboxMin.z() + tumorCenterZFrac * bboxExtents.z());
				const int gi = groupIndexFromWorldPoint(tumorInit);
				const int gx = gi % groupNx;
				const int gy = (gi / groupNx) % groupNy;
				const int gz = (gi / (groupNx * groupNy));
				const int ogi = tumorCenterGroupZ * groupNx * groupNy + tumorCenterGroupY * groupNx + tumorCenterGroupX;
				const bool centerGroupHard = (std::abs(effectiveYoungsForGroup(ogi, youngs) - youngs) > 1e-3f);
				std::cout << "[TumorMap] init=(" << tumorInit.x() << "," << tumorInit.y() << "," << tumorInit.z()
				          << ") frac=(" << tumorCenterXFrac << "," << tumorCenterYFrac << "," << tumorCenterZFrac
				          << ") group=(" << gx << "," << gy << "," << gz << ")"
				          << " overrideGroup=(" << tumorCenterGroupX << "," << tumorCenterGroupY << "," << tumorCenterGroupZ << ")"
				          << " centerHard=" << (centerGroupHard ? 1 : 0)
				          << " rFrac=" << tumorRadiusFrac << "\n";
				printedTumorMapping = true;
			}
		}

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
	std::array<Eigen::Quaternionf, kFingerCount> agentDeviceRotations; // Added for rotation mapping
	std::array<Eigen::Vector3f, kFingerCount> agentDevicePrevPositions;
		std::array<Eigen::Vector3f, kFingerCount> agentDeviceVelocities;
		std::array<Eigen::Vector3f, kFingerCount> agentProxyPositions;
		std::array<Eigen::Vector3f, kFingerCount> agentProxyVelocities;
		std::array<Eigen::Vector3f, kFingerCount> agentHomePositions;
	
		for (int fi = 0; fi < kFingerCount; ++fi) {
			const Eigen::Vector3f p = agentHandHomeAnchor + agentHandFingerOffsets[static_cast<size_t>(fi)];
		agentHomePositions[static_cast<size_t>(fi)] = p;
		agentDevicePositions[static_cast<size_t>(fi)] = p;
		agentDeviceRotations[static_cast<size_t>(fi)] = Eigen::Quaternionf::Identity();
		agentDevicePrevPositions[static_cast<size_t>(fi)] = p;
			agentDeviceVelocities[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
			agentProxyPositions[static_cast<size_t>(fi)] = p;
			agentProxyVelocities[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
		}
	
					std::array<Eigen::Vector3f, kFingerCount> agentLastDeviceForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentLastContactForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentLastCouplingForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentFilteredDeviceForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentFilteredContactForcesN;
					std::array<Eigen::Vector3f, kFingerCount> agentLastContactNormalsIn;
					std::array<Eigen::Vector3f, kFingerCount> agentFilteredContactNormalsIn;
					std::array<int, kFingerCount> agentLastContactCounts{};
					std::array<float, kFingerCount> agentLastContactPenetrations{};
					std::array<int, kFingerCount> agentLastActiveContactTriangle{};
					std::array<bool, kFingerCount> agentGripActive{};
					std::array<int, kFingerCount> agentGripTriangle{};
					std::array<Eigen::Vector3f, kFingerCount> agentGripBary{};
						agentLastDeviceForcesN.fill(Eigen::Vector3f::Zero());
						agentLastContactForcesN.fill(Eigen::Vector3f::Zero());
						agentLastCouplingForcesN.fill(Eigen::Vector3f::Zero());
						agentFilteredDeviceForcesN.fill(Eigen::Vector3f::Zero());
						agentFilteredContactForcesN.fill(Eigen::Vector3f::Zero());
						agentLastContactNormalsIn.fill(Eigen::Vector3f::Zero());
						agentFilteredContactNormalsIn.fill(Eigen::Vector3f::Zero());
						agentLastActiveContactTriangle.fill(-1);
						agentGripActive.fill(false);
						agentGripTriangle.fill(-1);
						agentGripBary.fill(Eigen::Vector3f::Zero());
		
#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
		LeapCTracker leapTracker;
		// Start with Leap input enabled by default (can still be toggled with 'B').
		bool leapUseInput = !pipelineProfiler.active;
		bool leapMappingCalibrated = false;
		Eigen::Vector3f leapCenterMm = Eigen::Vector3f::Zero();
		// Keep the Leap->world mapping stable: anchor to the initial "home" pose.
		// Recenter (if desired) is handled explicitly via the existing recenter action.
		Eigen::Vector3f leapAnchorWorld = agentHomePositions[static_cast<size_t>(kIndexFinger)];
		// User-tunable world-space translation offsets (for aligning real/virtual without changing the calibration math).
		// Default offsets requested by user (printed from runtime).
		// LEFT : (0.000000,-0.990932,0.990932)
		// RIGHT: (0.495466,-0.000000,0.990932)
		Eigen::Vector3f leapLeftWorldOffset(0.0f, -0.195466f, 0.590932f);
		Eigen::Vector3f leapRightWorldOffset(0.495466f, -0.1f, 0.790932f);
		enum class LeapOffsetTarget { Right, Left };
		LeapOffsetTarget leapOffsetTarget = LeapOffsetTarget::Right;
		std::array<Eigen::Vector3f, kFingerCount> leapLatestTipsMm;
		leapLatestTipsMm.fill(Eigen::Vector3f::Zero());
		double leapLatestTimeSec = -1.0;

		// Left-hand (non-haptic) interaction: track tips + palm, map to world, then collide as capsules.
		bool leapLeftMappingCalibrated = false;
		Eigen::Vector3f leapLeftCenterMm = Eigen::Vector3f::Zero();
		const Eigen::Vector3f leapLeftHomeAnchor(
			bboxCenter.x() - 0.25f * bboxDiag,
			bboxMax.y() + agentSphere.radius * 1.5f,
			bboxCenter.z());
		Eigen::Vector3f leapLeftAnchorWorld = leapLeftHomeAnchor;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftLatestTipsMm;
		leapLeftLatestTipsMm.fill(Eigen::Vector3f::Zero());
		std::array<Eigen::Vector3f, kFingerCount> leapLeftLatestPrevMm;
		leapLeftLatestPrevMm.fill(Eigen::Vector3f::Zero());
		Eigen::Vector3f leapLeftLatestPalmMm = Eigen::Vector3f::Zero();
		double leapLeftLatestTimeSec = -1.0;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftWorldTips;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftWorldPrevTips;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftWorldVelTips;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftWorldPrevJoints;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftWorldPrevPrevJoints;
		std::array<Eigen::Vector3f, kFingerCount> leapLeftWorldVelPrevJoints;
		Eigen::Vector3f leapLeftWorldPalm = leapLeftHomeAnchor + leapLeftWorldOffset;
		Eigen::Vector3f leapLeftWorldPrevPalm = leapLeftHomeAnchor + leapLeftWorldOffset;
		Eigen::Vector3f leapLeftWorldVelPalm = Eigen::Vector3f::Zero();
		for (int fi = 0; fi < kFingerCount; ++fi) {
			const Eigen::Vector3f p = leapLeftHomeAnchor + agentHandFingerOffsets[static_cast<size_t>(fi)] + leapLeftWorldOffset;
			leapLeftWorldTips[static_cast<size_t>(fi)] = p;
			leapLeftWorldPrevTips[static_cast<size_t>(fi)] = p;
			leapLeftWorldVelTips[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
			leapLeftWorldPrevJoints[static_cast<size_t>(fi)] = p;
			leapLeftWorldPrevPrevJoints[static_cast<size_t>(fi)] = p;
			leapLeftWorldVelPrevJoints[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
		}
		// Keep a "wanted" toggle so left hand comes back automatically after Leap is toggled back ON.
		static bool leftHandCapsulesWanted = leftHandEnabled;
		static bool leftHandCapsulesEnabledRuntime = leftHandEnabled;
		leftHandCapsulesEnabledRuntime = leftHandCapsulesWanted && leapUseInput;
		if (leapUseInput && !leapTracker.init()) {
			std::cerr << "[LeapC] init failed; disabling Leap input.\n";
			leapUseInput = false;
			leftHandCapsulesEnabledRuntime = false;
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

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
	const int leftHandSamplesClamped = std::clamp(leftHandCapsuleSamples, 2, 12);
	const int leftHandTotalSpheres = kFingerCount * leftHandSamplesClamped;
	const float leftHandSphereRadius = std::max(1e-6f, leftHandCapsuleRadiusBboxScale * bboxDiag);
	const float leftHandCapsuleLength = std::max(0.0f, leftHandCapsuleLengthBboxScale * bboxDiag);

	std::vector<Eigen::Vector3f> leftHandDevicePositions(static_cast<size_t>(leftHandTotalSpheres), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> leftHandDeviceVelocities(static_cast<size_t>(leftHandTotalSpheres), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> leftHandProxyPositions(static_cast<size_t>(leftHandTotalSpheres), Eigen::Vector3f::Zero());
	std::vector<Eigen::Vector3f> leftHandProxyVelocities(static_cast<size_t>(leftHandTotalSpheres), Eigen::Vector3f::Zero());
	std::vector<int> leftHandActiveContactTriangle(static_cast<size_t>(leftHandTotalSpheres), -1);

	// Initialize left-hand capsule spheres around the home anchor.
	for (int fi = 0; fi < kFingerCount; ++fi) {
		const Eigen::Vector3f tip = leapLeftWorldTips[static_cast<size_t>(fi)];
		Eigen::Vector3f dir = tip - leapLeftWorldPalm;
		const float dlen = dir.norm();
		if (dlen > 1e-8f) dir /= dlen;
		else dir = -Eigen::Vector3f::UnitY();

		const Eigen::Vector3f base = tip - dir * leftHandCapsuleLength;
		for (int si = 0; si < leftHandSamplesClamped; ++si) {
			const float t = (leftHandSamplesClamped > 1) ? (static_cast<float>(si) / static_cast<float>(leftHandSamplesClamped - 1)) : 1.0f;
			const Eigen::Vector3f p = base + t * (tip - base);
			const size_t idx = static_cast<size_t>(fi * leftHandSamplesClamped + si);
			leftHandDevicePositions[idx] = p;
			leftHandDeviceVelocities[idx].setZero();
			leftHandProxyPositions[idx] = p;
			leftHandProxyVelocities[idx].setZero();
			leftHandActiveContactTriangle[idx] = -1;
		}
	}
#endif

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
			// Abdominal cavity wall data (built from rest-pose surface triangles; used later in the sim loop + rendering).
			std::vector<char> cavityTriangleEnabled;
			std::vector<int> cavitySurfacePhysIds;
			std::vector<int> cavityActiveTriangleByPhysId;
			std::vector<Eigen::Vector3f> cavityVertexNormalByPhysId;
			float cavityGapWorld = 0.0f;
			int cavityOpenAxis = 1;
			float cavityOpenLoWorld = bboxMax.y();
			float cavityOpenHiWorld = bboxMax.y();
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
			// We also build a surface-vertex mask used by agent surface-vertex contact and optional "suspension"
			// ligaments (surface patch springs).
			std::vector<char> isSurfacePhys;
			isSurfacePhys.assign(physRep.size(), 0);
			const bool buildSurfacePhys = (!physRep.empty()) && (agentUseSurfaceTriangles || agentUseSurfaceVertices || suspensionEnabled);
			if (buildSurfacePhys) {
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

			if (agentUseSurfaceTriangles) {
				agentContactTriangles.reserve(faces.size());
				agentContactTrianglePhysIds.reserve(faces.size());
			}
				for (const auto& kv : faces) {
					const FaceRec& rec = kv.second;
					if (rec.count != 1) continue; // interior face (shared by two tets in phys space)
					if (!rec.a || !rec.b || !rec.c) continue;
					if (agentUseSurfaceTriangles) {
						agentContactTriangles.push_back(AgentTriangle{rec.a, rec.b, rec.c, rec.opp});
						agentContactTrianglePhysIds.push_back(rec.ids);
					}
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
				if (agentUseSurfaceTriangles && !agentContactTrianglePhysIds.empty()) {
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

				// ------------------------------------------------------------
				// Abdominal cavity wall (static, rest-pose liver-shaped boundary)
				// ------------------------------------------------------------
				// We derive a cavity boundary from the *rest pose* surface triangles, inflated outward by a small
				// gap. One side is left open by disabling those triangles (models the surgical exposure).
				{
					cavityGapWorld = std::max(0.0f, cavity_gap_bboxScale) * bboxDiag;
					cavityOpenAxis = std::clamp(cavity_open_axis, 0, 2);
					const int openSide = (cavity_open_side >= 0) ? 1 : -1;

					const Eigen::Vector3f extents = bboxMax - bboxMin;
					const float axisExtent = std::max(1e-6f, extents[cavityOpenAxis]);
					const float openFrac = std::clamp(cavity_open_frac, 0.0f, 1.0f);

					const float axisMin = bboxMin[cavityOpenAxis];
					const float axisMax = bboxMax[cavityOpenAxis];
					if (openSide > 0) {
						// Open near bboxMax: [openLo, axisMax]
						cavityOpenLoWorld = axisMax - openFrac * axisExtent;
						cavityOpenHiWorld = axisMax + 1e-6f;
					} else {
						// Open near bboxMin: [axisMin, openHi]
						cavityOpenLoWorld = axisMin - 1e-6f;
						cavityOpenHiWorld = axisMin + openFrac * axisExtent;
					}

					// Surface phys ids (only these vertices can collide with the cavity).
					cavitySurfacePhysIds.reserve(isSurfacePhys.size());
					for (int id = 0; id < static_cast<int>(isSurfacePhys.size()); ++id) {
						if (!isSurfacePhys[static_cast<size_t>(id)]) continue;
						cavitySurfacePhysIds.push_back(id);
					}

					// Smoothed per-vertex outward normals (rest pose) for cavity visualization.
					// This avoids a faceted "broken triangles" look from per-face normal offsets.
					cavityVertexNormalByPhysId.assign(agentVerticesByPhysId.size(), Eigen::Vector3f::Zero());
					if (!agentContactTrianglePhysIds.empty() &&
					    agentContactTrianglePhysIds.size() == agentContactTriangles.size()) {
						for (int ti = 0; ti < static_cast<int>(agentContactTriangles.size()); ++ti) {
							const AgentTriangle& tri = agentContactTriangles[static_cast<size_t>(ti)];
							if (!tri.a || !tri.b || !tri.c) continue;
							const Eigen::Vector3f a(tri.a->initx, tri.a->inity, tri.a->initz);
							const Eigen::Vector3f b(tri.b->initx, tri.b->inity, tri.b->initz);
							const Eigen::Vector3f c(tri.c->initx, tri.c->inity, tri.c->initz);
							Eigen::Vector3f n = Eigen::Vector3f::Zero();
							if (!outwardNormalForTriangleInit(tri, a, b, c, &n)) continue;
							const float area = 0.5f * ((b - a).cross(c - a)).norm();
							const Eigen::Vector3f nw = n * std::max(1e-8f, area);
							const auto& ids = agentContactTrianglePhysIds[static_cast<size_t>(ti)];
							for (int k = 0; k < 3; ++k) {
								const int pid = ids[static_cast<size_t>(k)];
								if (pid < 0 || pid >= static_cast<int>(cavityVertexNormalByPhysId.size())) continue;
								cavityVertexNormalByPhysId[static_cast<size_t>(pid)] += nw;
							}
						}
						for (size_t pid = 0; pid < cavityVertexNormalByPhysId.size(); ++pid) {
							Eigen::Vector3f& n = cavityVertexNormalByPhysId[pid];
							const float l2 = n.squaredNorm();
							if (l2 > 1e-20f) n /= std::sqrt(l2);
						}
					}

					// Triangle enable mask: disable the open side triangles so there's no "front wall".
					cavityTriangleEnabled.assign(agentContactTriangles.size(), 1);
					const float margin = 0.02f * axisExtent; // slightly bigger opening than the vertex gate
					float disableLo = cavityOpenLoWorld;
					float disableHi = cavityOpenHiWorld;
					if (openSide > 0) disableLo -= margin;
					else disableHi += margin;
					for (int ti = 0; ti < static_cast<int>(agentContactTriangles.size()); ++ti) {
						const AgentTriangle& tri = agentContactTriangles[static_cast<size_t>(ti)];
						if (!tri.a || !tri.b || !tri.c) { cavityTriangleEnabled[static_cast<size_t>(ti)] = 0; continue; }
						const Eigen::Vector3f a(tri.a->initx, tri.a->inity, tri.a->initz);
						const Eigen::Vector3f b(tri.b->initx, tri.b->inity, tri.b->initz);
						const Eigen::Vector3f c(tri.c->initx, tri.c->inity, tri.c->initz);
						Eigen::Vector3f n = Eigen::Vector3f::Zero();
						if (!outwardNormalForTriangleInit(tri, a, b, c, &n)) { cavityTriangleEnabled[static_cast<size_t>(ti)] = 0; continue; }
						const Eigen::Vector3f centroid = (a + b + c) * (1.0f / 3.0f);
						// Disable triangles on the open side (by position only).
						const float v = centroid[cavityOpenAxis];
						if (v >= std::min(disableLo, disableHi) && v <= std::max(disableLo, disableHi)) {
							cavityTriangleEnabled[static_cast<size_t>(ti)] = 0;
							continue;
						}
					}

					// Active triangle cache per physical vertex id (for fast local neighbor walk).
					cavityActiveTriangleByPhysId.assign(agentVerticesByPhysId.size(), 0);
					int firstEnabled = -1;
					for (int ti = 0; ti < static_cast<int>(cavityTriangleEnabled.size()); ++ti) {
						if (cavityTriangleEnabled[static_cast<size_t>(ti)]) { firstEnabled = ti; break; }
					}
					if (firstEnabled < 0) firstEnabled = 0;
					for (size_t i = 0; i < cavityActiveTriangleByPhysId.size(); ++i) cavityActiveTriangleByPhysId[i] = firstEnabled;

					// Better initialization: for each surface vertex id, seed its active triangle with an incident triangle
					// on the surface so the neighbor-walk starts locally (prevents "no collision" due to a bad seed).
					if (!agentContactTrianglePhysIds.empty() && agentContactTrianglePhysIds.size() == agentContactTriangles.size()) {
						std::vector<int> incident(static_cast<size_t>(agentVerticesByPhysId.size()), -1);
						for (int ti = 0; ti < static_cast<int>(agentContactTrianglePhysIds.size()); ++ti) {
							if (!cavityTriangleEnabled.empty() && !cavityTriangleEnabled[static_cast<size_t>(ti)]) continue;
							const auto& ids = agentContactTrianglePhysIds[static_cast<size_t>(ti)];
							for (int k = 0; k < 3; ++k) {
								const int pid = ids[static_cast<size_t>(k)];
								if (pid < 0 || pid >= static_cast<int>(incident.size())) continue;
								if (incident[static_cast<size_t>(pid)] == -1) incident[static_cast<size_t>(pid)] = ti;
							}
						}
						for (int pid : cavitySurfacePhysIds) {
							if (pid < 0 || pid >= static_cast<int>(incident.size())) continue;
							const int ti = incident[static_cast<size_t>(pid)];
							if (ti >= 0) cavityActiveTriangleByPhysId[static_cast<size_t>(pid)] = ti;
						}
					}

					if (cavity_enabled) {
						std::cout << "[Cavity] enabled=1 gap=" << cavityGapWorld
						          << " openAxis=" << cavityOpenAxis
						          << " openLo=" << cavityOpenLoWorld
						          << " openHi=" << cavityOpenHiWorld
						          << " surfaceVerts=" << cavitySurfacePhysIds.size()
						          << " tris=" << agentContactTriangles.size()
						          << std::endl;
					}
				}

				}

					if (agentUseSurfaceVertices && !isSurfacePhys.empty()) {
						for (int id = 0; id < static_cast<int>(isSurfacePhys.size()); ++id) {
							if (!isSurfacePhys[static_cast<size_t>(id)]) continue;
						if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
						Vertex* rep = physRep[static_cast<size_t>(id)];
						if (!rep) continue;
						agentContactVertices.push_back(rep);
						agentContactVertexPhysIds.push_back(id);
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

			// ------------------ Optional: 3-point suspension ligaments (surface patches -> wall anchors)
			struct SuspensionSpring {
				bool enabled = false;
				const char* name = "";
				Eigen::Vector3f anchorWorld = Eigen::Vector3f::Zero();
				Eigen::Vector3f centerRest = Eigen::Vector3f::Zero();
				float radius = 0.0f;
				float k = 0.0f;
				float damping = 0.0f;
				float maxAccel = 0.0f;
				std::vector<int> physIds;
				std::vector<Eigen::Vector3f> restOffsetFromAnchor; // desired = anchor + offset
				std::vector<float> weights;                        // [0..1] falloff in patch
			};
			std::vector<SuspensionSpring> suspensions;
			std::vector<int> customFixedPhysIds;
			if (suspensionEnabled && !physRep.empty() && !agentVerticesByPhysId.empty()) {
				auto initPosOf = [&](int id) -> Eigen::Vector3f {
					if (id < 0 || id >= static_cast<int>(physRep.size())) return Eigen::Vector3f::Zero();
					const Vertex* v = physRep[static_cast<size_t>(id)];
					return v ? Eigen::Vector3f(v->initx, v->inity, v->initz) : Eigen::Vector3f::Zero();
				};
				auto avgInitOf = [&](const std::vector<int>& ids) -> Eigen::Vector3f {
					Eigen::Vector3f c = Eigen::Vector3f::Zero();
					int n = 0;
					for (int id : ids) {
						if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
						const Vertex* v = physRep[static_cast<size_t>(id)];
						if (!v) continue;
						c += Eigen::Vector3f(v->initx, v->inity, v->initz);
						++n;
					}
					if (n > 0) c /= static_cast<float>(n);
					return c;
				};
				auto selectWithinRadius = [&](const std::vector<int>& candidates, const Eigen::Vector3f& center, float radius) -> std::vector<int> {
					std::vector<int> out;
					const float r2 = radius * radius;
					for (int id : candidates) {
						if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
						const Vertex* v = physRep[static_cast<size_t>(id)];
						if (!v) continue;
						const Eigen::Vector3f p(v->initx, v->inity, v->initz);
						if ((p - center).squaredNorm() <= r2) out.push_back(id);
					}
					return out;
				};
				auto makeWeights = [&](const std::vector<int>& ids, const Eigen::Vector3f& center, float radius) -> std::vector<float> {
					std::vector<float> w;
					w.reserve(ids.size());
					const float invR = (radius > 1e-8f) ? (1.0f / radius) : 0.0f;
					for (int id : ids) {
						if (id < 0 || id >= static_cast<int>(physRep.size())) {
							w.push_back(0.0f);
							continue;
						}
						const Vertex* v = physRep[static_cast<size_t>(id)];
						if (!v) {
							w.push_back(0.0f);
							continue;
						}
						const Eigen::Vector3f p(v->initx, v->inity, v->initz);
						float t = invR > 0.0f ? (p - center).norm() * invR : 0.0f;
						t = std::clamp(t, 0.0f, 1.0f);
						float ww = 1.0f - t;
						ww = ww * ww;
						w.push_back(ww);
					}
					return w;
				};

				// Surface phys ids (fallback to all phys if surface is unavailable).
				std::vector<int> surfaceIds;
				surfaceIds.reserve(physRep.size());
				if (!isSurfacePhys.empty()) {
					for (int id = 0; id < static_cast<int>(isSurfacePhys.size()); ++id) {
						if (isSurfacePhys[static_cast<size_t>(id)]) surfaceIds.push_back(id);
					}
				}
				if (surfaceIds.empty()) {
					surfaceIds.reserve(physRep.size());
					for (int id = 0; id < static_cast<int>(physRep.size()); ++id) surfaceIds.push_back(id);
				}

				const float radius = std::max(1e-6f, std::max(0.0f, susp_patchRadiusBboxFrac) * bboxDiag);
				const float topCut = bboxMax.y() - std::clamp(susp_topSliceFrac, 0.0f, 1.0f) * bboxExtents.y();
				const float backCut = bboxMin.z() + std::clamp(susp1_backSliceFrac, 0.0f, 1.0f) * bboxExtents.z();
				const float top1Cut = bboxMax.y() - std::clamp(susp1_topSliceFrac, 0.0f, 1.0f) * bboxExtents.y();
				const float sideFracClamped = std::clamp(susp_sideFrac, 0.0f, 0.5f);

				auto collect = [&](auto&& pred) -> std::vector<int> {
					std::vector<int> ids;
					for (int id : surfaceIds) {
						if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
						const Vertex* v = physRep[static_cast<size_t>(id)];
						if (!v) continue;
						if (pred(*v)) ids.push_back(id);
					}
					return ids;
				};

				// Support 2/3: top slice, left/right endpoints.
				const std::vector<int> candTop = collect([&](const Vertex& v) { return v.inity >= topCut; });
				float topXMin = std::numeric_limits<float>::infinity();
				float topXMax = -std::numeric_limits<float>::infinity();
				for (int id : candTop) {
					const Vertex* v = physRep[static_cast<size_t>(id)];
					if (!v) continue;
					topXMin = std::min(topXMin, v->initx);
					topXMax = std::max(topXMax, v->initx);
				}
				const float sideCutLeft = bboxMin.x() + sideFracClamped * bboxExtents.x();
				const float sideCutRight = bboxMax.x() - sideFracClamped * bboxExtents.x();
				std::vector<int> seedLeft = collect([&](const Vertex& v) { return v.inity >= topCut && v.initx <= sideCutLeft; });
				std::vector<int> seedRight = collect([&](const Vertex& v) { return v.inity >= topCut && v.initx >= sideCutRight; });
				if (seedLeft.empty() && !candTop.empty()) {
					seedLeft = candTop;
					std::sort(seedLeft.begin(), seedLeft.end(), [&](int a, int b) { return initPosOf(a).x() < initPosOf(b).x(); });
					if (seedLeft.size() > 64) seedLeft.resize(64);
				}
				if (seedRight.empty() && !candTop.empty()) {
					seedRight = candTop;
					std::sort(seedRight.begin(), seedRight.end(), [&](int a, int b) { return initPosOf(a).x() > initPosOf(b).x(); });
					if (seedRight.size() > 64) seedRight.resize(64);
				}
				const Eigen::Vector3f center2 = avgInitOf(seedLeft);
				const Eigen::Vector3f center3 = avgInitOf(seedRight);
				std::vector<int> ids2 = selectWithinRadius(candTop.empty() ? seedLeft : candTop, center2, radius);
				std::vector<int> ids3 = selectWithinRadius(candTop.empty() ? seedRight : candTop, center3, radius);
				if (ids2.empty()) ids2 = seedLeft;
				if (ids3.empty()) ids3 = seedRight;

				// Support 1: posterior-superior patch (back + top), bias to one side if possible.
				std::vector<int> cand1 = collect([&](const Vertex& v) { return v.inity >= top1Cut && v.initz <= backCut; });
				if (cand1.empty()) {
					// Relax in case the initial orientation differs.
					cand1 = collect([&](const Vertex& v) { return v.initz <= backCut; });
					if (cand1.empty()) cand1 = collect([&](const Vertex& v) { return v.inity >= top1Cut; });
				}
				float xMin1 = std::numeric_limits<float>::infinity();
				float xMax1 = -std::numeric_limits<float>::infinity();
				for (int id : cand1) {
					const Vertex* v = physRep[static_cast<size_t>(id)];
					if (!v) continue;
					xMin1 = std::min(xMin1, v->initx);
					xMax1 = std::max(xMax1, v->initx);
				}
				const float xMid1 = 0.5f * (xMin1 + xMax1);
				std::vector<int> cand1Left;
				cand1Left.reserve(cand1.size());
				for (int id : cand1) {
					const Vertex* v = physRep[static_cast<size_t>(id)];
					if (!v) continue;
					if (v->initx <= xMid1) cand1Left.push_back(id);
				}
				const std::vector<int>& seed1 = (cand1Left.size() >= 12) ? cand1Left : cand1;
				const Eigen::Vector3f center1 = avgInitOf(seed1);
				std::vector<int> ids1 = selectWithinRadius(seed1, center1, radius);
				if (ids1.empty()) ids1 = seed1;

				auto buildSusp = [&](const char* name,
					bool enabled,
					const Eigen::Vector3f& anchorWorld,
					const Eigen::Vector3f& centerRest,
					float k,
					float damping,
					float maxAccel,
					const std::vector<int>& ids) -> SuspensionSpring {
					SuspensionSpring s;
					s.enabled = enabled;
					s.name = name;
					s.anchorWorld = anchorWorld;
					s.centerRest = centerRest;
					s.radius = radius;
					s.k = std::max(0.0f, k);
					s.damping = std::max(0.0f, damping);
					s.maxAccel = maxAccel;
					s.physIds = ids;
					s.restOffsetFromAnchor.reserve(ids.size());
					for (int id : ids) {
						s.restOffsetFromAnchor.push_back(initPosOf(id) - anchorWorld);
					}
					s.weights = makeWeights(ids, centerRest, radius);
					return s;
				};

				const Eigen::Vector3f anchor1(center1.x(), center1.y(), wallZMin0);         // project to back wall
				const Eigen::Vector3f anchor2(center2.x(), wallYMax0, center2.z());         // project to top wall
				const Eigen::Vector3f anchor3(center3.x(), wallYMax0, center3.z());         // project to top wall

				suspensions.reserve(3);
				suspensions.push_back(buildSusp("susp1_posterior_superior", susp1Enabled, anchor1, center1, susp1_k, susp1_damping, susp1_maxAccel, ids1));
				suspensions.push_back(buildSusp("susp2_diaphragm_left", susp2Enabled, anchor2, center2, susp2_k, susp2_damping, susp2_maxAccel, ids2));
				suspensions.push_back(buildSusp("susp3_diaphragm_right", susp3Enabled, anchor3, center3, susp3_k, susp3_damping, susp3_maxAccel, ids3));

				for (const auto& s : suspensions) {
					if (!s.enabled) continue;
					std::cout << "[Suspension] " << s.name
							  << " physIds=" << s.physIds.size()
							  << " centerRest=" << s.centerRest.transpose()
							  << " anchor=" << s.anchorWorld.transpose()
							  << " r=" << s.radius
							  << " k=" << s.k << " c=" << s.damping << " maxA=" << s.maxAccel
							  << "\n";
				}
				if (!suspensions.empty()) {
					std::cout << "[Suspension] Visual: press 'L' to toggle ligament lines, 'K' to toggle patch points, or use UI buttons.\n";
				} else if (!suspensionEnabled) {
					std::cout << "[Suspension] Disabled by parameters (suspension_enabled=false).\n";
				}
			}

			// Custom fixed points (from picked init coords): hard pin directly, no spring.
			if (!physRep.empty() && !agentVerticesByPhysId.empty()) {
				auto initPosOf = [&](int id) -> Eigen::Vector3f {
					if (id < 0 || id >= static_cast<int>(physRep.size())) return Eigen::Vector3f::Zero();
					const Vertex* v = physRep[static_cast<size_t>(id)];
					return v ? Eigen::Vector3f(v->initx, v->inity, v->initz) : Eigen::Vector3f::Zero();
				};
				std::vector<int> surfaceIds;
				if (!isSurfacePhys.empty()) {
					for (int id = 0; id < static_cast<int>(isSurfacePhys.size()); ++id) {
						if (isSurfacePhys[static_cast<size_t>(id)]) surfaceIds.push_back(id);
					}
				}
				if (surfaceIds.empty()) {
					for (int id = 0; id < static_cast<int>(physRep.size()); ++id) surfaceIds.push_back(id);
				}

				const char* customPickedLog = R"PICK(
[PickSurface] world=(-0.0965648,0.00278592,0.492334) init=(-0.0967717,0.000205548,0.498429)
[PickSurface] world=(-0.144759,-0.0313461,0.505293) init=(-0.142718,-0.0267483,0.493113)
[PickSurface] world=(-0.155409,-0.0545921,0.531068) init=(-0.152867,-0.0453267,0.523024)
[PickSurface] world=(-0.157127,-0.0722063,0.574474) init=(-0.144172,-0.0485627,0.581415)
[PickSurface] world=(-0.134041,-0.0429761,0.574446) init=(-0.144172,-0.0485627,0.581415)
[PickSurface] world=(-0.0871722,-0.0576382,0.56918) init=(-0.0929277,-0.0516177,0.568762)
[PickSurface] world=(-0.09034,-0.0666273,0.625679) init=(-0.0983166,-0.0550954,0.611615)
[PickSurface] world=(-0.130429,-0.0562255,0.647734) init=(-0.144543,-0.0560426,0.663617)
[PickSurface] world=(-0.160635,-0.0824554,0.646754) init=(-0.16342,-0.0860656,0.624649)
[PickSurface] world=(-0.169628,-0.0983841,0.647553) init=(-0.174348,-0.107358,0.649195)
[PickSurface] world=(-0.168719,-0.0907674,0.699559) init=(-0.173357,-0.0967337,0.694634)
[PickSurface] world=(-0.158396,-0.0792155,0.705177) init=(-0.159584,-0.0844945,0.687114)
[PickSurface] world=(-0.125207,-0.0612001,0.708542) init=(-0.144821,-0.0616311,0.725033)
[PickSurface] world=(-0.108326,-0.0608027,0.703454) init=(-0.106505,-0.0603796,0.676726)
[PickSurface] world=(-0.109067,-0.0610399,0.749351) init=(-0.0959499,-0.0608781,0.737485)
[PickSurface] world=(-0.146307,-0.0610757,0.766037) init=(-0.149353,-0.0604804,0.788854)
[PickSurface] world=(-0.164372,-0.0761411,0.766496) init=(-0.164747,-0.0750121,0.785856)
[PickSurface] world=(-0.176329,-0.0900517,0.767721) init=(-0.174418,-0.080652,0.787438)
[PickSurface] world=(-0.176541,-0.0816708,0.803502) init=(-0.17546,-0.0780926,0.810396)
[PickSurface] world=(-0.150915,-0.0605419,0.804738) init=(-0.151336,-0.059977,0.816778)
[PickSurface] world=(-0.13804,-0.0602634,0.815112) init=(-0.144894,-0.0595607,0.817867)
[PickSurface] world=(-0.143056,-0.0562246,0.864294) init=(-0.153716,-0.0593725,0.850302)
[PickSurface] world=(-0.186098,-0.118672,0.843675) init=(-0.18462,-0.119593,0.833234)
[PickSurface] world=(-0.0380273,0.241718,0.710116) init=(-0.029371,0.246951,0.710334)
[PickSurface] world=(-0.0516275,0.206,0.712046) init=(-0.083719,0.200556,0.710326)
[PickSurface] world=(-0.0836784,0.167434,0.730461) init=(-0.083719,0.200556,0.710326)
[PickSurface] world=(-0.128665,0.148411,0.742483) init=(-0.150843,0.124165,0.757786)
[PickSurface] world=(-0.144542,0.156734,0.744863) init=(-0.150843,0.124165,0.757786)
[PickSurface] world=(-0.201608,0.170656,0.757085) init=(-0.207854,0.143176,0.758615)
[PickSurface] world=(-0.245601,0.182722,0.767915) init=(-0.213423,0.22121,0.766726)
[PickSurface] world=(-0.231164,0.139107,0.762225) init=(-0.207854,0.143176,0.758615)
[PickSurface] world=(-0.185699,0.10646,0.774855) init=(-0.200858,0.0839479,0.787144)
[PickSurface] world=(-0.0716804,0.277443,0.707074) init=(-0.0385488,0.273562,0.707462)
[PickSurface] world=(-0.111013,0.258481,0.709598) init=(-0.123586,0.259053,0.711175)
[PickSurface] world=(-0.141946,0.240347,0.730652) init=(-0.123586,0.259053,0.711175)
[PickSurface] world=(-0.15871,0.257845,0.74473) init=(-0.179991,0.250408,0.754452)
[PickSurface] world=(-0.0254398,0.303754,0.750137) init=(-0.0109793,0.314519,0.764445)
[PickSurface] world=(-0.0459595,0.345244,0.790106) init=(-0.0368805,0.34058,0.790013)
[PickSurface] world=(-0.0716858,0.362113,0.790257) init=(-0.104639,0.351918,0.784955)
[PickSurface] world=(-0.0968835,0.402296,0.758248) init=(-0.0978991,0.410127,0.760405)
[PickSurface] world=(-0.117338,0.402114,0.780374) init=(-0.10678,0.390714,0.77517)
[PickSurface] world=(-0.187215,0.420419,0.823067) init=(-0.187393,0.428378,0.823521)
[PickSurface] world=(-0.228547,0.389417,0.809667) init=(-0.222362,0.383717,0.81033)
[PickSurface] world=(-0.285574,0.360058,0.798928) init=(-0.310133,0.347925,0.794222)
[PickSurface] world=(-0.300073,0.344516,0.796817) init=(-0.310133,0.347925,0.794222)
[PickSurface] world=(-0.34755,0.284494,0.800716) init=(-0.366672,0.276287,0.803379)
[PickSurface] world=(-0.336042,0.188297,0.786716) init=(-0.312663,0.202983,0.785249)
[PickSurface] world=(-0.265086,0.155889,0.768274) init=(-0.29347,0.131763,0.7702)
[PickSurface] world=(-0.188096,0.281936,0.785407) init=(-0.197622,0.28437,0.796499)
[PickSurface] world=(-0.174257,0.323106,0.803493) init=(-0.147171,0.34107,0.796577)
[PickSurface] world=(-0.109982,0.330848,0.743016) init=(-0.103671,0.343732,0.745723)
[PickSurface] world=(-0.104868,0.238812,0.710171) init=(-0.123586,0.259053,0.711175)
[PickSurface] world=(-0.170239,-0.411634,0.686305) init=(-0.16182,-0.382352,0.66472)
[PickSurface] world=(-0.226804,-0.521128,0.679728) init=(-0.231647,-0.5304,0.682336)
[PickSurface] world=(-0.298225,-0.586407,0.704004) init=(-0.317237,-0.597732,0.710456)
[PickSurface] world=(-0.3833,-0.620514,0.706566) init=(-0.381766,-0.621774,0.713375)
[PickSurface] world=(-0.42783,-0.633272,0.728507) init=(-0.419691,-0.634447,0.748363)
[PickSurface] world=(-0.477496,-0.644376,0.758929) init=(-0.479892,-0.63987,0.72466)
[PickSurface] world=(-0.471416,-0.645746,0.775855) init=(-0.478945,-0.649746,0.800979)
[PickSurface] world=(-0.451451,-0.645689,0.796979) init=(-0.478945,-0.649746,0.800979)
[PickSurface] world=(-0.403847,-0.635889,0.782479) init=(-0.419691,-0.634447,0.748363)
[PickSurface] world=(-0.232089,-0.523223,0.769985) init=(-0.234396,-0.534477,0.738218)
[PickSurface] world=(-0.169304,-0.390207,0.763167) init=(-0.161948,-0.370739,0.752697)
[PickSurface] world=(-0.169769,-0.370532,0.795784) init=(-0.179492,-0.407064,0.806969)
[PickSurface] world=(-0.193388,-0.41763,0.838631) init=(-0.198699,-0.422339,0.850319)
[PickSurface] world=(-0.267204,-0.544191,0.850239) init=(-0.277363,-0.54856,0.872278)
[PickSurface] world=(-0.33657,-0.612446,0.851767) init=(-0.343037,-0.620687,0.829255)
[PickSurface] world=(-0.396694,-0.636685,0.845324) init=(-0.418744,-0.644323,0.824682)
[PickSurface] world=(-0.514902,-0.655584,0.834044) init=(-0.475162,-0.652105,0.860653)
[PickSurface] world=(-0.508068,-0.656726,0.873939) init=(-0.531581,-0.659887,0.896623)
[PickSurface] world=(-0.478133,-0.651997,0.886072) init=(-0.475162,-0.652105,0.860653)
[PickSurface] world=(-0.226805,-0.460008,0.880441) init=(-0.217907,-0.437614,0.893669)
)PICK";
				std::vector<Eigen::Vector3f> customPickedInit;
				{
					const std::string s(customPickedLog);
					size_t p = 0;
					while (true) {
						size_t k = s.find("init=(", p);
						if (k == std::string::npos) break;
						k += 6;
						size_t e = s.find(")", k);
						if (e == std::string::npos) break;
						std::string t = s.substr(k, e - k);
						for (char& c : t) if (c == ',') c = ' ';
						std::stringstream ss(t);
						float x = 0.0f, y = 0.0f, z = 0.0f;
						if (ss >> x >> y >> z) customPickedInit.emplace_back(x, y, z);
						p = e + 1;
					}
				}

				std::vector<char> used(static_cast<size_t>(physRep.size()), 0);
				const float tol2 = std::pow(std::max(1e-6f, 0.03f * bboxDiag), 2.0f);
				for (const auto& p : customPickedInit) {
					int bestId = -1;
					float bestD2 = std::numeric_limits<float>::infinity();
					for (int id : surfaceIds) {
						if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
						if (used[static_cast<size_t>(id)]) continue;
						const float d2 = (initPosOf(id) - p).squaredNorm();
						if (d2 < bestD2) { bestD2 = d2; bestId = id; }
					}
					if (bestId >= 0 && bestD2 <= tol2) {
						used[static_cast<size_t>(bestId)] = 1;
						customFixedPhysIds.push_back(bestId);
					}
				}

				// Clear any previous hard-fixed flags, then apply only this curated set.
				for (auto& list : agentVerticesByPhysId) {
					for (Vertex* v : list) {
						if (!v) continue;
						v->isFixed = false;
					}
				}

				// We use these points as soft spring constraints (not hard-fixed) to avoid contact blow-ups.
				for (int id : customFixedPhysIds) {
					if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
					for (Vertex* v : agentVerticesByPhysId[static_cast<size_t>(id)]) {
						if (!v) continue;
						v->isFixed = false;
					}
				}

				std::cout << "[FixedPointsSpring] requested=" << customPickedInit.size()
				          << " matched=" << customFixedPhysIds.size()
				          << " tol=" << std::sqrt(tol2) << std::endl;
				if (customFixedPhysIds.empty()) {
					std::cout << "[FixedPointsSpring] No fixed points matched current mesh; nothing to visualize.\n";
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
	static bool showCavityWallVisual = true;
	static bool showFixedPointVisual = false;
	static bool showSuspensionVisual = true;
	static bool showSuspensionPatchPoints = false;
	static bool showLiverSmoothRender = true;
	static int anisoDemoState = 0; // 0: Off, 1: Isotropic Demo, 2: Anisotropic Demo
	static Vertex* anisoDemoVertex = nullptr;
	static float anisoDemoForceMag = 2700.0f; 
	static float anisoDemoRadius = 0.35f;    
	static float explodedScale = 0.5f;
	static bool whiteBackground = false;
	// Visualization toggle: highlight locally stiffer material override regions (e.g. tumor patch) in white.
	static bool showMaterialOverrideOverlay = false;
	static bool showAgentForceGraph = false;
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
			const auto pipelineFrameStart = std::chrono::steady_clock::now();
			double pipelinePreSimMs = 0.0;
			double pipelinePhysicsMs = 0.0;
			double pipelineSimMs = 0.0;
			double pipelineRenderMs = 0.0;
			double pipelineHapticTxMs = 0.0;
			const auto pipelinePreSimMark = pipelineFrameStart;

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
					std::array<Eigen::Vector3f, kFingerCount> leftTipsMm;
					std::array<Eigen::Vector3f, kFingerCount> leftPrevMm;
					Eigen::Vector3f leftPalmMm = Eigen::Vector3f::Zero();
					double leftTimeSec = -1.0;
					if (leapTracker.getLeftHandTipsMm(&leftTipsMm, &leftPalmMm, &leftTimeSec)) {
						leapLeftLatestTipsMm = leftTipsMm;
						leapLeftLatestPalmMm = leftPalmMm;
						leapLeftLatestTimeSec = leftTimeSec;
						if (leapTracker.getLeftHandDistalPrevMm(&leftPrevMm)) {
							leapLeftLatestPrevMm = leftPrevMm;
						}
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

		// ------------------ Minimal runtime controls (no "expert" hotkeys; prefer UI)
		static bool agentUseVC = agentVirtualCoupling;
		static int agentForceGraphMode = 1; // 0=CONTACT, 1=DEVICE (filtered)
		static bool agentGripEnabledRuntime = agentGripEnabled;
		static KeyLatch suspensionVisualLatch;
		static KeyLatch suspensionPatchLatch;

		if (suspensionVisualLatch.consume(window, GLFW_KEY_L)) {
			showSuspensionVisual = !showSuspensionVisual;
		}
		if (suspensionPatchLatch.consume(window, GLFW_KEY_K)) {
			showSuspensionPatchPoints = !showSuspensionPatchPoints;
		}

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
		auto printLeapOffset = [&](LeapOffsetTarget target) {
			const Eigen::Vector3f& off = (target == LeapOffsetTarget::Right) ? leapRightWorldOffset : leapLeftWorldOffset;
			std::ostringstream ss;
			ss << std::fixed << std::setprecision(6);
			ss << "[LeapOffset] target=" << ((target == LeapOffsetTarget::Right) ? "RIGHT" : "LEFT")
			   << " off=(" << off.x() << "," << off.y() << "," << off.z() << ")\n";
			std::cout << ss.str();
		};

		// QWEASD: adjust translation offset for the currently selected hand (world space).
		//   A/D: X -/+
		//   Q/E: Y -/+
		//   W/S: Z +/-
		{
			static KeyLatch qLatch, wLatch, eLatch, aLatch, sLatch, dLatch;
			const float step = 0.2f * bboxDiag;
			Eigen::Vector3f* off = (leapOffsetTarget == LeapOffsetTarget::Right) ? &leapRightWorldOffset : &leapLeftWorldOffset;
			bool changed = false;

			if (aLatch.consume(window, GLFW_KEY_A)) { off->x() -= step; changed = true; }
			if (dLatch.consume(window, GLFW_KEY_D)) { off->x() += step; changed = true; }
			if (qLatch.consume(window, GLFW_KEY_Q)) { off->y() -= step; changed = true; }
			if (eLatch.consume(window, GLFW_KEY_E)) { off->y() += step; changed = true; }
			if (wLatch.consume(window, GLFW_KEY_W)) { off->z() += step; changed = true; }
			if (sLatch.consume(window, GLFW_KEY_S)) { off->z() -= step; changed = true; }

			if (changed) printLeapOffset(leapOffsetTarget);
		}

		auto doLeapCalibrate = [&]() {
			// Treat 'B' / UI action as a "recenter / calibrate" (not a toggle-off).
			// If Leap input is currently OFF, try to turn it ON first.
			if (!leapUseInput) {
				leapUseInput = true;
				if (!leapTracker.init()) {
					std::cerr << "[LeapC] init failed; Leap input remains disabled.\n";
					leapUseInput = false;
				}
			}

			leapMappingCalibrated = false;
			leapAnchorWorld = agentDevicePositions[static_cast<size_t>(kIndexFinger)] - leapRightWorldOffset;

			leapLeftMappingCalibrated = false;
			leapLeftAnchorWorld = leapLeftHomeAnchor;
			leapLeftCenterMm.setZero();

			// Reset the left-hand visualization to a stable "home" pose (including any manual offset).
			for (int fi = 0; fi < kFingerCount; ++fi) {
				const Eigen::Vector3f p = leapLeftHomeAnchor + agentHandFingerOffsets[static_cast<size_t>(fi)] + leapLeftWorldOffset;
				leapLeftWorldTips[static_cast<size_t>(fi)] = p;
				leapLeftWorldPrevTips[static_cast<size_t>(fi)] = p;
				leapLeftWorldVelTips[static_cast<size_t>(fi)].setZero();
			}
			leapLeftWorldPalm = leapLeftHomeAnchor + leapLeftWorldOffset;
			leapLeftWorldPrevPalm = leapLeftHomeAnchor + leapLeftWorldOffset;
			leapLeftWorldVelPalm.setZero();
			for (int fi = 0; fi < kFingerCount; ++fi) {
				const Eigen::Vector3f tip = leapLeftWorldTips[static_cast<size_t>(fi)];
				Eigen::Vector3f dir = tip - leapLeftWorldPalm;
				const float dlen = dir.norm();
				if (dlen > 1e-8f) dir /= dlen;
				else dir = -Eigen::Vector3f::UnitY();
				const Eigen::Vector3f base = tip - dir * leftHandCapsuleLength;
				for (int si = 0; si < leftHandSamplesClamped; ++si) {
					const float t = (leftHandSamplesClamped > 1) ? (static_cast<float>(si) / static_cast<float>(leftHandSamplesClamped - 1)) : 1.0f;
					const Eigen::Vector3f p = base + t * (tip - base);
					const size_t idx = static_cast<size_t>(fi * leftHandSamplesClamped + si);
					if (idx >= leftHandDevicePositions.size()) continue;
					leftHandDevicePositions[idx] = p;
					leftHandDeviceVelocities[idx].setZero();
					leftHandProxyPositions[idx] = p;
					leftHandProxyVelocities[idx].setZero();
					leftHandActiveContactTriangle[idx] = -1;
				}
			}

			leftHandCapsulesEnabledRuntime = leftHandCapsulesWanted && leapUseInput;
			std::cout << "[LeapC] Calibrate"
			          << " | Input " << (leapUseInput ? "ON" : "OFF")
			          << " | workspace(mm)=(" << leapWorkspaceXmm << "," << leapWorkspaceYmm << "," << leapWorkspaceZmm << ")"
			          << " worldMargin=" << leapWorldMargin
			          << " gain=" << leapGain
			          << " yOffset=" << leapYOffsetBboxFrac
			          << " spread=" << leapFingerSpreadGain
			          << " smoothing=" << leapSmoothingTime
			          << " flip=(" << (leapFlipX ? 1 : 0) << "," << (leapFlipY ? 1 : 0) << "," << (leapFlipZ ? 1 : 0) << ")"
			          << "\n";
		};

		static KeyLatch leapToggleLatch;
		if (leapToggleLatch.consume(window, GLFW_KEY_B)) {
			doLeapCalibrate();
		}
#endif

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
				bool leftHandWorldFresh = false;
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
									// Keep leapAnchorWorld stable. (Recenter is an explicit user action.)

									// Get rotations
									std::array<Eigen::Quaternionf, kFingerCount> rotations;
									leapTracker.getRightHandRotations(&rotations);

										// Keep the base mostly unconstrained (avoid "air walls").
										// Only apply a very wide safety clamp to prevent runaway on tracking glitches.
										const float safe = 5.0f * bboxDiag;
										const Eigen::Vector3f clampMin = bboxCenter - Eigen::Vector3f::Ones() * safe;
										const Eigen::Vector3f clampMax = bboxCenter + Eigen::Vector3f::Ones() * safe;

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
											const Eigen::Vector3f target = base + relWorld + leapRightWorldOffset;

											agentDevicePositions[static_cast<size_t>(fi)] = target;
											agentDeviceRotations[static_cast<size_t>(fi)] = rotations[static_cast<size_t>(fi)];
											agentDevicePrevPositions[static_cast<size_t>(fi)] = target;
											agentDeviceVelocities[static_cast<size_t>(fi)].setZero();
										}
										leapMappingCalibrated = true;
									} else {
										// Keep the base mostly unconstrained (avoid "air walls").
										const float safe = 5.0f * bboxDiag;
										const Eigen::Vector3f clampMin = bboxCenter - Eigen::Vector3f::Ones() * safe;
										const Eigen::Vector3f clampMax = bboxCenter + Eigen::Vector3f::Ones() * safe;

										Eigen::Vector3f base = leapAnchorWorld + (indexTipMm - leapCenterMm).cwiseProduct(scale);
										base.y() += yOffset;
										base = base.cwiseMax(clampMin).cwiseMin(clampMax);

										const float maxRel = 0.5f * bboxDiag;
										const Eigen::Vector3f relClamp(maxRel, maxRel, maxRel);
										
										// Get rotations every frame
										std::array<Eigen::Quaternionf, kFingerCount> rotations;
										leapTracker.getRightHandRotations(&rotations);

										for (int fi = 0; fi < kFingerCount; ++fi) {
											const Eigen::Vector3f tipMm = leapLatestTipsMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
											const Eigen::Vector3f relMm = tipMm - indexTipMm;
											Eigen::Vector3f relWorld = relMm.cwiseProduct(scale.cwiseProduct(spreadScale));
											relWorld = relWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
											const Eigen::Vector3f target = base + relWorld + leapRightWorldOffset;

											auto& p = agentDevicePositions[static_cast<size_t>(fi)];
											auto& pPrev = agentDevicePrevPositions[static_cast<size_t>(fi)];
											auto& v = agentDeviceVelocities[static_cast<size_t>(fi)];

											p = p + alpha * (target - p);
											agentDeviceRotations[static_cast<size_t>(fi)] = rotations[static_cast<size_t>(fi)];
											v = (p - pPrev) / std::max(1e-8f, timeStep);
											pPrev = p;
										}
									}
									usedLeap = true;
								} else {
									// If tracking is stale, let keyboard drive. Keep the last calibration so the
									// mapping doesn't jump when tracking resumes.
								}
						}
	#endif
						if (!usedLeap) {
							// No keyboard driving; keep the last pose when Leap is unavailable/stale.
							for (int fi = 0; fi < kFingerCount; ++fi) {
								auto& p = agentDevicePositions[static_cast<size_t>(fi)];
								auto& pPrev = agentDevicePrevPositions[static_cast<size_t>(fi)];
								auto& v = agentDeviceVelocities[static_cast<size_t>(fi)];
								v.setZero();
								pPrev = p;
							}
						}

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
						// Left-hand mapping (Leap mm -> world). Drives non-haptic capsule collision.
						leftHandWorldFresh = false;
						if (leapUseInput) {
							const double now = glfwGetTime();
							const double staleSec = 0.2;
							if (leapLeftLatestTimeSec >= 0.0 && (now - leapLeftLatestTimeSec) <= staleSec) {
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

								// Left hand: keep smoothing independent from the right-hand haptic proxy.
								const float smooth = std::max(0.0f, leftHandExtraSmoothingTime);
								const float alpha = (smooth > 1e-6f) ? (1.0f - std::exp(-timeStep / smooth)) : 1.0f;
								const float yOffset = leapYOffsetBboxFrac * extents.y();

								const float spreadGain = std::max(0.0f, leapFingerSpreadGain);
								const Eigen::Vector3f spreadScale(spreadGain, 1.0f, spreadGain);

								const Eigen::Vector3f indexTipMm =
									leapLeftLatestTipsMm[static_cast<size_t>(kIndexFinger)].cwiseProduct(axisSign);
								if (!leapLeftMappingCalibrated) {
									leapLeftCenterMm = indexTipMm;
									leapLeftAnchorWorld = leapLeftHomeAnchor;

									// Keep left-hand base mostly unconstrained, same as right hand, to avoid "air walls".
									const float safe = 5.0f * bboxDiag;
									const Eigen::Vector3f clampMin = bboxCenter - Eigen::Vector3f::Ones() * safe;
									const Eigen::Vector3f clampMax = bboxCenter + Eigen::Vector3f::Ones() * safe;
									Eigen::Vector3f base = leapLeftAnchorWorld;
									base.y() += yOffset;
									base = base.cwiseMax(clampMin).cwiseMin(clampMax);

									const float maxRel = 0.5f * bboxDiag;
									const Eigen::Vector3f relClamp(maxRel, maxRel, maxRel);

									for (int fi = 0; fi < kFingerCount; ++fi) {
										const Eigen::Vector3f tipMm = leapLeftLatestTipsMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
										const Eigen::Vector3f prevMm = leapLeftLatestPrevMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
										const Eigen::Vector3f relMm = tipMm - indexTipMm;
										Eigen::Vector3f relWorld = relMm.cwiseProduct(scale.cwiseProduct(spreadScale));
										relWorld = relWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
										const Eigen::Vector3f target = base + relWorld + leapLeftWorldOffset;

										leapLeftWorldTips[static_cast<size_t>(fi)] = target;
										leapLeftWorldPrevTips[static_cast<size_t>(fi)] = target;
										leapLeftWorldVelTips[static_cast<size_t>(fi)].setZero();

										const Eigen::Vector3f relPrevMm = prevMm - indexTipMm;
										Eigen::Vector3f relPrevWorld = relPrevMm.cwiseProduct(scale.cwiseProduct(spreadScale));
										relPrevWorld = relPrevWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
										const Eigen::Vector3f targetPrev = base + relPrevWorld + leapLeftWorldOffset;
										leapLeftWorldPrevJoints[static_cast<size_t>(fi)] = targetPrev;
										leapLeftWorldPrevPrevJoints[static_cast<size_t>(fi)] = targetPrev;
										leapLeftWorldVelPrevJoints[static_cast<size_t>(fi)].setZero();
									}
									{
										const Eigen::Vector3f palmMm = leapLeftLatestPalmMm.cwiseProduct(axisSign);
										const Eigen::Vector3f relMm = palmMm - indexTipMm;
										const Eigen::Vector3f targetPalm = base + relMm.cwiseProduct(scale) + leapLeftWorldOffset;
										leapLeftWorldPalm = targetPalm;
										leapLeftWorldPrevPalm = targetPalm;
										leapLeftWorldVelPalm.setZero();
									}
									leapLeftMappingCalibrated = true;
								} else {
									const float safe = 5.0f * bboxDiag;
									const Eigen::Vector3f clampMin = bboxCenter - Eigen::Vector3f::Ones() * safe;
									const Eigen::Vector3f clampMax = bboxCenter + Eigen::Vector3f::Ones() * safe;
									Eigen::Vector3f base = leapLeftAnchorWorld + (indexTipMm - leapLeftCenterMm).cwiseProduct(scale);
									base.y() += yOffset;
									base = base.cwiseMax(clampMin).cwiseMin(clampMax);

									const float maxRel = 0.5f * bboxDiag;
									const Eigen::Vector3f relClamp(maxRel, maxRel, maxRel);

									for (int fi = 0; fi < kFingerCount; ++fi) {
										const Eigen::Vector3f tipMm = leapLeftLatestTipsMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
										const Eigen::Vector3f prevMm = leapLeftLatestPrevMm[static_cast<size_t>(fi)].cwiseProduct(axisSign);
										const Eigen::Vector3f relMm = tipMm - indexTipMm;
										Eigen::Vector3f relWorld = relMm.cwiseProduct(scale.cwiseProduct(spreadScale));
										relWorld = relWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
										const Eigen::Vector3f target = base + relWorld + leapLeftWorldOffset;

										auto& p = leapLeftWorldTips[static_cast<size_t>(fi)];
										auto& pPrev = leapLeftWorldPrevTips[static_cast<size_t>(fi)];
										auto& v = leapLeftWorldVelTips[static_cast<size_t>(fi)];
										p = p + alpha * (target - p);
										v = (p - pPrev) / std::max(1e-8f, timeStep);
										pPrev = p;

										const Eigen::Vector3f relPrevMm = prevMm - indexTipMm;
										Eigen::Vector3f relPrevWorld = relPrevMm.cwiseProduct(scale.cwiseProduct(spreadScale));
										relPrevWorld = relPrevWorld.cwiseMax(-relClamp).cwiseMin(relClamp);
										const Eigen::Vector3f targetPrev = base + relPrevWorld + leapLeftWorldOffset;
										auto& pj = leapLeftWorldPrevJoints[static_cast<size_t>(fi)];
										auto& pjPrev = leapLeftWorldPrevPrevJoints[static_cast<size_t>(fi)];
										auto& vj = leapLeftWorldVelPrevJoints[static_cast<size_t>(fi)];
										pj = pj + alpha * (targetPrev - pj);
										vj = (pj - pjPrev) / std::max(1e-8f, timeStep);
										pjPrev = pj;
									}
									{
										const Eigen::Vector3f palmMm = leapLeftLatestPalmMm.cwiseProduct(axisSign);
										const Eigen::Vector3f relMm = palmMm - indexTipMm;
										const Eigen::Vector3f targetPalm = base + relMm.cwiseProduct(scale) + leapLeftWorldOffset;
										leapLeftWorldPalm = leapLeftWorldPalm + alpha * (targetPalm - leapLeftWorldPalm);
										leapLeftWorldVelPalm = (leapLeftWorldPalm - leapLeftWorldPrevPalm) / std::max(1e-8f, timeStep);
										leapLeftWorldPrevPalm = leapLeftWorldPalm;
									}
								}

								leftHandWorldFresh = true;
							} else {
								// Keep the last calibration so the left hand doesn't jump when tracking resumes.
							}
						}
						// Update capsule sample points (device) from the mapped tips+palm.
						if (leftHandWorldFresh) {
							for (int fi = 0; fi < kFingerCount; ++fi) {
								const Eigen::Vector3f tip = leapLeftWorldTips[static_cast<size_t>(fi)];
								const Eigen::Vector3f tipV = leapLeftWorldVelTips[static_cast<size_t>(fi)];
								const Eigen::Vector3f prev = leapLeftWorldPrevJoints[static_cast<size_t>(fi)];
								const Eigen::Vector3f prevV = leapLeftWorldVelPrevJoints[static_cast<size_t>(fi)];

								Eigen::Vector3f base = prev;
								Eigen::Vector3f baseV = prevV;
								Eigen::Vector3f tip2 = tip;
								Eigen::Vector3f tipV2 = tipV;

								// If prev-joint data is missing/degenerate, fall back to a short segment towards the palm.
								if ((tip2 - base).squaredNorm() < 1e-16f) {
									const Eigen::Vector3f palm = leapLeftWorldPalm;
									const Eigen::Vector3f palmV = leapLeftWorldVelPalm;

									Eigen::Vector3f dir = tip2 - palm;
									const float dlen = dir.norm();
									if (dlen > 1e-8f) dir /= dlen;
									else dir = -Eigen::Vector3f::UnitY();

									base = tip2 - dir * leftHandCapsuleLength;
									baseV = palmV;
								}
								for (int si = 0; si < leftHandSamplesClamped; ++si) {
									const float t = (leftHandSamplesClamped > 1) ? (static_cast<float>(si) / static_cast<float>(leftHandSamplesClamped - 1)) : 1.0f;
									const Eigen::Vector3f target = base + t * (tip2 - base);
									const Eigen::Vector3f vTarget = baseV * (1.0f - t) + tipV2 * t;
									const size_t idx = static_cast<size_t>(fi * leftHandSamplesClamped + si);
									if (idx >= leftHandDevicePositions.size()) continue;
									leftHandDevicePositions[idx] = target;
									leftHandDeviceVelocities[idx] = vTarget;
								}
							}
						} else {
							for (auto& v : leftHandDeviceVelocities) v.setZero();
						}
#endif
				}

		// UI button triggers deterministic Experiment 3 (one-click).

		// ------------------ Manual right-click drag force (restores RMB "apply force")
		// Holding RMB drags the nearest vertex under the cursor and applies a spring-like force.
		static bool prevRightDown = false;
		const bool rightDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
		const bool rightPressed = rightDown && !prevRightDown;
		const bool rightReleased = !rightDown && prevRightDown;
		prevRightDown = rightDown;
		static bool prevLeftDown = false;
		const bool leftDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
		const bool leftPressed = leftDown && !prevLeftDown;
		const bool leftReleased = !leftDown && prevLeftDown;
		prevLeftDown = leftDown;

		auto pointInRect = [](double x, double y, const SimpleUI::Rect& r) {
			return x >= r.x && x <= (r.x + r.w) && y >= r.y && y <= (r.y + r.h);
		};
		const bool cursorInUiButton = pointInRect(ui.state().mouseXWindow, ui.state().mouseYWindow, uiRunRect);

		// Temporary picking helper: print only on a real click (press+release without drag).
		static bool leftPickCandidate = false;
		static double leftPickPressX = 0.0;
		static double leftPickPressY = 0.0;
		static double leftPickPressT = 0.0;
		if (leftPressed) {
			leftPickCandidate = !cursorInUiButton;
			leftPickPressX = ui.state().mouseXWindow;
			leftPickPressY = ui.state().mouseYWindow;
			leftPickPressT = glfwGetTime();
		}
		if (leftDown && leftPickCandidate) {
			const double dx = ui.state().mouseXWindow - leftPickPressX;
			const double dy = ui.state().mouseYWindow - leftPickPressY;
			const double drag2 = dx * dx + dy * dy;
			if (drag2 > 25.0) leftPickCandidate = false; // >5 px movement => drag, do not print
		}
		if (leftReleased && leftPickCandidate) {
			leftPickCandidate = false;
			const double clickDt = glfwGetTime() - leftPickPressT;
			if (clickDt <= 0.40) {
				const float cameraDist = std::max(1e-6f, 1.5f * bboxDiag * zoomFactor);
				const float cameraLift = 0.0f;
				Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
				const Eigen::Matrix3f viewRotation = rotation.toRotationMatrix();
				model.block<3, 3>(0, 0) = viewRotation;
				const Eigen::Vector3f viewTranslation =
					Eigen::Vector3f(0.0f, cameraLift, -cameraDist) - viewRotation * bboxCenter;
				model.block<3, 1>(0, 3) = viewTranslation;
				const Eigen::Matrix4f projection = buildProjectionMatrix();
				const Eigen::Matrix4f invProjectionModel = (projection * model).inverse();
				const Eigen::Vector3f rayNear = unprojectCursorToWorld(
					ui.state().mouseXFramebuffer, ui.state().mouseYFramebuffer, -1.0f,
					invProjectionModel, ui.state().framebufferWidth, ui.state().framebufferHeight);
				const Eigen::Vector3f rayFar = unprojectCursorToWorld(
					ui.state().mouseXFramebuffer, ui.state().mouseYFramebuffer, 1.0f,
					invProjectionModel, ui.state().framebufferWidth, ui.state().framebufferHeight);
				Eigen::Vector3f rayDir = rayFar - rayNear;
				const float rayLen = rayDir.norm();
				if (rayLen > 1e-8f) rayDir /= rayLen;

				float bestT = std::numeric_limits<float>::infinity();
				Vertex* bestV = nullptr;
				Vertex* bestA = nullptr;
				Vertex* bestB = nullptr;
				Vertex* bestC = nullptr;
				Eigen::Vector3f bestHit = Eigen::Vector3f::Zero();
				for (const auto& tri : agentContactTriangles) {
					if (!tri.a || !tri.b || !tri.c) continue;
					const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
					const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
					const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
					float t = 0.0f;
					if (!rayIntersectsTriangle(rayNear, rayDir, a, b, c, &t)) continue;
					if (t >= bestT) continue;
					const Eigen::Vector3f hit = rayNear + rayDir * t;
					bestT = t;
					bestHit = hit;
					const float da2 = (a - hit).squaredNorm();
					const float db2 = (b - hit).squaredNorm();
					const float dc2 = (c - hit).squaredNorm();
					if (da2 <= db2 && da2 <= dc2) bestV = tri.a;
					else if (db2 <= da2 && db2 <= dc2) bestV = tri.b;
					else bestV = tri.c;
					bestA = tri.a;
					bestB = tri.b;
					bestC = tri.c;
				}

				if (bestV) {
					Eigen::Vector3f pickedInit(bestV->initx, bestV->inity, bestV->initz);
					if (bestA && bestB && bestC) {
						const Eigen::Vector3f wa(bestA->x, bestA->y, bestA->z);
						const Eigen::Vector3f wb(bestB->x, bestB->y, bestB->z);
						const Eigen::Vector3f wc(bestC->x, bestC->y, bestC->z);
						const Eigen::Vector3f v0 = wb - wa;
						const Eigen::Vector3f v1 = wc - wa;
						const Eigen::Vector3f v2 = bestHit - wa;
						const float d00 = v0.dot(v0);
						const float d01 = v0.dot(v1);
						const float d11 = v1.dot(v1);
						const float d20 = v2.dot(v0);
						const float d21 = v2.dot(v1);
						const float denom = d00 * d11 - d01 * d01;
						if (std::abs(denom) > 1e-12f) {
							const float vb = (d11 * d20 - d01 * d21) / denom;
							const float vc = (d00 * d21 - d01 * d20) / denom;
							const float va = 1.0f - vb - vc;
							const Eigen::Vector3f ia(bestA->initx, bestA->inity, bestA->initz);
							const Eigen::Vector3f ib(bestB->initx, bestB->inity, bestB->initz);
							const Eigen::Vector3f ic(bestC->initx, bestC->inity, bestC->initz);
							pickedInit = ia * va + ib * vb + ic * vc;
						}
					}
					std::cout << "[PickSurface] world=("
					          << bestHit.x() << ","
					          << bestHit.y() << ","
					          << bestHit.z() << ") init=("
					          << pickedInit.x() << ","
					          << pickedInit.y() << ","
					          << pickedInit.z() << ")"
					          << std::endl;
				}
			}
		}

		if (!isAutoTestActive && !experiment3.isActive() && !experiment1.isActive() && !experiment2.isActive() && !experiment4.isActive()) {
			if (rightReleased) {
				dragState.active = false;
				dragState.target = nullptr;
			}

			if (rightPressed && !cursorInUiButton) {
				const float cameraDist = std::max(1e-6f, 1.5f * bboxDiag * zoomFactor);
				const float cameraLift = 0.0f;
				Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
				const Eigen::Matrix3f viewRotation = rotation.toRotationMatrix();
				model.block<3, 3>(0, 0) = viewRotation;
				const Eigen::Vector3f viewTranslation =
					Eigen::Vector3f(0.0f, cameraLift, -cameraDist) - viewRotation * bboxCenter;
				model.block<3, 1>(0, 3) = viewTranslation;
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
					const float cameraDist = std::max(1e-6f, 1.5f * bboxDiag * zoomFactor);
					const float cameraLift = 0.0f;
					Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
					const Eigen::Matrix3f viewRotation = rotation.toRotationMatrix();
					model.block<3, 3>(0, 0) = viewRotation;
					const Eigen::Vector3f viewTranslation =
						Eigen::Vector3f(0.0f, cameraLift, -cameraDist) - viewRotation * bboxCenter;
					model.block<3, 1>(0, 3) = viewTranslation;
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

			// Optional: 3-point suspension ligaments (surface patch springs -> wall anchors).
			// Implemented as per-vertex acceleration contributions, similar to drag/anchor springs.
			if (suspensionEnabled && !suspensions.empty() && !agentVerticesByPhysId.empty()) {
				for (const auto& s : suspensions) {
					if (!s.enabled) continue;
					if (s.physIds.empty()) continue;
					if (!(s.k > 0.0f) && !(s.damping > 0.0f)) continue;

					for (size_t ii = 0; ii < s.physIds.size(); ++ii) {
						const int id = s.physIds[ii];
						if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
						const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
						if (list.empty()) continue;

						bool anyFixed = false;
						for (Vertex* v : list) {
							if (v && v->isFixed) {
								anyFixed = true;
								break;
							}
						}
						if (anyFixed) continue;

						const float w = (ii < s.weights.size()) ? s.weights[ii] : 1.0f;
						if (!(w > 0.0f)) continue;

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

						const Eigen::Vector3f restOff = (ii < s.restOffsetFromAnchor.size()) ? s.restOffsetFromAnchor[ii] : Eigen::Vector3f::Zero();
						const Eigen::Vector3f desired = s.anchorWorld + restOff;
						Eigen::Vector3f accel = s.k * (desired - pAvg) - s.damping * vAvg;
						accel *= w;

						const float maxA = s.maxAccel;
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
			}

			// Custom picked points as soft "ligament-like" springs to their init positions.
			// This replaces hard pinning to keep interaction stable under left-hand contact.
			if (!customFixedPhysIds.empty() && !agentVerticesByPhysId.empty()) {
				const float k = 800.0f;
				const float c = 55.0f;
				const float maxA = 12000.0f;
				const float leashBeta = 0.20f; // per-frame positional leash toward rest
				const float leashMaxStep = 0.008f;
				for (int id : customFixedPhysIds) {
					if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
					if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
					const Vertex* rep = physRep[static_cast<size_t>(id)];
					if (!rep) continue;
					const Eigen::Vector3f rest(rep->initx, rep->inity, rep->initz);
					const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
					if (list.empty()) continue;

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

					Eigen::Vector3f accel = k * (rest - pAvg) - c * vAvg;
					const float aLen = accel.norm();
					if (aLen > maxA && aLen > 1e-12f) accel *= (maxA / aLen);
					if (accel.squaredNorm() <= 1e-12f) continue;

					for (Vertex* v : list) {
						if (!v || v->isFixed) continue;
						const int idx = v->index;
						if (idx >= 0 && idx < static_cast<int>(dragForces.size())) {
							dragForces[static_cast<size_t>(idx)] += accel;
						}
						// Extra "bind" without hard-fixing: small direct correction each frame.
						Eigen::Vector3f dp = leashBeta * (rest - Eigen::Vector3f(v->x, v->y, v->z));
						const float dplen = dp.norm();
						if (dplen > leashMaxStep && dplen > 1e-12f) dp *= (leashMaxStep / dplen);
						v->x += dp.x();
						v->y += dp.y();
						v->z += dp.z();
						v->velx *= 0.5f;
						v->vely *= 0.5f;
						v->velz *= 0.5f;
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

								// VC stability without losing tangential "grip":
								// Reduce only the NORMAL spring stiffness when deeply pressed, keep tangential stiffness.
								Eigen::Vector3f nIn = agentFilteredContactNormalsIn[static_cast<size_t>(fi)];
								const bool hasN = (wasContact && nIn.squaredNorm() > 1e-12f);
								if (hasN) nIn.normalize();

								float kScaleN = 1.0f;
								float kScaleT = 1.0f;
								float cScaleN = 1.0f;
								float cScaleT = 1.0f;
								if (wasContact) {
									const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
									const float penFrac = lastPen / std::max(1e-6f, agentSphere.radius);
									const float start = 0.20f;
									const float end = 0.50f;
									if (penFrac > start) {
										const float t = std::clamp((penFrac - start) / (end - start), 0.0f, 1.0f);
										// Keep some normal stiffness for a firm press, but avoid chatter at deep indentation.
										const float kMinN = 0.35f;
										kScaleN = 1.0f - (1.0f - kMinN) * t;
										// Increase normal damping in deep contact to kill residual oscillations.
										cScaleN = 1.0f + 1.5f * t;
									}
								}
								
								for (int si = 0; si < substeps; ++si) {
									Eigen::Vector3f disp = (devPos - proxyPos);
									const float dispLen = disp.norm();
									if (maxVcDist > 1e-6f && dispLen > maxVcDist) {
										disp *= (maxVcDist / dispLen);
									}
									Eigen::Vector3f springForceN = Eigen::Vector3f::Zero();
									Eigen::Vector3f dampingForceN = Eigen::Vector3f::Zero();
									if (hasN) {
										const float dn = disp.dot(nIn);
										const Eigen::Vector3f dispN = nIn * dn;
										const Eigen::Vector3f dispT = disp - dispN;
										springForceN = (agentVcKLen * kScaleN) * dispN + (agentVcKLen * kScaleT) * dispT;

										const Eigen::Vector3f relV = (devVel - proxyVel);
										const float vn = relV.dot(nIn);
										const Eigen::Vector3f relVN = nIn * vn;
										const Eigen::Vector3f relVT = relV - relVN;
										dampingForceN = (vcCLen * cScaleN) * relVN + (vcCLen * cScaleT) * relVT;
									} else {
										springForceN = (agentVcKLen * kScaleN) * disp;
										dampingForceN = vcCLen * (devVel - proxyVel);
									}
									const Eigen::Vector3f couplingForceN = springForceN + dampingForceN;

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
								Eigen::Vector3f springForceOut = Eigen::Vector3f::Zero();
								Eigen::Vector3f dampingForceOut = Eigen::Vector3f::Zero();
								if (hasN) {
									const float dn = dispOut.dot(nIn);
									const Eigen::Vector3f dispN = nIn * dn;
									const Eigen::Vector3f dispT = dispOut - dispN;
									springForceOut = (agentVcKLen * kScaleN) * dispN + (agentVcKLen * kScaleT) * dispT;

									const float vn = devVel.dot(nIn);
									const Eigen::Vector3f vN = nIn * vn;
									const Eigen::Vector3f vT = devVel - vN;
									dampingForceOut = (vcCLen * cScaleN) * vN + (vcCLen * cScaleT) * vT;
								} else {
									springForceOut = (agentVcKLen * kScaleN) * dispOut;
									dampingForceOut = vcCLen * devVel;
								}
								const Eigen::Vector3f deviceForceN = -(springForceOut + dampingForceOut);
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
								agentFilteredDeviceForcesN[static_cast<size_t>(fi)].setZero();
								agentFilteredContactForcesN[static_cast<size_t>(fi)].setZero();
							agentLastContactForcesN[static_cast<size_t>(fi)].setZero();
						agentLastCouplingForcesN[static_cast<size_t>(fi)].setZero();
						agentLastContactCounts[static_cast<size_t>(fi)] = 0;
						agentLastContactPenetrations[static_cast<size_t>(fi)] = 0.0f;
						agentLastContactNormalsIn[static_cast<size_t>(fi)].setZero();
						agentFilteredContactNormalsIn[static_cast<size_t>(fi)].setZero();
						agentLastActiveContactTriangle[static_cast<size_t>(fi)] = -1;
						agentGripActive[static_cast<size_t>(fi)] = false;
						agentGripTriangle[static_cast<size_t>(fi)] = -1;
						agentGripBary[static_cast<size_t>(fi)].setZero();
					}
				}

		Eigen::Vector3f inputForce = Eigen::Vector3f::Zero(); // Placeholder for removed manual input


		static bool drawFaces = true;
		static bool drawEdges = false;

		if (pipelineProfiler.active && agentSphere.enabled) {
			const float phase = static_cast<float>(frame) * 0.04f;
			const float depth = 0.10f * bboxDiag * (0.50f + 0.50f * std::sin(phase));
			const Eigen::Vector3f pressTarget = tumorPresetInits[0] + Eigen::Vector3f(0.0f, depth, 0.0f);
			for (int fi = 0; fi < kFingerCount; ++fi) {
				auto& p = agentDevicePositions[static_cast<size_t>(fi)];
				auto& pPrev = agentDevicePrevPositions[static_cast<size_t>(fi)];
				const Eigen::Vector3f target = pressTarget + agentHandFingerOffsets[static_cast<size_t>(fi)];
				p = target;
				agentDeviceVelocities[static_cast<size_t>(fi)] = (p - pPrev) / std::max(1e-8f, timeStep);
				pPrev = p;
				agentProxyPositions[static_cast<size_t>(fi)] = p;
				agentProxyVelocities[static_cast<size_t>(fi)] = agentDeviceVelocities[static_cast<size_t>(fi)];
			}
		}
		
		// Physics update only when not paused
		if (!isPaused) {
			const auto pipelineSimBlockStart = std::chrono::steady_clock::now();
			pipelinePreSimMs = std::chrono::duration<double, std::milli>(
				pipelineSimBlockStart - pipelinePreSimMark).count();
			const auto pipelinePhysicsStart = pipelineSimBlockStart;
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
			pipelinePhysicsMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - pipelinePhysicsStart).count();
			
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
								float maxProxyStep = 0.0f;
								for (int fi = 0; fi < kFingerCount; ++fi) {
									const Eigen::Vector3f dp = agentProxyPositions[static_cast<size_t>(fi)] -
										agentProxyStartPositions[static_cast<size_t>(fi)];
									const float step = dp.norm();
									if (step > maxProxyStep) maxProxyStep = step;
								}
								const float epsBase = std::max(1e-4f * bboxDiag, 0.02f * r);
								const float epsSpeed = std::min(maxProxyStep, 2.0f * r);
								const float eps = epsBase + epsSpeed;
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

											Eigen::Vector3f nIn = agentFilteredContactNormalsIn[static_cast<size_t>(fi)];
											const bool hasN = (wasContact && nIn.squaredNorm() > 1e-12f);
											if (hasN) nIn.normalize();

											float kScaleN = 1.0f;
											float kScaleT = 1.0f;
											float cScaleN = 1.0f;
											float cScaleT = 1.0f;
											if (wasContact) {
												const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
												const float penFrac = lastPen / std::max(1e-6f, r);
												const float start = 0.20f;
												const float end = 0.50f;
												if (penFrac > start) {
													const float t = std::clamp((penFrac - start) / (end - start), 0.0f, 1.0f);
													const float kMinN = 0.35f;
													kScaleN = 1.0f - (1.0f - kMinN) * t;
													cScaleN = 1.0f + 1.5f * t;
												}
											}

											Eigen::Vector3f springForceN = Eigen::Vector3f::Zero();
											Eigen::Vector3f dampingForceN = Eigen::Vector3f::Zero();
											if (hasN) {
												const float dn = disp.dot(nIn);
												const Eigen::Vector3f dispN = nIn * dn;
												const Eigen::Vector3f dispT = disp - dispN;
												springForceN = (agentVcKLen * kScaleN) * dispN + (agentVcKLen * kScaleT) * dispT;

												const Eigen::Vector3f relV = (devVel - sphereVel);
												const float vn = relV.dot(nIn);
												const Eigen::Vector3f relVN = nIn * vn;
												const Eigen::Vector3f relVT = relV - relVN;
												dampingForceN = (vcCLen * cScaleN) * relVN + (vcCLen * cScaleT) * relVT;
											} else {
												springForceN = (agentVcKLen * kScaleN) * disp;
												dampingForceN = vcCLen * (devVel - sphereVel);
											}
											const Eigen::Vector3f driveForceN = springForceN + dampingForceN;
											const float sphereInvMass = 1.0f / agentProxyMassKg;
											const float localScale = std::max(1e-6f, materialScaleAtWorldPoint(sphereCenter));
											const float allowedPenLocal = std::clamp(allowedPen / localScale, 0.0f, 0.95f * r);
											float proxyInvMassScaleLocal = contactProxyInvMassScale;
											// Make the harder side push the proxy back slightly (bigger VC force -> "harder" feel).
											if (localScale > 1.05f) proxyInvMassScaleLocal = std::max(proxyInvMassScaleLocal, 0.85f);
											const float muLocal = std::clamp(agentFrictionMu * std::sqrt(std::clamp(localScale, 0.25f, 16.0f)), 0.0f, 10.0f);
											const float tangentialDampLocal = std::clamp(tangentialDamp * std::sqrt(std::clamp(localScale, 0.25f, 16.0f)), 0.0f, 1.0f);
											if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
													contact = solveAgentSphereTriangleCollisionConstraint(
														sphereCenter,
														sphereVel,
														sphereInvMass,
														proxyInvMassScaleLocal,
														contactVelRelax,
														contactVelRelaxMin,
														contactNormalDamp,
														/*injectVelocityFromPositionCorrection=*/true,
														r,
														allowedPenLocal,
														eps,
														timeStep,
														corr,
														tangentialDampLocal,
														muLocal,
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
													proxyInvMassScaleLocal,
												contactVelRelax,
												contactVelRelaxMin,
												contactNormalDamp,
												r,
												allowedPenLocal,
													eps,
													timeStep,
													corr,
													tangentialDampLocal,
													muLocal,
													iters,
													driveForceN,
													agentContactVertexPhysIds,
													agentVerticesByPhysId,
												physMassSumKg);
										}
											{
												// 1DOF haptics: make normal reaction scale with material stiffness so a "sampling"
												// probe press produces a clear step when crossing soft->hard regions.
												const float exp = std::clamp(agentContactForceMaterialExponent, 0.0f, 4.0f);
												const float fScale = std::pow(localScale, exp);
												contact.reactionForceN *= fScale;
											}
									} else {
										// Kinematic drive: substep along the proxy motion to avoid tunneling when the user moves fast.
										const Eigen::Vector3f p0 = agentProxyStartPositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f p1 = agentProxyPositions[static_cast<size_t>(fi)];
										const Eigen::Vector3f dp = p1 - p0;
										const float stepLen = dp.norm();
										const int substeps = std::clamp(static_cast<int>(std::ceil(stepLen / maxStep)), 1, 64);
										const float dtSub = timeStep / static_cast<float>(substeps);
										// For stability, keep a minimum number of collision iterations per substep.
										// Otherwise, fast motion increases substeps but *decreases* per-substep work,
										// causing accumulated penetration and contact buzzing.
										const int itersPerSubstepMin = 6;
										const int itersSub = std::clamp(
											std::max(itersPerSubstepMin, static_cast<int>(std::ceil(static_cast<float>(iters) / static_cast<float>(substeps)))),
											1,
											64);

										Eigen::Vector3f impulseNsec = Eigen::Vector3f::Zero();
										int contactVerts = 0;
										float maxPen = 0.0f;
										Eigen::Vector3f sumN = Eigen::Vector3f::Zero();

										Eigen::Vector3f prevP = p0;
										for (int si = 0; si < substeps; ++si) {
											const float a = static_cast<float>(si + 1) / static_cast<float>(substeps);
											Eigen::Vector3f p = p0 + dp * a;
											Eigen::Vector3f v = (p - prevP) / std::max(1e-8f, dtSub);
											const float localScale = std::max(1e-6f, materialScaleAtWorldPoint(p));
											const float allowedPenLocal = std::clamp(allowedPen / localScale, 0.0f, 0.95f * r);
											float proxyInvMassScaleLocal = contactProxyInvMassScale;
											if (localScale > 1.05f) proxyInvMassScaleLocal = std::max(proxyInvMassScaleLocal, 0.85f);
											const float muLocal = std::clamp(agentFrictionMu * std::sqrt(std::clamp(localScale, 0.25f, 16.0f)), 0.0f, 10.0f);
											const float tangentialDampLocal = std::clamp(tangentialDamp * std::sqrt(std::clamp(localScale, 0.25f, 16.0f)), 0.0f, 1.0f);

											AgentContactResult c{};
												if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
														c = solveAgentSphereTriangleCollisionConstraint(
															p,
															v,
															0.0f,
															proxyInvMassScaleLocal,
															contactVelRelax,
															contactVelRelaxMin,
															contactNormalDamp,
															/*injectVelocityFromPositionCorrection=*/true,
															r,
															allowedPenLocal,
															eps,
															dtSub,
															corr,
															tangentialDampLocal,
															muLocal,
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
													proxyInvMassScaleLocal,
													contactVelRelax,
													contactVelRelaxMin,
													contactNormalDamp,
													r,
													allowedPenLocal,
													eps,
													dtSub,
													corr,
													tangentialDampLocal,
													muLocal,
													itersSub,
													Eigen::Vector3f::Zero(),
													agentContactVertexPhysIds,
													agentVerticesByPhysId,
													physMassSumKg);
											}

											{
												const float exp = std::clamp(agentContactForceMaterialExponent, 0.0f, 4.0f);
												const float fScale = std::pow(localScale, exp);
												c.reactionForceN *= fScale;
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
										{
											Eigen::Vector3f nIn = contact.avgNormal;
											if (contact.contactVertexCount > 0 && nIn.squaredNorm() > 1e-12f) {
												nIn.normalize();
												agentLastContactNormalsIn[static_cast<size_t>(fi)] = nIn;
											} else {
												agentLastContactNormalsIn[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
											}
										}
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

												Eigen::Vector3f nIn = agentFilteredContactNormalsIn[static_cast<size_t>(fi)];
												const bool hasN = (wasContact && nIn.squaredNorm() > 1e-12f);
												if (hasN) nIn.normalize();

												float kScaleN = 1.0f;
												float kScaleT = 1.0f;
												float cScaleN = 1.0f;
												float cScaleT = 1.0f;
												if (wasContact) {
													const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
													const float penFrac = lastPen / std::max(1e-6f, r);
													const float start = 0.20f;
													const float end = 0.50f;
													if (penFrac > start) {
														const float t = std::clamp((penFrac - start) / (end - start), 0.0f, 1.0f);
														const float kMinN = 0.35f;
														kScaleN = 1.0f - (1.0f - kMinN) * t;
														cScaleN = 1.0f + 1.5f * t;
													}
												}

												Eigen::Vector3f springForceN = Eigen::Vector3f::Zero();
												Eigen::Vector3f dampingForceN = Eigen::Vector3f::Zero();
												if (hasN) {
													const float dn = disp.dot(nIn);
													const Eigen::Vector3f dispN = nIn * dn;
													const Eigen::Vector3f dispT = disp - dispN;
													springForceN = (agentVcKLen * kScaleN) * dispN + (agentVcKLen * kScaleT) * dispT;

													const Eigen::Vector3f relV = (devVel - sphereVel);
													const float vn = relV.dot(nIn);
													const Eigen::Vector3f relVN = nIn * vn;
													const Eigen::Vector3f relVT = relV - relVN;
													dampingForceN = (vcCLen * cScaleN) * relVN + (vcCLen * cScaleT) * relVT;
												} else {
													springForceN = (agentVcKLen * kScaleN) * disp;
													dampingForceN = vcCLen * (devVel - sphereVel);
												}

												const Eigen::Vector3f driveForceN = springForceN + dampingForceN;
												const float sphereInvMass = 1.0f / agentProxyMassKg;
												const float localScale = std::max(1e-6f, materialScaleAtWorldPoint(sphereCenter));
												const float allowedPenLocal = std::clamp(allowedPen / localScale, 0.0f, 0.95f * r);
												float proxyInvMassScaleLocal = contactProxyInvMassScale;
												if (localScale > 1.05f) proxyInvMassScaleLocal = std::max(proxyInvMassScaleLocal, 0.85f);
												const float muLocal = std::clamp(agentFrictionMu * std::sqrt(std::clamp(localScale, 0.25f, 16.0f)), 0.0f, 10.0f);
												const float tangentialDampLocal = std::clamp(tangentialDamp * std::sqrt(std::clamp(localScale, 0.25f, 16.0f)), 0.0f, 1.0f);

												if (agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
														contact2 = solveAgentSphereTriangleCollisionConstraint(
															sphereCenter,
															sphereVel,
															sphereInvMass,
															proxyInvMassScaleLocal,
															contactVelRelax,
															contactVelRelaxMin,
															contactNormalDamp,
															/*injectVelocityFromPositionCorrection=*/true,
															r,
															allowedPenLocal,
															eps,
															timeStep,
															corr,
															tangentialDampLocal,
															muLocal,
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
													proxyInvMassScaleLocal,
													contactVelRelax,
													contactVelRelaxMin,
													contactNormalDamp,
													r,
													allowedPenLocal,
													eps,
													timeStep,
													corr,
													tangentialDampLocal,
													muLocal,
													iters2,
													driveForceN,
													agentContactVertexPhysIds,
													agentVerticesByPhysId,
													physMassSumKg);
											}

												{
													const float exp = std::clamp(agentContactForceMaterialExponent, 0.0f, 4.0f);
													const float fScale = std::pow(localScale, exp);
													contact2.reactionForceN *= fScale;
												}

												agentLastContactForcesN[static_cast<size_t>(fi)] = contact2.reactionForceN;
												agentLastContactCounts[static_cast<size_t>(fi)] = contact2.contactVertexCount;
												agentLastContactPenetrations[static_cast<size_t>(fi)] = contact2.maxPenetration;
												{
													Eigen::Vector3f nIn = contact2.avgNormal;
													if (contact2.contactVertexCount > 0 && nIn.squaredNorm() > 1e-12f) {
														nIn.normalize();
														agentLastContactNormalsIn[static_cast<size_t>(fi)] = nIn;
													} else {
														agentLastContactNormalsIn[static_cast<size_t>(fi)] = Eigen::Vector3f::Zero();
													}
												}
												anyContact2 = anyContact2 || (contact2.contactVertexCount > 0);
												maxPen2 = std::max(maxPen2, contact2.maxPenetration);
											}
										anyContact = anyContact || anyContact2;
										maxPenetrationThisFrame = std::max(maxPenetrationThisFrame, maxPen2);
									}
								}

								// Smooth (filter) contact force and normal for stable visualization/logging and
								// for less noisy normal-based force decomposition on the next frame.
								{
									const float tauF = std::max(0.0f, agentContactForceFilterTauSec);
									const float tauN = std::max(0.0f, agentContactNormalFilterTauSec);
									const float aF = (tauF > 0.0f) ? std::exp(-timeStep / std::max(1e-6f, tauF)) : 0.0f;
									const float aN = (tauN > 0.0f) ? std::exp(-timeStep / std::max(1e-6f, tauN)) : 0.0f;

									for (int fi = 0; fi < kFingerCount; ++fi) {
										const size_t idx = static_cast<size_t>(fi);

										// Force.
										const bool inContact = (agentLastContactCounts[idx] > 0);
										const Eigen::Vector3f rawF = inContact ? agentLastContactForcesN[idx] : Eigen::Vector3f::Zero();
										Eigen::Vector3f& f = agentFilteredContactForcesN[idx];
										if (tauF > 0.0f) f = f * aF + rawF * (1.0f - aF);
										else f = rawF;
										// For 1DOF haptics, eliminate "force tail" on release.
										if (!inContact) f.setZero();

										// Normal (unit, inward). Align sign for continuity before filtering.
										const Eigen::Vector3f rawN0 = agentLastContactNormalsIn[idx];
										Eigen::Vector3f& n = agentFilteredContactNormalsIn[idx];
										if (!inContact || rawN0.squaredNorm() <= 1e-12f) {
											if (tauN > 0.0f) n *= aN;
											else n.setZero();
											continue;
										}

										Eigen::Vector3f rawN = rawN0;
										if (n.squaredNorm() > 1e-12f && rawN.squaredNorm() > 1e-12f) {
											if (n.dot(rawN) < 0.0f) rawN = -rawN;
										}
										if (tauN > 0.0f) n = n * aN + rawN * (1.0f - aN);
										else n = rawN;
										if (n.squaredNorm() > 1e-12f) n.normalize();
									}
								}

								// Optional: grip/adhesion (tangential spring) to help "grab" and drag the surface.
								if (agentGripEnabledRuntime && agentUseSurfaceTriangles && !agentContactTriangles.empty()) {
									const float gripCorr = std::clamp(agentGripTangentCorrection, 0.0f, 1.0f);
									const float gripSlip = std::max(0.0f, agentGripSlipDistanceFrac) * r;
									const float gripMaxStep = std::max(0.0f, agentGripMaxTangentStepFrac) * r;
									const float gripMinPen = std::max(0.0f, agentGripMinPenetrationFrac) * r;
									const float invDt = 1.0f / std::max(1e-8f, timeStep);
									const float invMs = 1.0f / std::max(1e-8f, agentProxyMassKg);
									const float invMsResp = invMs * std::clamp(contactProxyInvMassScale, 0.0f, 1.0f);

									auto invMassOfPhysId = [&](int id) -> float {
										if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) return 0.0f;
										const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
										if (list.empty() || !list.front()) return 0.0f;
										if (list.front()->isFixed) return 0.0f;
										if (id < 0 || id >= static_cast<int>(physMassSumKg.size())) return 0.0f;
										const float m = physMassSumKg[static_cast<size_t>(id)];
										return (m > 1e-12f) ? (1.0f / m) : 0.0f;
									};

									auto applyDeltaToPhysId = [&](int id, const Eigen::Vector3f& dp) {
										if (dp.squaredNorm() <= 1e-24f) return;
										if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) return;
										const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
										for (Vertex* v : list) {
											if (!v || v->isFixed) continue;
											v->x += dp.x();
											v->y += dp.y();
											v->z += dp.z();
											v->velx += dp.x() * invDt;
											v->vely += dp.y() * invDt;
											v->velz += dp.z() * invDt;
										}
									};

									for (int fi = 0; fi < kFingerCount; ++fi) {
										const size_t idx = static_cast<size_t>(fi);
										if (agentLastContactCounts[idx] <= 0) {
											agentGripActive[idx] = false;
											agentGripTriangle[idx] = -1;
											continue;
										}

										const float pen = agentLastContactPenetrations[idx];
										if (pen < gripMinPen) {
											agentGripActive[idx] = false;
											agentGripTriangle[idx] = -1;
											continue;
										}

										const int preferredTi = agentLastActiveContactTriangle[idx];
										if (preferredTi < 0 || preferredTi >= static_cast<int>(agentContactTriangles.size())) {
											agentGripActive[idx] = false;
											agentGripTriangle[idx] = -1;
											continue;
										}

										if (!agentGripActive[idx]) {
											const auto& tri = agentContactTriangles[static_cast<size_t>(preferredTi)];
											if (!tri.a || !tri.b || !tri.c) continue;
											const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
											const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
											const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
											Eigen::Vector3f bary(0.0f, 0.0f, 0.0f);
											(void)closestPointOnTriangle(agentProxyPositions[idx], a, b, c, &bary);
											agentGripActive[idx] = true;
											agentGripTriangle[idx] = preferredTi;
											agentGripBary[idx] = bary;
										}

										if (!agentGripActive[idx]) continue;
										const int gripTi = agentGripTriangle[idx];
										if (gripTi < 0 || gripTi >= static_cast<int>(agentContactTriangles.size())) {
											agentGripActive[idx] = false;
											agentGripTriangle[idx] = -1;
											continue;
										}

										const auto& tri = agentContactTriangles[static_cast<size_t>(gripTi)];
										if (!tri.a || !tri.b || !tri.c) {
											agentGripActive[idx] = false;
											agentGripTriangle[idx] = -1;
											continue;
										}

										const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
										const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
										const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
										Eigen::Vector3f outwardN = Eigen::Vector3f::Zero();
										if (!outwardNormalForTriangle(tri, a, b, c, &outwardN)) continue;

										const Eigen::Vector3f bary = agentGripBary[idx];
										const Eigen::Vector3f anchor = a * bary.x() + b * bary.y() + c * bary.z();
										Eigen::Vector3f rel = agentProxyPositions[idx] - anchor;
										Eigen::Vector3f relT = rel - outwardN * rel.dot(outwardN);
										const float relTLen = relT.norm();
										if (gripSlip > 0.0f && relTLen > gripSlip) {
											agentGripActive[idx] = false;
											agentGripTriangle[idx] = -1;
											continue;
										}
										if (relTLen <= 1e-8f || gripCorr <= 0.0f) continue;

										Eigen::Vector3f dp = -relT * gripCorr;
										const float dpLen = dp.norm();
										if (gripMaxStep > 0.0f && dpLen > gripMaxStep) {
											dp *= (gripMaxStep / std::max(1e-12f, dpLen));
										}

										const auto& ids = agentContactTrianglePhysIds[static_cast<size_t>(gripTi)];
										const float w0 = bary.x();
										const float w1 = bary.y();
										const float w2 = bary.z();
										const float invMa = invMassOfPhysId(ids[0]);
										const float invMb = invMassOfPhysId(ids[1]);
										const float invMc = invMassOfPhysId(ids[2]);
										const float denom = (w0 * w0) * invMa + (w1 * w1) * invMb + (w2 * w2) * invMc + invMsResp;
										if (denom <= 1e-18f) continue;

										if (invMsResp > 0.0f) {
											const Eigen::Vector3f dpSphere = dp * (invMsResp / denom);
											agentProxyPositions[idx] += dpSphere;
											agentProxyVelocities[idx] += dpSphere * invDt;
										}
										applyDeltaToPhysId(ids[0], -dp * (w0 * invMa / denom));
										applyDeltaToPhysId(ids[1], -dp * (w1 * invMb / denom));
										applyDeltaToPhysId(ids[2], -dp * (w2 * invMc / denom));
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
											// Run every frame while the proxy is in the "suspicious" state (no contact but
											// plausibly inside). This avoids multi-frame "fully inside" states which
											// manifest as jittery force spikes when the solver re-acquires the surface.
											constexpr int kInsideTestIntervalFrames = 1;
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

												const bool wasContact = (agentLastContactCounts[static_cast<size_t>(fi)] > 0);
												const float vcCLen = wasContact ? agentVcCLenContact : agentVcCLenFree;

												Eigen::Vector3f nIn = agentFilteredContactNormalsIn[static_cast<size_t>(fi)];
												const bool hasN = (wasContact && nIn.squaredNorm() > 1e-12f);
												if (hasN) nIn.normalize();

												float kScaleN = 1.0f;
												float kScaleT = 1.0f;
												float cScaleN = 1.0f;
												float cScaleT = 1.0f;
												if (wasContact) {
													const float lastPen = agentLastContactPenetrations[static_cast<size_t>(fi)];
													const float penFrac = lastPen / std::max(1e-6f, agentSphere.radius);
													const float start = 0.20f;
													const float end = 0.50f;
													if (penFrac > start) {
														const float t = std::clamp((penFrac - start) / (end - start), 0.0f, 1.0f);
														const float kMinN = 0.35f;
														kScaleN = 1.0f - (1.0f - kMinN) * t;
														cScaleN = 1.0f + 1.5f * t;
													}
												}

												Eigen::Vector3f springForceN = Eigen::Vector3f::Zero();
												Eigen::Vector3f dampingForceN = Eigen::Vector3f::Zero();
												if (hasN) {
													const float dn = disp.dot(nIn);
													const Eigen::Vector3f dispN = nIn * dn;
													const Eigen::Vector3f dispT = disp - dispN;
													springForceN = (agentVcKLen * kScaleN) * dispN + (agentVcKLen * kScaleT) * dispT;

													const Eigen::Vector3f relV = (devVel - proxyVel);
													const float vn = relV.dot(nIn);
													const Eigen::Vector3f relVN = nIn * vn;
													const Eigen::Vector3f relVT = relV - relVN;
													dampingForceN = (vcCLen * cScaleN) * relVN + (vcCLen * cScaleT) * relVT;
												} else {
													springForceN = (agentVcKLen * kScaleN) * disp;
													dampingForceN = vcCLen * (devVel - proxyVel);
												}

												const Eigen::Vector3f couplingForceN = springForceN + dampingForceN;
												agentLastCouplingForcesN[static_cast<size_t>(fi)] = couplingForceN;

												// Output force (do NOT use proxyVel; only spring + device-velocity damping).
												Eigen::Vector3f dampingForceOut = Eigen::Vector3f::Zero();
												if (hasN) {
													const float vn = devVel.dot(nIn);
													const Eigen::Vector3f vN = nIn * vn;
													const Eigen::Vector3f vT = devVel - vN;
													dampingForceOut = (vcCLen * cScaleN) * vN + (vcCLen * cScaleT) * vT;
												} else {
													dampingForceOut = vcCLen * devVel;
												}
												agentLastDeviceForcesN[static_cast<size_t>(fi)] = -(springForceN + dampingForceOut);
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

									// Optional: gain/clamp the output force (for haptics). This does not affect simulation.
									{
										const float gain = std::max(0.0f, agentDeviceForceGain);
										const float maxN = std::max(0.0f, agentDeviceForceMaxN);
										if (gain != 1.0f || maxN > 0.0f) {
											for (int fi = 0; fi < kFingerCount; ++fi) {
												Eigen::Vector3f f = agentLastDeviceForcesN[static_cast<size_t>(fi)] * gain;
												if (maxN > 0.0f) {
													const float len = f.norm();
													if (len > maxN && len > 1e-12f) f *= (maxN / len);
												}
												agentLastDeviceForcesN[static_cast<size_t>(fi)] = f;
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
												const Eigen::Vector3f& contactForceRaw = agentLastContactForcesN[static_cast<size_t>(fi)];
												const Eigen::Vector3f& contactForce = agentFilteredContactForcesN[static_cast<size_t>(fi)];
												const int contacts = agentLastContactCounts[static_cast<size_t>(fi)];

												f << "finger" << idx << "_name " << kFingerNames[static_cast<size_t>(fi)] << "\n";
												f << "finger" << idx << "_devicePos " << devPos.x() << " " << devPos.y() << " " << devPos.z() << "\n";
												f << "finger" << idx << "_proxyPos " << proxyPos.x() << " " << proxyPos.y() << " " << proxyPos.z() << "\n";
												f << "finger" << idx << "_deviceForceN " << devForce.x() << " " << devForce.y() << " " << devForce.z() << "\n";
												f << "finger" << idx << "_contactForceN_raw " << contactForceRaw.x() << " " << contactForceRaw.y() << " " << contactForceRaw.z() << "\n";
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

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
					// Left-hand capsule collision: virtual-coupled proxy spheres against surface triangles.
					// Capsules are approximated as multiple spheres along each fingertip->palm direction.
					if (leftHandCapsulesEnabledRuntime && leapUseInput && leftHandWorldFresh &&
					    !leftHandDevicePositions.empty() &&
					    !agentContactTriangles.empty() &&
					    agentContactTrianglePhysIds.size() == agentContactTriangles.size() &&
					    !agentVerticesByPhysId.empty() &&
					    !physMassSumKg.empty()) {
						const float r = std::max(1e-6f, leftHandSphereRadius);
						const float maxPenFrac = std::clamp(leftHandMaxPenetrationFrac, 0.0f, 0.95f);
						const float allowedPen = maxPenFrac * r;
						const float corr = std::clamp(leftHandProxyPositionCorrection, 0.0f, 1.0f);
						const float tangentialDamp = std::clamp(leftHandCollisionTangentialDamp, 0.0f, 1.0f);
						const float contactVelRelax = std::clamp(leftHandContactVelocityRelaxation, 0.0f, 1.0f);
						const float contactVelRelaxMin = std::clamp(leftHandContactVelocityRelaxationMin, 0.0f, contactVelRelax);
						const float contactNormalDamp = std::clamp(leftHandContactNormalDamp, 0.0f, 1.0f);
						const int iters = std::clamp(leftHandCollisionIterations, 1, 64);
						const int manifoldK = std::clamp(leftHandContactManifoldTriangles, 1, 8);
						// Left hand is non-haptic: drive proxies kinematically (no VC integration) to avoid
						// laggy "fist opening" and collision-induced buzzing/spiraling.
						const float sphereInvMass = 0.0f;

						const std::vector<Eigen::Vector3f> proxyStart = leftHandProxyPositions;
						float maxProxyStep = 0.0f;
						for (size_t i = 0; i < proxyStart.size() && i < leftHandDevicePositions.size(); ++i) {
							const float step = (leftHandDevicePositions[i] - proxyStart[i]).norm();
							if (step > maxProxyStep) maxProxyStep = step;
						}
						const float epsBase = std::max(1e-4f * bboxDiag, 0.02f * r);
						const float epsSpeed = std::min(maxProxyStep, 2.0f * r);
						const float eps = epsBase + epsSpeed;
						const float rayEps = std::max(1e-9f, 1e-5f * bboxDiag);
						const float insideRejectDist = bboxDiag + 4.0f * r;

						bool anyContact = false;
						float maxPenetrationThisFrame = 0.0f;

						for (size_t si = 0; si < leftHandProxyPositions.size(); ++si) {
							Eigen::Vector3f& sphereCenter = leftHandProxyPositions[si];
							Eigen::Vector3f& sphereVel = leftHandProxyVelocities[si];
							const Eigen::Vector3f& devPos = leftHandDevicePositions[si];
							const Eigen::Vector3f& devVel = leftHandDeviceVelocities[si];

							const Eigen::Vector3f prev = sphereCenter;
							sphereCenter = devPos;
							sphereVel = devVel;
							(void)prev;

							AgentContactResult contact{};
							contact = solveAgentSphereTriangleCollisionConstraint(
								sphereCenter,
								sphereVel,
								sphereInvMass,
								/*proxyInvMassScale=*/0.0f,
								contactVelRelax,
								contactVelRelaxMin,
								contactNormalDamp,
								/*injectVelocityFromPositionCorrection=*/false,
								r,
								allowedPen,
								eps,
								timeStep,
								corr,
								tangentialDamp,
								leftHandFrictionMu,
								iters,
								manifoldK,
								&leftHandActiveContactTriangle[si],
								Eigen::Vector3f::Zero(),
								agentContactTriangles,
								agentContactTrianglePhysIds,
								agentContactTriangleNeighbors,
								agentVerticesByPhysId,
								physMassSumKg);

							// Rare anti-tunneling: if the proxy CENTER ends up fully inside the closed surface without
							// intersecting the contact shell, push it back out to avoid permanent penetration states.
							if (contact.contactVertexCount == 0) {
								const Eigen::Vector3f p = sphereCenter;
								if ((p - bboxCenter).norm() <= insideRejectDist) {
									if (isPointInsideSurfaceRayCastMulti(p, agentContactTriangles, rayEps)) {
										const AgentSurfaceQueryResult q = queryAgentSurface(p, agentContactTriangles);
										if (q.found && q.outwardNormal.squaredNorm() > 1e-12f && q.distanceToSurface > (r + epsBase)) {
											sphereCenter = q.closestPoint + q.outwardNormal * epsBase;
											const float vn = sphereVel.dot(q.outwardNormal);
											if (vn < 0.0f) sphereVel -= q.outwardNormal * vn;
										}
									}
								}
							}

							anyContact = anyContact || (contact.contactVertexCount > 0);
							maxPenetrationThisFrame = std::max(maxPenetrationThisFrame, contact.maxPenetration);
						}

						// For very deep presses, optionally stabilize volumes and re-apply contact once.
						if (tetVolumeConstraintEnabled && anyContact && maxPenetrationThisFrame > 0.25f * r) {
							const bool didVolume = applyTetVolumeStabilization(
								tetVolumeConstraintIterations,
								0.5f * tetVolumeConstraintCorrection);
							if (didVolume) {
								const int iters2 = std::clamp(std::max(4, iters / 2), 1, 64);
								for (size_t si = 0; si < leftHandProxyPositions.size(); ++si) {
									Eigen::Vector3f& sphereCenter = leftHandProxyPositions[si];
									Eigen::Vector3f& sphereVel = leftHandProxyVelocities[si];
									const Eigen::Vector3f& devPos = leftHandDevicePositions[si];
									const Eigen::Vector3f& devVel = leftHandDeviceVelocities[si];
									sphereCenter = devPos;
									sphereVel = devVel;

									(void)solveAgentSphereTriangleCollisionConstraint(
										sphereCenter,
										sphereVel,
										sphereInvMass,
										/*proxyInvMassScale=*/0.0f,
										contactVelRelax,
										contactVelRelaxMin,
										contactNormalDamp,
										/*injectVelocityFromPositionCorrection=*/false,
										r,
										allowedPen,
										eps,
										timeStep,
										corr,
										tangentialDamp,
										leftHandFrictionMu,
										iters2,
										manifoldK,
										&leftHandActiveContactTriangle[si],
										Eigen::Vector3f::Zero(),
										agentContactTriangles,
										agentContactTrianglePhysIds,
										agentContactTriangleNeighbors,
										agentVerticesByPhysId,
										physMassSumKg);
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
					}
#endif

					// "Walls" around the organ:
					// - Default: simple axis-aligned planes (Y±, X+).
					// - Optional: a static liver-shaped "abdominal cavity" boundary (rest pose surface inflated by a gap),
					//   leaving the exposed -X side open.
					if ((wallEnabled || cavity_enabled) && !agentVerticesByPhysId.empty()) {
						int wallHits = 0;

						// Collision mode selection:
						// 1) If cavity is enabled, use cavity collision only (or none if cavity collision is disabled).
						// 2) Legacy axis-aligned wall collision is used only when cavity is disabled.
						if (cavity_enabled && cavity_collision_enabled && !agentContactTriangles.empty() && !cavitySurfacePhysIds.empty()) {
							wallHits = applyLiverCavityConstraints(
								agentVerticesByPhysId,
								agentContactTriangles,
								agentContactTriangleNeighbors,
								cavityTriangleEnabled,
								cavitySurfacePhysIds,
								cavityActiveTriangleByPhysId,
								cavityGapWorld,
								cavityOpenAxis,
								cavityOpenLoWorld,
								cavityOpenHiWorld,
								timeStep,
								wallRestitution,
								wallTangentialDamp);
						} else if (!cavity_enabled && wallEnabled) {
							const Eigen::Vector3f extents = bboxMax - bboxMin;
							const Eigen::Vector3f margin = std::max(0.0f, wallMarginBboxScale) * extents;
							const float wallXMax = bboxMax.x() + margin.x();
							const float wallYMin = bboxMin.y() - margin.y();
							const float wallYMax = bboxMax.y() + margin.y();

							wallHits = applyAxisAlignedWallConstraints(
								agentVerticesByPhysId,
								wallXMax,
								wallYMin,
								wallYMax,
								timeStep,
								wallRestitution,
								wallTangentialDamp);
						}

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
			pipelineSimMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - pipelineSimBlockStart).count();
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

		// Render here
		const auto pipelineRenderStart = std::chrono::steady_clock::now();
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

		const float cameraDist = std::max(1e-6f, 1.5f * bboxDiag * zoomFactor);
		const float cameraLift = 0.0f;
			mat = Eigen::Matrix4f::Identity();
			const Eigen::Matrix3f viewRotation = rotation.toRotationMatrix();
			mat.block<3, 3>(0, 0) = viewRotation;
			const Eigen::Vector3f viewTranslation =
				Eigen::Vector3f(0.0f, cameraLift, -cameraDist) - viewRotation * bboxCenter;
			mat.block<3, 1>(0, 3) = viewTranslation;
			glMultMatrixf(mat.data());

					// Draw suspension ligaments (visual debug): line from patch center to wall anchor + endpoints.
					if (showSuspensionVisual && suspensionEnabled && !suspensions.empty() && !agentVerticesByPhysId.empty()) {
						const std::array<Eigen::Vector3f, 3> colors = {
							Eigen::Vector3f(1.00f, 0.20f, 0.20f), // susp1 red
							Eigen::Vector3f(0.20f, 1.00f, 0.20f), // susp2 green
							Eigen::Vector3f(0.20f, 0.60f, 1.00f)  // susp3 blue
						};

						glDisable(GL_LIGHTING);
						glLineWidth(3.0f);
						glBegin(GL_LINES);
						for (size_t si = 0; si < suspensions.size(); ++si) {
							const auto& s = suspensions[si];
							if (!s.enabled) continue;
							if (s.physIds.empty()) continue;

							Eigen::Vector3f c = Eigen::Vector3f::Zero();
							int n = 0;
							for (int id : s.physIds) {
								if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
								const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
								const Vertex* v0 = (!list.empty()) ? list.front() : nullptr;
								if (!v0) continue;
								c += Eigen::Vector3f(v0->x, v0->y, v0->z);
								++n;
							}
							if (n > 0) c /= static_cast<float>(n);
							else c = s.centerRest;

							const Eigen::Vector3f col = colors[std::min<size_t>(colors.size() - 1, si)];
							glColor3f(col.x(), col.y(), col.z());
							glVertex3f(s.anchorWorld.x(), s.anchorWorld.y(), s.anchorWorld.z());
							glVertex3f(c.x(), c.y(), c.z());
						}
						glEnd();

						glPointSize(10.0f);
						glBegin(GL_POINTS);
						for (size_t si = 0; si < suspensions.size(); ++si) {
							const auto& s = suspensions[si];
							if (!s.enabled) continue;
							const Eigen::Vector3f col = colors[std::min<size_t>(colors.size() - 1, si)];
							glColor3f(col.x(), col.y(), col.z());
							glVertex3f(s.anchorWorld.x(), s.anchorWorld.y(), s.anchorWorld.z());
							glVertex3f(s.centerRest.x(), s.centerRest.y(), s.centerRest.z());
						}
						glEnd();

						if (showSuspensionPatchPoints) {
							glPointSize(3.0f);
							glBegin(GL_POINTS);
							for (size_t si = 0; si < suspensions.size(); ++si) {
								const auto& s = suspensions[si];
								if (!s.enabled) continue;
								const Eigen::Vector3f col = colors[std::min<size_t>(colors.size() - 1, si)];
								glColor3f(col.x(), col.y(), col.z());
								for (int id : s.physIds) {
									if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
									const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
									const Vertex* v0 = (!list.empty()) ? list.front() : nullptr;
									if (!v0) continue;
									glVertex3f(v0->x, v0->y, v0->z);
								}
							}
							glEnd();
						}
					}

					// Visualize custom hard-fixed points clearly.
					if (showFixedPointVisual && !customFixedPhysIds.empty()) {
						glDisable(GL_LIGHTING);
						glEnable(GL_DEPTH_TEST);
						glDepthMask(GL_TRUE);
						glLineWidth(2.5f);
						glBegin(GL_LINES);
						glColor3f(1.0f, 0.2f, 0.2f);
						for (int id : customFixedPhysIds) {
							if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
							if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
							const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
							const Vertex* v0 = (!list.empty()) ? list.front() : nullptr;
							const Vertex* vr = physRep[static_cast<size_t>(id)];
							if (!v0 || !vr) continue;
							glVertex3f(v0->x, v0->y, v0->z);
							glVertex3f(vr->initx, vr->inity, vr->initz);
						}
						glEnd();

						glPointSize(14.0f);
						glBegin(GL_POINTS);
						glColor3f(1.0f, 0.0f, 0.0f);
						for (int id : customFixedPhysIds) {
							if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
							const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
							const Vertex* v0 = (!list.empty()) ? list.front() : nullptr;
							if (!v0) continue;
							glVertex3f(v0->x, v0->y, v0->z);
						}
						glEnd();

						// Rest anchors (init positions) of the custom ligament points.
						glPointSize(8.0f);
						glBegin(GL_POINTS);
						glColor3f(1.0f, 1.0f, 0.2f);
						for (int id : customFixedPhysIds) {
							if (id < 0 || id >= static_cast<int>(physRep.size())) continue;
							const Vertex* vr = physRep[static_cast<size_t>(id)];
							if (!vr) continue;
							glVertex3f(vr->initx, vr->inity, vr->initz);
						}
						glEnd();
					}

					// Draw agent sphere ("finger") device/proxy.
					if (agentSphere.enabled) {
						glLineWidth(2.0f);

					// Proxies (high-contrast palette + outline).
					const std::array<Eigen::Vector3f, kFingerCount> proxyColors = {
						Eigen::Vector3f(0.98f, 0.20f, 0.75f), // thumb (magenta)
						Eigen::Vector3f(0.10f, 0.90f, 0.95f), // index (cyan)
						Eigen::Vector3f(1.00f, 0.90f, 0.15f), // middle (yellow)
						Eigen::Vector3f(0.20f, 1.00f, 0.30f), // ring (lime)
						Eigen::Vector3f(1.00f, 0.50f, 0.10f)  // pinky (orange)
					};
					// Visual-only: draw right-hand proxies as capsules with the SAME radius/length style
					// as left-hand capsules. Physics/contact algorithm for the right hand is unchanged.
					const int rightVisSamples = std::clamp(leftHandCapsuleSamples, 2, 12);
					const float rightVisRadius = std::max(1e-6f, leftHandCapsuleRadiusBboxScale * bboxDiag);
					const float rightVisCapsuleLen = std::max(0.0f, leftHandCapsuleLengthBboxScale * bboxDiag);
					for (int fi = 0; fi < kFingerCount; ++fi) {
						Eigen::Vector3f c = proxyColors[static_cast<size_t>(fi)];
						if (whiteBackground) c *= 0.80f;
						const float matScale = materialScaleAtWorldPoint(agentProxyPositions[static_cast<size_t>(fi)]);
						if (showMaterialOverrideOverlay && matScale > 1.05f) {
							// Hard side: show as white proxy for quick confirmation when sliding.
							c = whiteBackground ? Eigen::Vector3f(0.15f, 0.15f, 0.15f) : Eigen::Vector3f(0.98f, 0.98f, 0.98f);
						}
						
						const Eigen::Vector3f& pos = agentProxyPositions[static_cast<size_t>(fi)];
						const Eigen::Quaternionf& rot = agentDeviceRotations[static_cast<size_t>(fi)];
						// Align right-hand visual capsule axis with fingertip distal direction.
						// Using -Z here matches the left-hand capsule orientation in current Leap frame convention.
						Eigen::Vector3f dir = rot * (-Eigen::Vector3f::UnitZ());
						const float dlen = dir.norm();
						if (dlen > 1e-8f) dir /= dlen;
						else dir = -Eigen::Vector3f::UnitY();
						const Eigen::Vector3f base = pos - dir * rightVisCapsuleLen;

						glBegin(GL_LINE_STRIP);
						glColor3f(c.x(), c.y(), c.z());
						for (int si = 0; si < rightVisSamples; ++si) {
							const float t = (rightVisSamples > 1) ? (static_cast<float>(si) / static_cast<float>(rightVisSamples - 1)) : 1.0f;
							const Eigen::Vector3f p = base + t * (pos - base);
							glVertex3f(p.x(), p.y(), p.z());
						}
						glEnd();

						for (int si = 0; si < rightVisSamples; ++si) {
							const float t = (rightVisSamples > 1) ? (static_cast<float>(si) / static_cast<float>(rightVisSamples - 1)) : 1.0f;
							const Eigen::Vector3f p = base + t * (pos - base);
							glPushMatrix();
							glTranslatef(p.x(), p.y(), p.z());
							glColor3f(c.x(), c.y(), c.z());
							drawWireSphereCircles(Eigen::Vector3f::Zero(), rightVisRadius, 18);
							glPopMatrix();
						}
					}
				}

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
				// Draw left-hand capsules using PHYSICS PROXIES, so what you see is what collides.
				// NOTE: draw even if this frame's Leap data is stale; otherwise the hand "disappears" on brief occlusion
				// or immediately after toggling Leap input. Collision still requires fresh data elsewhere.
				if (leftHandCapsulesEnabledRuntime && !leftHandProxyPositions.empty()) {
					glLineWidth(2.0f);
					const float r = std::max(1e-6f, leftHandSphereRadius);
					const std::array<Eigen::Vector3f, kFingerCount> colors = {
						Eigen::Vector3f(0.98f, 0.20f, 0.75f), // thumb
						Eigen::Vector3f(0.10f, 0.90f, 0.95f), // index
						Eigen::Vector3f(1.00f, 0.90f, 0.15f), // middle
						Eigen::Vector3f(0.20f, 1.00f, 0.30f), // ring
						Eigen::Vector3f(1.00f, 0.50f, 0.10f)  // pinky
					};
					for (int fi = 0; fi < kFingerCount; ++fi) {
						Eigen::Vector3f c = colors[static_cast<size_t>(fi)];
						if (whiteBackground) c *= 0.80f;

						glBegin(GL_LINE_STRIP);
						glColor3f(c.x(), c.y(), c.z());
						for (int si = 0; si < leftHandSamplesClamped; ++si) {
							const size_t idx = static_cast<size_t>(fi * leftHandSamplesClamped + si);
							if (idx >= leftHandProxyPositions.size()) continue;
							const Eigen::Vector3f& pos = leftHandProxyPositions[idx];
							glVertex3f(pos.x(), pos.y(), pos.z());
						}
						glEnd();

						for (int si = 0; si < leftHandSamplesClamped; ++si) {
							const size_t idx = static_cast<size_t>(fi * leftHandSamplesClamped + si);
							if (idx >= leftHandProxyPositions.size()) continue;
							const Eigen::Vector3f& pos = leftHandProxyPositions[idx];
							glPushMatrix();
							glTranslatef(pos.x(), pos.y(), pos.z());
							glColor3f(c.x(), c.y(), c.z());
							drawWireSphereCircles(Eigen::Vector3f::Zero(), r, 18);
							glPopMatrix();
						}
					}
				}
#endif

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

			// Draw walls around the organ.
			if (showCavityWallVisual && cavity_enabled && !agentContactTriangles.empty() && !cavityTriangleEnabled.empty()) {
				const bool blendWasEnabled = (glIsEnabled(GL_BLEND) == GL_TRUE);
				const bool depthWasEnabled = (glIsEnabled(GL_DEPTH_TEST) == GL_TRUE);
				const bool lightingWasEnabled = (glIsEnabled(GL_LIGHTING) == GL_TRUE);
				const bool light0WasEnabled = (glIsEnabled(GL_LIGHT0) == GL_TRUE);
				const bool colorMatWasEnabled = (glIsEnabled(GL_COLOR_MATERIAL) == GL_TRUE);
				GLint lightModelTwoSideWas = GL_FALSE;
				glGetIntegerv(GL_LIGHT_MODEL_TWO_SIDE, &lightModelTwoSideWas);
				glEnable(GL_DEPTH_TEST);
				// Write depth so the cavity shell can correctly occlude liver parts behind it.
				glDepthMask(GL_TRUE);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
				const bool cavitySmoothLit = showLiverSmoothRender;
				if (cavitySmoothLit) {
					// Smooth mode: render cavity as an opaque two-sided lit shell.
					glDisable(GL_BLEND);
					glDisable(GL_CULL_FACE);
					glEnable(GL_LIGHTING);
					glEnable(GL_LIGHT0);
					glEnable(GL_COLOR_MATERIAL);
					glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
					glShadeModel(GL_SMOOTH);
					glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
					const GLfloat lightAmbient[] = { 0.18f, 0.18f, 0.20f, 1.0f };
					const GLfloat lightDiffuse[] = { 0.88f, 0.90f, 0.94f, 1.0f };
					const GLfloat lightSpecular[] = { 0.28f, 0.28f, 0.30f, 1.0f };
					const GLfloat lightPos[] = { 0.35f, 0.90f, 0.55f, 0.0f };
					glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
					glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
					glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);
					glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
					const GLfloat matSpec[] = { 0.15f, 0.15f, 0.18f, 1.0f };
					glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matSpec);
					glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 14.0f);
					if (whiteBackground) glColor4f(0.62f, 0.76f, 0.96f, 1.0f);
					else glColor4f(0.54f, 0.68f, 0.94f, 1.0f);
				} else {
					glEnable(GL_BLEND);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					glDisable(GL_CULL_FACE);
					// Keep cavity shell subtle: ~20% visibility for cleaner surgical-view aesthetics.
					if (whiteBackground) glColor4f(0.16f, 0.28f, 0.62f, 0.20f);
					else glColor4f(0.58f, 0.72f, 0.98f, 0.20f);
				}

				glBegin(GL_TRIANGLES);
				const float gap = std::max(0.0f, cavityGapWorld);
				// Visual-only extra offset so the cavity shell reads as a continuous outer wall
				// instead of fragmented patches caused by near-overlap with the liver surface.
				const float visualGap = gap + std::max(0.015f * bboxDiag, 0.0f);
				for (int ti = 0; ti < static_cast<int>(agentContactTriangles.size()); ++ti) {
					if (!cavityTriangleEnabled[static_cast<size_t>(ti)]) continue;
					const AgentTriangle& tri = agentContactTriangles[static_cast<size_t>(ti)];
					if (!tri.a || !tri.b || !tri.c) continue;
					const Eigen::Vector3f a(tri.a->initx, tri.a->inity, tri.a->initz);
					const Eigen::Vector3f b(tri.b->initx, tri.b->inity, tri.b->initz);
					const Eigen::Vector3f c(tri.c->initx, tri.c->inity, tri.c->initz);
					Eigen::Vector3f nFace = Eigen::Vector3f::Zero();
					if (!outwardNormalForTriangleInit(tri, a, b, c, &nFace)) continue;

					Eigen::Vector3f na = nFace;
					Eigen::Vector3f nb = nFace;
					Eigen::Vector3f nc = nFace;
					if (!agentContactTrianglePhysIds.empty() &&
					    agentContactTrianglePhysIds.size() == agentContactTriangles.size() &&
					    !cavityVertexNormalByPhysId.empty()) {
						const auto& ids = agentContactTrianglePhysIds[static_cast<size_t>(ti)];
						const int ia = ids[0], ib = ids[1], ic = ids[2];
						if (ia >= 0 && ia < static_cast<int>(cavityVertexNormalByPhysId.size()) &&
						    cavityVertexNormalByPhysId[static_cast<size_t>(ia)].squaredNorm() > 1e-12f) {
							na = cavityVertexNormalByPhysId[static_cast<size_t>(ia)];
						}
						if (ib >= 0 && ib < static_cast<int>(cavityVertexNormalByPhysId.size()) &&
						    cavityVertexNormalByPhysId[static_cast<size_t>(ib)].squaredNorm() > 1e-12f) {
							nb = cavityVertexNormalByPhysId[static_cast<size_t>(ib)];
						}
						if (ic >= 0 && ic < static_cast<int>(cavityVertexNormalByPhysId.size()) &&
						    cavityVertexNormalByPhysId[static_cast<size_t>(ic)].squaredNorm() > 1e-12f) {
							nc = cavityVertexNormalByPhysId[static_cast<size_t>(ic)];
						}
					}

					Eigen::Vector3f ao = a + na * visualGap;
					Eigen::Vector3f bo = b + nb * visualGap;
					Eigen::Vector3f co = c + nc * visualGap;
					// Enforce consistent winding for stable back-face culling in smooth mode.
					if (cavitySmoothLit) {
						const Eigen::Vector3f faceGeom = (bo - ao).cross(co - ao);
						if (faceGeom.dot(nFace) < 0.0f) {
							std::swap(bo, co);
							std::swap(nb, nc);
						}
					}
					if (cavitySmoothLit) glNormal3f(na.x(), na.y(), na.z());
					glVertex3f(ao.x(), ao.y(), ao.z());
					if (cavitySmoothLit) glNormal3f(nb.x(), nb.y(), nb.z());
					glVertex3f(bo.x(), bo.y(), bo.z());
					if (cavitySmoothLit) glNormal3f(nc.x(), nc.y(), nc.z());
					glVertex3f(co.x(), co.y(), co.z());
				}
				glEnd();

				if (cavitySmoothLit) {
					if (!colorMatWasEnabled) glDisable(GL_COLOR_MATERIAL);
					if (!light0WasEnabled) glDisable(GL_LIGHT0);
					if (!lightingWasEnabled) glDisable(GL_LIGHTING);
					glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, lightModelTwoSideWas);
				}
				glDisable(GL_CULL_FACE);
				glDepthMask(GL_TRUE);
				if (!depthWasEnabled) glDisable(GL_DEPTH_TEST);
				if (!blendWasEnabled) glDisable(GL_BLEND);
				if (!showVolumePreservation) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			// Fallback: simple axis-aligned walls (Y±, X+) around the initial bbox.
			// Use only when cavity mode is disabled.
			else if (!cavity_enabled && wallEnabled) {
				const Eigen::Vector3f extents = bboxMax - bboxMin;
				const Eigen::Vector3f margin = std::max(0.0f, wallMarginBboxScale) * extents;
				const float x0 = bboxMin.x() - margin.x();
				const float x1 = bboxMax.x() + margin.x();
				const float y0 = bboxMin.y() - margin.y();
				const float y1 = bboxMax.y() + margin.y();
				const float z0 = bboxMin.z() - margin.z();
				const float z1 = bboxMax.z() + margin.z();

				const bool blendWasEnabled = (glIsEnabled(GL_BLEND) == GL_TRUE);
				const bool depthWasEnabled = (glIsEnabled(GL_DEPTH_TEST) == GL_TRUE);
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
				glDisable(GL_DEPTH_TEST);
				glDepthMask(GL_FALSE);
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

				if (whiteBackground) {
					glColor4f(0.1f, 0.2f, 0.6f, 0.24f);
				} else {
					glColor4f(0.6f, 0.7f, 1.0f, 0.24f);
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
				if (depthWasEnabled) glEnable(GL_DEPTH_TEST);
				if (!blendWasEnabled) glDisable(GL_BLEND);
				if (!showVolumePreservation) {
					glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
				}
			}
			
			// Draw vertices (debug). Skip in smooth render + stress/volume modes for cleaner visualization.
			if (!showLiverSmoothRender && !showStressCloud && !showVolumePreservation) {
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

			const bool canSmoothRender =
				showLiverSmoothRender &&
				!showStressCloud && !showExplodedView && !showFiberFlow && !showGhostLinks && !showVolumePreservation &&
				!agentContactTriangles.empty() &&
				agentContactTrianglePhysIds.size() == agentContactTriangles.size() &&
				!physRep.empty();

			if (canSmoothRender) {
				// Smooth shaded surface rendering from extracted outer surface triangles.
				std::vector<Eigen::Vector3f> nByPhys(static_cast<size_t>(physRep.size()), Eigen::Vector3f::Zero());
				for (size_t ti = 0; ti < agentContactTriangles.size(); ++ti) {
					const auto& tri = agentContactTriangles[ti];
					const auto& ids = agentContactTrianglePhysIds[ti];
					if (!tri.a || !tri.b || !tri.c) continue;
					const Eigen::Vector3f a(tri.a->x, tri.a->y, tri.a->z);
					const Eigen::Vector3f b(tri.b->x, tri.b->y, tri.b->z);
					const Eigen::Vector3f c(tri.c->x, tri.c->y, tri.c->z);
					Eigen::Vector3f nRaw = (b - a).cross(c - a);
					if (nRaw.squaredNorm() <= 1e-18f) continue;
					Eigen::Vector3f outN = Eigen::Vector3f::Zero();
					if (outwardNormalForTriangle(tri, a, b, c, &outN)) {
						if (outN.dot(nRaw) < 0.0f) nRaw = -nRaw;
					}
					for (int k = 0; k < 3; ++k) {
						const int pid = ids[static_cast<size_t>(k)];
						if (pid < 0 || pid >= static_cast<int>(nByPhys.size())) continue;
						nByPhys[static_cast<size_t>(pid)] += nRaw;
					}
				}
				for (auto& n : nByPhys) {
					const float l2 = n.squaredNorm();
					if (l2 > 1e-18f) n /= std::sqrt(l2);
				}

				glDisable(GL_BLEND);
				glEnable(GL_LIGHTING);
				glEnable(GL_LIGHT0);
				glEnable(GL_COLOR_MATERIAL);
				glShadeModel(GL_SMOOTH);
				glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

				const GLfloat lightAmbient[] = { 0.20f, 0.20f, 0.22f, 1.0f };
				const GLfloat lightDiffuse[] = { 0.92f, 0.92f, 0.95f, 1.0f };
				const GLfloat lightSpecular[] = { 0.35f, 0.35f, 0.35f, 1.0f };
				const GLfloat lightPos[] = { 0.35f, 0.90f, 0.55f, 0.0f }; // directional (view space)
				glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbient);
				glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuse);
				glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecular);
				glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

				const GLfloat matSpec[] = { 0.18f, 0.18f, 0.18f, 1.0f };
				glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, matSpec);
				glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 18.0f);

				const Eigen::Vector3f baseCol = whiteBackground ? Eigen::Vector3f(0.60f, 0.22f, 0.18f) : Eigen::Vector3f(0.72f, 0.28f, 0.22f);
				glBegin(GL_TRIANGLES);
				for (size_t ti = 0; ti < agentContactTriangles.size(); ++ti) {
					const auto& tri = agentContactTriangles[ti];
					const auto& ids = agentContactTrianglePhysIds[ti];
					if (!tri.a || !tri.b || !tri.c) continue;
					const Vertex* vs[3] = { tri.a, tri.b, tri.c };
					for (int k = 0; k < 3; ++k) {
						const int pid = ids[static_cast<size_t>(k)];
						Eigen::Vector3f n = Eigen::Vector3f::Zero();
						if (pid >= 0 && pid < static_cast<int>(nByPhys.size())) n = nByPhys[static_cast<size_t>(pid)];
						if (n.squaredNorm() <= 1e-12f) {
							const Eigen::Vector3f a(vs[0]->x, vs[0]->y, vs[0]->z);
							const Eigen::Vector3f b(vs[1]->x, vs[1]->y, vs[1]->z);
							const Eigen::Vector3f c(vs[2]->x, vs[2]->y, vs[2]->z);
							Eigen::Vector3f outN = Eigen::Vector3f::Zero();
							if (outwardNormalForTriangle(tri, a, b, c, &outN)) n = outN;
						}
						if (n.squaredNorm() > 1e-12f) glNormal3f(n.x(), n.y(), n.z());

						Eigen::Vector3f col = baseCol;
						if (showMaterialOverrideOverlay) {
							Eigen::Vector3f p(vs[k]->x, vs[k]->y, vs[k]->z);
							if (pid >= 0 && pid < static_cast<int>(physRep.size())) {
								const Vertex* vInit = physRep[static_cast<size_t>(pid)];
								if (vInit) p = Eigen::Vector3f(vInit->initx, vInit->inity, vInit->initz);
							}
							const float matScale = materialScaleAtWorldPoint(p);
							if (matScale > 1.05f) col = whiteBackground ? Eigen::Vector3f(0.20f, 0.20f, 0.20f) : Eigen::Vector3f(0.98f, 0.98f, 0.98f);
						}
						glColor3f(col.x(), col.y(), col.z());
						glVertex3f(vs[k]->x, vs[k]->y, vs[k]->z);
					}
				}
				glEnd();

				glDisable(GL_COLOR_MATERIAL);
				glDisable(GL_LIGHT0);
				glDisable(GL_LIGHTING);
			} else {
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
							const bool hardGroup = (std::abs(effectiveYoungsForGroup(groupIdx, youngs) - youngs) > 1e-3f);
							// Highlight any Young's modulus override region (e.g. "tumor" patch) in the default view.
							if (showMaterialOverrideOverlay && hardGroup && !showStressCloud && !showVolumePreservation) {
								glColor4f(1.0f, 1.0f, 1.0f, alpha);
								return;
							}
							if (showVolumePreservation) {
								float volumeRatio = (initialVolume > 1e-6f) ? (currentVolume / initialVolume) : 1.0f;
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

						setVertexColor(v[0]); glVertex3f(v[0]->x + offset.x(), v[0]->y + offset.y(), v[0]->z + offset.z());
						setVertexColor(v[1]); glVertex3f(v[1]->x + offset.x(), v[1]->y + offset.y(), v[1]->z + offset.z());
						setVertexColor(v[2]); glVertex3f(v[2]->x + offset.x(), v[2]->y + offset.y(), v[2]->z + offset.z());

						setVertexColor(v[0]); glVertex3f(v[0]->x + offset.x(), v[0]->y + offset.y(), v[0]->z + offset.z());
						setVertexColor(v[1]); glVertex3f(v[1]->x + offset.x(), v[1]->y + offset.y(), v[1]->z + offset.z());
						setVertexColor(v[3]); glVertex3f(v[3]->x + offset.x(), v[3]->y + offset.y(), v[3]->z + offset.z());

						setVertexColor(v[0]); glVertex3f(v[0]->x + offset.x(), v[0]->y + offset.y(), v[0]->z + offset.z());
						setVertexColor(v[2]); glVertex3f(v[2]->x + offset.x(), v[2]->y + offset.y(), v[2]->z + offset.z());
						setVertexColor(v[3]); glVertex3f(v[3]->x + offset.x(), v[3]->y + offset.y(), v[3]->z + offset.z());

						setVertexColor(v[1]); glVertex3f(v[1]->x + offset.x(), v[1]->y + offset.y(), v[1]->z + offset.z());
						setVertexColor(v[2]); glVertex3f(v[2]->x + offset.x(), v[2]->y + offset.y(), v[2]->z + offset.z());
						setVertexColor(v[3]); glVertex3f(v[3]->x + offset.x(), v[3]->y + offset.y(), v[3]->z + offset.z());
					}
				}
				glEnd();
			}

			// (no debug split plane)
			
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
				const auto pipelineHapticStart = std::chrono::steady_clock::now();
				Eigen::Vector3f fN = Eigen::Vector3f::Zero();
				if (agentForceGraphMode != 0) {
					// DEVICE force (haptic output; already filtered/gain/clamped).
					fN = agentLastDeviceForcesN[static_cast<size_t>(kForceGraphFingerIndex)];
				} else {
					// CONTACT force (filtered; raw PBD-derived contact force is too noisy to use directly).
					fN = agentFilteredContactForcesN[static_cast<size_t>(kForceGraphFingerIndex)];
				}

				if (haptic_uart_enabled) {
					// Tumor-specific "impact" vibration: a short burst when entering contact on a hard region.
					// This makes stiffness differences perceptible even for 1DOF devices (no lateral force).
					static double lastHapticTimeSec = 0.0;
					const double nowHapticTimeSec = glfwGetTime();
					float hapticDtSec = 0.0f;
					if (lastHapticTimeSec > 0.0) hapticDtSec = static_cast<float>(nowHapticTimeSec - lastHapticTimeSec);
					lastHapticTimeSec = nowHapticTimeSec;
					hapticDtSec = std::clamp(hapticDtSec, 0.0f, 0.05f);
					static std::array<float, 5> vibTimeLeftSec{}; // 0=thumb..4=pinky
					static std::array<float, 5> vibPhaseRad{};
					static std::array<bool, 5> prevInContact{};
					static std::array<bool, 5> prevHardRegion{};
					constexpr float kTwoPi = 6.28318530718f;

					// Send force for each configured finger->motor pair
					// Finger 0 = Thumb  -> Motor 0
					// Finger 1 = Index  -> Motor 1
					struct FingerMotorPair { int fingerIdx; int motorId; };
					const FingerMotorPair pairs[] = {
						{ 0, haptic_uart_thumb_motor_id },  // Thumb
						{ 1, haptic_uart_motor_id },        // Index
						{ 2, haptic_uart_middle_motor_id }, // Middle
						{ 3, haptic_uart_ring_motor_id },   // Ring
					};
					for (const auto& pair : pairs) {
						if (pair.motorId < 0) continue; // Skip if disabled (-1)
						
						Eigen::Vector3f contactF = agentFilteredContactForcesN[static_cast<size_t>(pair.fingerIdx)];
						// 1DOF devices: output the NORMAL reaction magnitude.
						// Using finger rotation to define "pad normal" is often wrong/noisy for 1DOF hardware.
						Eigen::Vector3f nIn = agentFilteredContactNormalsIn[static_cast<size_t>(pair.fingerIdx)];
						// Filtered normals can briefly go to zero; fall back to the most recent raw normal.
						if (nIn.squaredNorm() < 1e-12f) nIn = agentLastContactNormalsIn[static_cast<size_t>(pair.fingerIdx)];
						float forceMag = 0.0f;
						const float pen = agentLastContactPenetrations[static_cast<size_t>(pair.fingerIdx)];
						const bool inContact = (pen > 0.0f);
						if (inContact) {
							// Some contact cases can flip the normal sign; for 1DOF magnitude output we want the
							// normal-component magnitude, not a signed value that can clamp to 0 ("no force")
							// when pressing at an angle.
							if (nIn.squaredNorm() > 1e-12f) {
								nIn.normalize();
								const float dotAbs = std::abs(contactF.dot(nIn));
								const float fNorm = contactF.norm();
								// If the normal is degenerate/misaligned, fall back to magnitude.
								forceMag = (dotAbs > 1e-6f * fNorm) ? dotAbs : fNorm;
							} else {
								forceMag = contactF.norm();
							}
						}

						// Overall output gain (separate from simulation/contact).
						forceMag *= std::max(0.0f, agentDeviceForceGain);

						// Simple, brutal 1DOF effect: amplify output when sampling over a locally stiffer region.
						const float matScale = materialScaleAtWorldPoint(agentProxyPositions[static_cast<size_t>(pair.fingerIdx)]);
						const bool hardRegion = (matScale > 1.05f);
						{
							const float hardGain = std::max(0.0f, agentDeviceForceHardGain);
							if (hardRegion && hardGain != 1.0f) forceMag *= hardGain;
						}

						// Soft-clip to preserve dynamic range: avoids "touch -> max PWM" for both soft/hard.
						if (haptic_softclip_enabled) {
							const float maxF = std::max(1e-6f, haptic_max_force_input);
							const float knee = std::max(1e-6f, haptic_softclip_knee);
							const float f = std::max(0.0f, forceMag);
							forceMag = maxF * (f / (f + knee));
						}

						// Tumor "impact" vibration burst on contact start/entry into hard region.
						// Modulate after soft-clip so the vibration isn't crushed by the nonlinearity.
						if (haptic_tumor_vib_enabled) {
							const int fi = pair.fingerIdx;
							if (!inContact) {
								vibTimeLeftSec[static_cast<size_t>(fi)] = 0.0f;
								vibPhaseRad[static_cast<size_t>(fi)] = 0.0f;
							} else {
								const bool startBurst = hardRegion && (!prevInContact[static_cast<size_t>(fi)] || !prevHardRegion[static_cast<size_t>(fi)]);
								if (startBurst) {
									if (vibTimeLeftSec[static_cast<size_t>(fi)] <= 0.0f) {
										vibTimeLeftSec[static_cast<size_t>(fi)] = std::max(0.0f, haptic_tumor_vib_duration_sec);
										vibPhaseRad[static_cast<size_t>(fi)] = 0.0f;
									}
								}

								float& tLeft = vibTimeLeftSec[static_cast<size_t>(fi)];
								float& phase = vibPhaseRad[static_cast<size_t>(fi)];
								if (tLeft > 0.0f && hapticDtSec > 0.0f) {
									const float dur = std::max(1e-4f, haptic_tumor_vib_duration_sec);
									const float elapsed = std::max(0.0f, dur - tLeft);
									// Cosine fade from 1->0 over the burst.
									const float u = std::clamp(elapsed / dur, 0.0f, 1.0f);
									const float env = 0.5f * (1.0f + std::cos(PI * u));
									const float freq = std::max(0.0f, haptic_tumor_vib_freq_hz);
									const float amp = std::max(0.0f, haptic_tumor_vib_amp);
									const float s = std::sin(phase);
									// 1DOF cable devices usually only "pull". Use a unipolar modulation.
									const float vib = amp * env * (0.5f * (1.0f + s));
									forceMag = std::max(0.0f, forceMag + vib);
									phase += kTwoPi * freq * hapticDtSec;
									if (phase > kTwoPi) phase = std::fmod(phase, kTwoPi);
									tLeft = std::max(0.0f, tLeft - hapticDtSec);
								}
							}

							prevInContact[static_cast<size_t>(fi)] = inContact;
							prevHardRegion[static_cast<size_t>(fi)] = hardRegion;
						}

						// Clamp to configured range before mapping->PWM.
						forceMag = std::min(forceMag, std::max(0.0f, haptic_max_force_input));
						// "Pop" (fast cable tightening) feels like hardness. Suppress it on soft tissue by
						// applying the slew limiter only on soft regions; allow hard regions to respond fast.
						haptic.sendForce(pair.motorId, forceMag, /*bypassSlew=*/hardRegion);
					}
				}

				pipelineHapticTxMs = std::chrono::duration<double, std::milli>(
					std::chrono::steady_clock::now() - pipelineHapticStart).count();
				pipelineRenderMs = std::chrono::duration<double, std::milli>(
					pipelineHapticStart - pipelineRenderStart).count();

				agentForceHistory.push(Eigen::Vector4f(fN.x(), fN.y(), fN.z(), fN.norm()));
			} else {
				pipelineRenderMs = std::chrono::duration<double, std::milli>(
					std::chrono::steady_clock::now() - pipelineRenderStart).count();
			}

			// Draw suspension overlays again as a foreground pass so they are not hidden by the liver surface.
			// Fixed-point markers are intentionally excluded here: they are already drawn once with depth
			// testing enabled above, and redrawing them in screen-space overlay makes them incorrectly float
			// on top of the liver even when they should be occluded.
			if (showSuspensionVisual && !agentVerticesByPhysId.empty()) {
				const bool depthWasEnabled = (glIsEnabled(GL_DEPTH_TEST) == GL_TRUE);
				const bool lightingWasEnabled = (glIsEnabled(GL_LIGHTING) == GL_TRUE);
				glDisable(GL_DEPTH_TEST);
				glDisable(GL_LIGHTING);
				glDepthMask(GL_FALSE);

				if (showSuspensionVisual && suspensionEnabled && !suspensions.empty()) {
					const std::array<Eigen::Vector3f, 3> colors = {
						Eigen::Vector3f(1.00f, 0.20f, 0.20f),
						Eigen::Vector3f(0.20f, 1.00f, 0.20f),
						Eigen::Vector3f(0.20f, 0.60f, 1.00f)
					};
					glLineWidth(3.5f);
					glBegin(GL_LINES);
					for (size_t si = 0; si < suspensions.size(); ++si) {
						const auto& s = suspensions[si];
						if (!s.enabled || s.physIds.empty()) continue;
						Eigen::Vector3f c = Eigen::Vector3f::Zero();
						int n = 0;
						for (int id : s.physIds) {
							if (id < 0 || id >= static_cast<int>(agentVerticesByPhysId.size())) continue;
							const auto& list = agentVerticesByPhysId[static_cast<size_t>(id)];
							const Vertex* v0 = (!list.empty()) ? list.front() : nullptr;
							if (!v0) continue;
							c += Eigen::Vector3f(v0->x, v0->y, v0->z);
							++n;
						}
						c = (n > 0) ? (c / static_cast<float>(n)) : s.centerRest;
						const Eigen::Vector3f col = colors[std::min<size_t>(colors.size() - 1, si)];
						glColor3f(col.x(), col.y(), col.z());
						glVertex3f(s.anchorWorld.x(), s.anchorWorld.y(), s.anchorWorld.z());
						glVertex3f(c.x(), c.y(), c.z());
					}
					glEnd();
				}

				glDepthMask(GL_TRUE);
				if (depthWasEnabled) glEnable(GL_DEPTH_TEST);
				if (lightingWasEnabled) glEnable(GL_LIGHTING);
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

#if defined(TETFEM_HAVE_LEAPC) && TETFEM_HAVE_LEAPC
		// Leap alignment: choose which hand QWEASD affects.
		const SimpleUI::Rect uiLeapOffsetTargetRect{ uiMargin, uiMargin + 4.0f * (uiH + 8.0f), uiW, uiH };
		const char* offsetLabel = (leapOffsetTarget == LeapOffsetTarget::Right) ? "Offset Hand: RIGHT" : "Offset Hand: LEFT";
		if (ui.button(uiLeapOffsetTargetRect, offsetLabel)) {
			leapOffsetTarget = (leapOffsetTarget == LeapOffsetTarget::Right) ? LeapOffsetTarget::Left : LeapOffsetTarget::Right;
		}
#endif

		// Right side buttons
		const float rightMargin = ui.state().windowWidth - uiW - uiMargin;
		const SimpleUI::Rect uiBgColorRect{ rightMargin, uiMargin, uiW, uiH };
		if (ui.button(uiBgColorRect, whiteBackground ? "Dark Background" : "White Background")) {
			whiteBackground = !whiteBackground;
		}

		const SimpleUI::Rect uiMaterialOverlayRect{ rightMargin, uiMargin + uiH + 8.0f, uiW, uiH };
		if (ui.button(uiMaterialOverlayRect, showMaterialOverrideOverlay ? "Hide Tumor" : "Show Tumor")) {
			showMaterialOverrideOverlay = !showMaterialOverrideOverlay;
		}

		const SimpleUI::Rect uiTumorPosRect{ rightMargin, uiMargin + 11.0f * (uiH + 8.0f), uiW, uiH };
		const char* tumorModeLabel = "Tumor: OFF";
		if (tumorModeIndex == 1) tumorModeLabel = "Tumor: POS 1";
		else if (tumorModeIndex == 2) tumorModeLabel = "Tumor: POS 2";
		else if (tumorModeIndex == 3) tumorModeLabel = "Tumor: POS 3";
		if (ui.button(uiTumorPosRect, tumorModeLabel)) {
			applyTumorMode(tumorModeIndex + 1, true);
			const bool anisNow = (std::abs(youngs1 - youngs2) > 1e-1f || std::abs(youngs1 - youngs3) > 1e-1f);
			#pragma omp parallel for
			for (int i = 0; i < object.groupNum; ++i) {
				const float scale = effectiveYoungsScaleForGroup(i);
				if (anisNow) {
					object.groups[i].calGroupKAni(youngs1 * scale, youngs2 * scale, youngs3 * scale, poisson);
				} else {
					const float E = effectiveYoungsForGroup(i, youngs);
					object.groups[i].calGroupK(E, poisson);
				}
				object.groups[i].calLHS();
			}
		}

		const SimpleUI::Rect uiStressRect{ rightMargin, uiMargin + 2.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiStressRect, showStressCloud ? "Show Groups" : "Show Stress")) {
			showStressCloud = !showStressCloud;
			if (showStressCloud) {
				showFiberFlow = false;
				showGhostLinks = false;
				showExplodedView = false;
			}
		}

		const SimpleUI::Rect uiFiberRect{ rightMargin, uiMargin + 3.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiFiberRect, showFiberFlow ? "Hide Fiber" : "Show Fiber")) {
			showFiberFlow = !showFiberFlow;
			if (showFiberFlow) showStressCloud = false;
		}

		const SimpleUI::Rect uiGhostRect{ rightMargin, uiMargin + 4.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiGhostRect, showGhostLinks ? "Hide Coupling" : "Show Coupling")) {
			showGhostLinks = !showGhostLinks;
			if (showGhostLinks) showStressCloud = false;
		}

		const SimpleUI::Rect uiExplodedRect{ rightMargin, uiMargin + 5.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiExplodedRect, showExplodedView ? "Show Integrated" : "Exploded View")) {
			showExplodedView = !showExplodedView;
			if (showExplodedView) {
				showStressCloud = false;
				showVolumePreservation = false;
			}
		}

		const SimpleUI::Rect uiVolumePreservationRect{ rightMargin, uiMargin + 6.0f * (uiH + 8.0f), uiW, uiH };
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
		if (ui.button(uiPauseRect, isPaused ? "Resume" : "Pause")) {
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
				const float scale = effectiveYoungsScaleForGroup(i);
				object.groups[i].calGroupKAni(youngs1 * scale, youngs2 * scale, youngs3 * scale, poisson);
				object.groups[i].calLHS();
			}
			}

		const SimpleUI::Rect uiCavityWallRect{ rightMargin, uiMargin + 8.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiCavityWallRect, showCavityWallVisual ? "Hide Cavity Wall" : "Show Cavity Wall")) {
			showCavityWallVisual = !showCavityWallVisual;
		}

		const SimpleUI::Rect uiAnatomyRect{ rightMargin, uiMargin + 9.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiAnatomyRect, (showSuspensionVisual || showFixedPointVisual) ? "Hide Lig+Fixed" : "Show Lig+Fixed")) {
			const bool next = !(showSuspensionVisual || showFixedPointVisual);
			showSuspensionVisual = next;
			showFixedPointVisual = next;
		}

		const SimpleUI::Rect uiFixedPointsRect{ rightMargin, uiMargin + 10.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiFixedPointsRect, showFixedPointVisual ? "Hide Fixed Points" : "Show Fixed Points")) {
			showFixedPointVisual = !showFixedPointVisual;
		}

		const SimpleUI::Rect uiLigamentRect{ rightMargin, uiMargin + 11.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiLigamentRect, showSuspensionVisual ? "Hide Ligaments" : "Show Ligaments")) {
			showSuspensionVisual = !showSuspensionVisual;
		}

		const SimpleUI::Rect uiRenderModeRect{ rightMargin, uiMargin + 12.0f * (uiH + 8.0f), uiW, uiH };
		if (ui.button(uiRenderModeRect, showLiverSmoothRender ? "Render: Smooth" : "Render: Groups")) {
			showLiverSmoothRender = !showLiverSmoothRender;
		}

			// Agent force mini graph (bottom-left, above the status label).
			if (showAgentForceGraph && agentSphere.enabled) {
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
					const std::string mode = (agentForceGraphMode != 0) ? "DEVICE" : "CONTACT";
					const std::string label =
						"INDEX " + mode + " FORCE N  SCALE " + formatSignedInt(agentForceGraphScaleN) +
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
					Eigen::Vector3f fN = Eigen::Vector3f::Zero();
					if (agentForceGraphMode != 0) {
						fN = agentLastDeviceForcesN[static_cast<size_t>(kForceGraphFingerIndex)];
					} else {
						fN = agentFilteredContactForcesN[static_cast<size_t>(kForceGraphFingerIndex)];
					}
					const int contacts = agentLastContactCounts[static_cast<size_t>(kForceGraphFingerIndex)];
					const std::string label =
						"AGENT ON VC " + std::string(agentUseVC ? "1" : "0") +
						" IDX FX " + formatSignedInt(fN.x()) +
						" FY " + formatSignedInt(fN.y()) +
						" FZ " + formatSignedInt(fN.z()) +
						" CNT " + std::to_string(contacts);
					ui.drawLabel(agentLabelRect, label, sizePx);
				} else {
				ui.drawLabel(agentLabelRect, "AGENT OFF", sizePx);
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
		if (pipelineProfiler.active) {
			const double pipelineTotalMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - pipelineFrameStart).count();
			pipelineProfiler.record(
				pipelineTotalMs,
				pipelinePreSimMs,
				pipelinePhysicsMs,
				pipelineSimMs,
				pipelineRenderMs,
				pipelineHapticTxMs);
			if (pipelineProfiler.finished()) {
				pipelineProfiler.printSummary();
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}
		}
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
