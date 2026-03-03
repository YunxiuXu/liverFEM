#include "params.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <string>
#include <cmath>

// Define global variables
float youngs, youngs1, youngs2, youngs3, poisson, density;
bool halfYoungsEnabled = false;
float halfYoungsValue = 0.0f;
int halfYoungsAxis = 0;
int halfYoungsSide = 0;
bool tumorYoungsEnabled = false;
float tumorYoungsValue = 0.0f;
float tumorTopFrac = 0.12f;
float tumorRadiusFrac = 0.22f;
float tumorCenterXFrac = 0.5f;
float tumorCenterZFrac = 0.5f;
int groupNum, groupNumX, groupNumY, groupNumZ;
const float PI = 3.1415926535f; // This can be hardcoded as it won't change
float timeStep, dampingConst, Gravity, bindForce, bindVelocity, constraintHardness;
float bind_maxAccel = 0.0f;
float dragInfluenceRadius = 0.6f;
float dragStiffness = 2500.0f;
float dragMaxAccel = 50000.0f;
float dragMaxDisplacement = 1.0f;
int anchorMode = 0;
float anchorBackSliceFrac = 0.12f;
float anchorRadiusDepthFrac = 0.20f;
float anchorCenterPushFrac = 0.02f;
float anchorSpringK = 25.0f;
float anchorSpringDamping = 10.0f;
float anchorSpringMaxAccel = 5000.0f;

bool suspensionEnabled = false;
bool susp1Enabled = true;
bool susp2Enabled = true;
bool susp3Enabled = true;
float susp1_backSliceFrac = 0.12f;
float susp1_topSliceFrac = 0.15f;
float susp_topSliceFrac = 0.12f;
float susp_sideFrac = 0.25f;
float susp_patchRadiusBboxFrac = 0.12f;
float susp1_k = 160.0f;
float susp1_damping = 25.0f;
float susp1_maxAccel = 20000.0f;
float susp2_k = 80.0f;
float susp2_damping = 15.0f;
float susp2_maxAccel = 12000.0f;
float susp3_k = 80.0f;
float susp3_damping = 15.0f;
float susp3_maxAccel = 12000.0f;

bool tetVolumeConstraintEnabled = false;
float tetVolumeConstraintCorrection = 0.15f;
int tetVolumeConstraintIterations = 2;
bool agentEnabled = false;
bool agentUseSurfaceVertices = true;
bool agentUseSurfaceTriangles = true;
bool agentVirtualCoupling = false;
float agentRadiusBboxScale = 0.03f;
float agentContactStiffness = 40000.0f;
float agentContactDamping = 200.0f;
float agentMoveSpeedBboxPerSec = 0.25f;
float agentProxyMassFracOfObject = 0.01f;
float agentVcStiffnessNPerBbox = 250.0f;
float agentVcDampingNsPerBbox = 20.0f;
float agentVcDampingNsPerBboxInContact = 20.0f;
bool agentVcAutoDamping = false;
float agentVcDampingRatioFree = 1.0f;
float agentVcDampingRatioContact = 2.0f;
float agentVcMaxDistanceRadiusFrac = 4.0f;
int agentVcSubsteps = 6;
int agentCollisionIterations = 30;
int agentContactManifoldTriangles = 1;
float agentMaxPenetrationFrac = 0.0f;
float agentProxyPositionCorrection = 1.0f;
float agentInfluenceRadiusFrac = 3.0f;
float agentCollisionTangentialDamp = 0.6f;
float agentContactProxyInvMassScale = 1.0f;
float agentContactVelocityRelaxation = 0.15f;
float agentContactVelocityRelaxationMin = 0.02f;
float agentContactNormalDamp = 0.8f;
float agentFrictionMu = 1.0f;
float agentContactForceMaterialExponent = 1.0f;
bool agentGripEnabled = false;
float agentGripTangentCorrection = 0.6f;
float agentGripMaxTangentStepFrac = 0.25f;
float agentGripSlipDistanceFrac = 0.6f;
float agentGripMinPenetrationFrac = 0.03f;
float agentDeviceForceFilterTauSec = 0.0f;
float agentContactForceFilterTauSec = 0.03f;
float agentContactNormalFilterTauSec = 0.02f;
float agentDeviceForceGain = 1.0f;
float agentDeviceForceHardGain = 1.0f;
float agentDeviceForceMaxN = 0.0f;
bool agentWriteLiveFile = false;
int agentLiveFileIntervalFrames = 2;

bool wallEnabled = true;
float wallMarginBboxScale = 0.05f;
float wallRestitution = 0.0f;
float wallTangentialDamp = 0.2f;

bool leapEnabled = false;
float leapWorkspaceXmm = 260.0f;
float leapWorkspaceYmm = 260.0f;
float leapWorkspaceZmm = 260.0f;
float leapWorldMargin = 0.25f;
float leapGain = 1.0f;
float leapYOffsetBboxFrac = -0.2f;
float leapFingerSpreadGain = 2.0f;
float leapSmoothingTime = 0.03f;
bool leapFlipX = false;
bool leapFlipY = false;
bool leapFlipZ = false;

// Leap left-hand capsule collision (non-haptic, strong interaction).
bool leftHandEnabled = true;
float leftHandCapsuleRadiusBboxScale = 0.038f;
float leftHandCapsuleLengthBboxScale = 0.075f;
int leftHandCapsuleSamples = 7;
float leftHandExtraSmoothingTime = 0.06f;
float leftHandProxyMassFracOfObject = 0.04f;
float leftHandVcStiffnessNPerBbox = 6000.0f;
float leftHandVcDampingNsPerBbox = 600.0f;
float leftHandVcMaxDistanceRadiusFrac = 3.0f;
int leftHandVcSubsteps = 40;
int leftHandCollisionIterations = 80;
int leftHandContactManifoldTriangles = 6;
float leftHandMaxPenetrationFrac = 0.0f;
float leftHandProxyPositionCorrection = 1.0f;
float leftHandCollisionTangentialDamp = 0.90f;
float leftHandContactProxyInvMassScale = 0.0f;
float leftHandContactVelocityRelaxation = 0.25f;
float leftHandContactVelocityRelaxationMin = 0.10f;
float leftHandContactNormalDamp = 0.90f;
float leftHandFrictionMu = 2.0f;
int exp3SettleSteps = 120;
int exp3DragSteps = 240;
float exp3ExOverEy = 5.0f;
bool exp3OverridePoisson = true;
float exp3PoissonOverride = 0.08f;
float exp3DragDistanceBboxScale = 0.15f;
float exp3DragDistanceMin = 0.15f;
float exp3DragDistanceMax = 0.8f;
bool exp3ResetAfterFinish = true;
int exp1SettleSteps = 120;
int exp1DragSteps = 240;
int exp1HoldSteps = 240;
float exp1PullAccel = 800.0f;
float exp1ForceInfluenceRadius = 0.6f;
float exp1SweepAccel1 = 800.0f;
float exp1SweepAccel2 = 1500.0f;
float exp1SweepAccel3 = 2000.0f;
int exp1PbdIterationsFast = 10;
int exp1PbdIterationsReference = 60;
bool exp1ResetAfterFinish = true;
int exp2SettleSteps = 120;
int exp2DragSteps = 240;
int exp2HoldSteps = 240;
float exp2PoissonIncompressible = 0.49f;
float exp2DragDistanceBboxScale = 0.35f;
float exp2DragDistanceMin = 0.30f;
float exp2DragDistanceMax = 0.90f;
float exp2AnchorSliceFrac = 0.05f;
float exp2PullSliceFrac = 0.05f;
int exp2MinRegionVertexCount = 24;
float exp2PullStiffness = 3500.0f;
float exp2PullMaxAccel = 50000.0f;
int exp2PbdIterations = 60;
bool exp2ResetAfterFinish = true;
int exp4WarmupFrames = 60;
int exp4MeasureFrames = 240;
int exp4PbdIterations = 10;
int exp4TargetTets1 = 1000;
int exp4TargetTets2 = 10000;
int exp4TargetTets3 = 20000;
int exp4TargetTets4 = 40000;
int exp4TargetTets5 = 65000;
float exp4MaxVolumeStart = 0.0f;
float exp4MaxVolume1 = 0.0f;
float exp4MaxVolume2 = 0.0f;
float exp4MaxVolume3 = 0.0f;
float exp4MaxVolume4 = 0.0f;
float exp4MaxVolume5 = 0.0f;
int exp4TuneIters = 3;
int exp4Thread1 = 1;
int exp4Thread2 = 4;
int exp4Thread3 = 8;
std::string modelDir;
std::string stlFile, tetgenArgs, nodeFile, eleFile;
bool useDirectLoading;
bool autoSaveMesh = true;
float model_rotateY_deg = 0.0f;

// Haptic UART Interface
bool haptic_uart_enabled = false;
std::string haptic_uart_port = "/dev/cu.usbserial-AUOK5THN";
int haptic_uart_motor_id = 1;
int haptic_uart_thumb_motor_id = 0;
int haptic_uart_middle_motor_id = 2;
int haptic_uart_ring_motor_id = 3;
float haptic_min_force_input = 0.0f;
float haptic_max_force_input = 10.0f;
float haptic_min_pwm_output = 0.0f;
float haptic_max_pwm_output = 255.0f;
float haptic_gamma = 1.0f;
bool haptic_softclip_enabled = false;
float haptic_softclip_knee = 200.0f;


namespace {
int clampInt(int v, int lo, int hi) {
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

bool isHalfYoungsGroup(int groupIdx) {
	if (!halfYoungsEnabled) return false;

	const int nx = std::max(1, groupNumX);
	const int ny = std::max(1, groupNumY);
	const int nz = std::max(1, groupNumZ);
	const int axis = clampInt(halfYoungsAxis, 0, 2);
	const int side = clampInt(halfYoungsSide, 0, 1);

	int axisCount = nx;
	int coord = groupIdx % nx;
	if (axis == 1) {
		axisCount = ny;
		coord = (groupIdx / nx) % ny;
	} else if (axis == 2) {
		axisCount = nz;
		coord = (groupIdx / (nx * ny));
	}

	if (axisCount <= 1) return true;

	const int split = axisCount / 2;
	if (split <= 0) return true;

	if (side == 0) return coord < split;
	return coord >= split;
}

bool isTumorYoungsGroup(int groupIdx) {
	if (!tumorYoungsEnabled) return false;

	const int nx = std::max(1, groupNumX);
	const int ny = std::max(1, groupNumY);
	const int nz = std::max(1, groupNumZ);
	if (groupIdx < 0 || groupIdx >= nx * ny * nz) return false;

	const int x = groupIdx % nx;
	const int y = (groupIdx / nx) % ny;
	const int z = (groupIdx / (nx * ny));

	const float topFrac = std::clamp(tumorTopFrac, 0.0f, 1.0f);
	const int topLayers = std::clamp(static_cast<int>(std::ceil(topFrac * static_cast<float>(ny))), 1, ny);
	if (y < (ny - topLayers)) return false;

	const int cx = std::clamp(static_cast<int>(std::lround(std::clamp(tumorCenterXFrac, 0.0f, 1.0f) * static_cast<float>(nx - 1))), 0, nx - 1);
	const int cz = std::clamp(static_cast<int>(std::lround(std::clamp(tumorCenterZFrac, 0.0f, 1.0f) * static_cast<float>(nz - 1))), 0, nz - 1);

	const int m = std::max(1, std::min(nx, nz));
	const float rFrac = std::clamp(tumorRadiusFrac, 0.0f, 1.0f);
	const int rCells = std::clamp(static_cast<int>(std::lround(rFrac * static_cast<float>(m))), 1, m);
	const int dx = x - cx;
	const int dz = z - cz;
	return (dx * dx + dz * dz) <= (rCells * rCells);
}

std::string trim(std::string s) {
	auto notSpace = [](unsigned char ch) { return !std::isspace(ch); };
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
	s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
	return s;
}

bool isAbsolutePath(const std::string& p) {
	if (p.empty()) return false;
	if (p[0] == '/' || p[0] == '\\') return true;
	if (p.size() >= 3 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':' &&
		(p[2] == '\\' || p[2] == '/')) {
		return true;
	}
	return false;
}

bool hasDirSeparator(const std::string& p) {
	return p.find('/') != std::string::npos || p.find('\\') != std::string::npos;
}

void prefixModelDirIfNeeded(std::string& path, const std::string& dir) {
	if (dir.empty() || path.empty()) return;
	if (isAbsolutePath(path) || hasDirSeparator(path)) return;
	if (dir.back() == '/' || dir.back() == '\\') path = dir + path;
	else path = dir + "/" + path;
}
} // namespace

float effectiveYoungsForGroup(int groupIdx, float baseYoungs) {
	if (tumorYoungsEnabled && isTumorYoungsGroup(groupIdx)) return tumorYoungsValue;
	if (halfYoungsEnabled && isHalfYoungsGroup(groupIdx)) return halfYoungsValue;
	return baseYoungs;
}

float effectiveYoungsScaleForGroup(int groupIdx) {
	const float base = youngs;
	if (std::abs(base) < 1e-6f) return 1.0f;
	const float eff = effectiveYoungsForGroup(groupIdx, base);
	return eff / base;
}

void loadParams(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << "\n";
        return;
    }

    std::unordered_map<std::string, float*> floatParams = {
        {"youngs", &youngs}, {"youngs1", &youngs1}, {"youngs2", &youngs2},
        {"youngs3", &youngs3}, {"poisson", &poisson}, {"density", &density},
        {"half_youngs_value", &halfYoungsValue},
        {"tumor_youngs_value", &tumorYoungsValue},
        {"tumor_topFrac", &tumorTopFrac},
        {"tumor_radiusFrac", &tumorRadiusFrac},
        {"tumor_centerXFrac", &tumorCenterXFrac},
        {"tumor_centerZFrac", &tumorCenterZFrac},
        {"timeStep", &timeStep}, {"dampingConst", &dampingConst},
        {"Gravity", &Gravity}, {"bindForce", &bindForce}, {"bindVelocity", &bindVelocity},
        {"constraintHardness", &constraintHardness},
        {"bind_maxAccel", &bind_maxAccel},
        {"dragInfluenceRadius", &dragInfluenceRadius},
        {"dragStiffness", &dragStiffness},
        {"dragMaxAccel", &dragMaxAccel},
        {"dragMaxDisplacement", &dragMaxDisplacement},
        {"anchor_backSliceFrac", &anchorBackSliceFrac},
        {"anchor_radiusDepthFrac", &anchorRadiusDepthFrac},
        {"anchor_centerPushFrac", &anchorCenterPushFrac},
        {"anchor_springK", &anchorSpringK},
        {"anchor_springDamping", &anchorSpringDamping},
        {"anchor_springMaxAccel", &anchorSpringMaxAccel},
        {"susp1_backSliceFrac", &susp1_backSliceFrac},
        {"susp1_topSliceFrac", &susp1_topSliceFrac},
        {"susp_topSliceFrac", &susp_topSliceFrac},
        {"susp_sideFrac", &susp_sideFrac},
        {"susp_patchRadiusBboxFrac", &susp_patchRadiusBboxFrac},
        {"susp1_k", &susp1_k},
        {"susp1_damping", &susp1_damping},
        {"susp1_maxAccel", &susp1_maxAccel},
        {"susp2_k", &susp2_k},
        {"susp2_damping", &susp2_damping},
        {"susp2_maxAccel", &susp2_maxAccel},
        {"susp3_k", &susp3_k},
        {"susp3_damping", &susp3_damping},
        {"susp3_maxAccel", &susp3_maxAccel},
        {"tet_volumeConstraintCorrection", &tetVolumeConstraintCorrection},
        {"agent_radiusBboxScale", &agentRadiusBboxScale},
        {"agent_contactStiffness", &agentContactStiffness},
        {"agent_contactDamping", &agentContactDamping},
        {"agent_moveSpeedBboxPerSec", &agentMoveSpeedBboxPerSec},
        {"agent_proxyMassFracOfObject", &agentProxyMassFracOfObject},
        {"agent_vcStiffnessNPerBbox", &agentVcStiffnessNPerBbox},
        {"agent_vcDampingNsPerBbox", &agentVcDampingNsPerBbox},
        {"agent_vcDampingNsPerBboxInContact", &agentVcDampingNsPerBboxInContact},
        {"agent_vcDampingRatioFree", &agentVcDampingRatioFree},
        {"agent_vcDampingRatioContact", &agentVcDampingRatioContact},
        {"agent_vcMaxDistanceRadiusFrac", &agentVcMaxDistanceRadiusFrac},
        {"agent_maxPenetrationFrac", &agentMaxPenetrationFrac},
        {"agent_proxyPositionCorrection", &agentProxyPositionCorrection},
        {"agent_influenceRadiusFrac", &agentInfluenceRadiusFrac},
        {"agent_collisionTangentialDamp", &agentCollisionTangentialDamp},
        {"agent_contactProxyInvMassScale", &agentContactProxyInvMassScale},
        {"agent_contactVelocityRelaxation", &agentContactVelocityRelaxation},
        {"agent_contactVelocityRelaxationMin", &agentContactVelocityRelaxationMin},
        {"agent_contactNormalDamp", &agentContactNormalDamp},
        {"agent_frictionMu", &agentFrictionMu},
        {"agent_contactForceMaterialExponent", &agentContactForceMaterialExponent},
        {"agent_gripTangentCorrection", &agentGripTangentCorrection},
        {"agent_gripMaxTangentStepFrac", &agentGripMaxTangentStepFrac},
        {"agent_gripSlipDistanceFrac", &agentGripSlipDistanceFrac},
        {"agent_gripMinPenetrationFrac", &agentGripMinPenetrationFrac},
        {"agent_deviceForceFilterTauSec", &agentDeviceForceFilterTauSec},
        {"agent_contactForceFilterTauSec", &agentContactForceFilterTauSec},
        {"agent_contactNormalFilterTauSec", &agentContactNormalFilterTauSec},
        {"agent_deviceForceGain", &agentDeviceForceGain},
        {"agent_deviceForceHardGain", &agentDeviceForceHardGain},
        {"agent_deviceForceMaxN", &agentDeviceForceMaxN},
        {"wall_marginBboxScale", &wallMarginBboxScale},
        {"wall_restitution", &wallRestitution},
        {"wall_tangentialDamp", &wallTangentialDamp},
        {"leap_workspaceXmm", &leapWorkspaceXmm},
        {"leap_workspaceYmm", &leapWorkspaceYmm},
        {"leap_workspaceZmm", &leapWorkspaceZmm},
        {"leap_worldMargin", &leapWorldMargin},
        {"leap_gain", &leapGain},
        {"leap_yOffsetBboxFrac", &leapYOffsetBboxFrac},
        {"leap_fingerSpreadGain", &leapFingerSpreadGain},
        {"leap_smoothingTime", &leapSmoothingTime},
        {"model_rotateY_deg", &model_rotateY_deg},
        {"left_hand_capsuleRadiusBboxScale", &leftHandCapsuleRadiusBboxScale},
        {"left_hand_capsuleLengthBboxScale", &leftHandCapsuleLengthBboxScale},
        {"left_hand_extraSmoothingTime", &leftHandExtraSmoothingTime},
        {"left_hand_proxyMassFracOfObject", &leftHandProxyMassFracOfObject},
        {"left_hand_vcStiffnessNPerBbox", &leftHandVcStiffnessNPerBbox},
        {"left_hand_vcDampingNsPerBbox", &leftHandVcDampingNsPerBbox},
        {"left_hand_vcMaxDistanceRadiusFrac", &leftHandVcMaxDistanceRadiusFrac},
        {"left_hand_maxPenetrationFrac", &leftHandMaxPenetrationFrac},
        {"left_hand_proxyPositionCorrection", &leftHandProxyPositionCorrection},
        {"left_hand_collisionTangentialDamp", &leftHandCollisionTangentialDamp},
        {"left_hand_contactProxyInvMassScale", &leftHandContactProxyInvMassScale},
        {"left_hand_contactVelocityRelaxation", &leftHandContactVelocityRelaxation},
        {"left_hand_contactVelocityRelaxationMin", &leftHandContactVelocityRelaxationMin},
        {"left_hand_contactNormalDamp", &leftHandContactNormalDamp},
        {"left_hand_frictionMu", &leftHandFrictionMu},
        {"exp3_exOverEy", &exp3ExOverEy},
        {"exp3_poissonOverride", &exp3PoissonOverride},
        {"exp3_dragDistanceBboxScale", &exp3DragDistanceBboxScale},
        {"exp3_dragDistanceMin", &exp3DragDistanceMin},
        {"exp3_dragDistanceMax", &exp3DragDistanceMax},
        {"exp1_pullAccel", &exp1PullAccel},
        {"exp1_forceInfluenceRadius", &exp1ForceInfluenceRadius},
        {"exp1_sweepAccel1", &exp1SweepAccel1},
        {"exp1_sweepAccel2", &exp1SweepAccel2},
        {"exp1_sweepAccel3", &exp1SweepAccel3},
        {"exp2_poissonIncompressible", &exp2PoissonIncompressible},
        {"exp2_dragDistanceBboxScale", &exp2DragDistanceBboxScale},
        {"exp2_dragDistanceMin", &exp2DragDistanceMin},
        {"exp2_dragDistanceMax", &exp2DragDistanceMax},
        {"exp2_anchorSliceFrac", &exp2AnchorSliceFrac},
        {"exp2_pullSliceFrac", &exp2PullSliceFrac},
        {"exp2_pullStiffness", &exp2PullStiffness},
        {"exp2_pullMaxAccel", &exp2PullMaxAccel},
        {"exp4_maxVolumeStart", &exp4MaxVolumeStart},
        {"exp4_maxVolume1", &exp4MaxVolume1},
        {"exp4_maxVolume2", &exp4MaxVolume2},
        {"exp4_maxVolume3", &exp4MaxVolume3},
        {"exp4_maxVolume4", &exp4MaxVolume4},
        {"exp4_maxVolume5", &exp4MaxVolume5},
        // Haptic params
        {"haptic_min_force_input", &haptic_min_force_input},
        {"haptic_max_force_input", &haptic_max_force_input},
        {"haptic_min_pwm_output", &haptic_min_pwm_output},
        {"haptic_max_pwm_output", &haptic_max_pwm_output},
        {"haptic_gamma", &haptic_gamma},
        {"haptic_softclip_knee", &haptic_softclip_knee}
    };

    std::unordered_map<std::string, int*> intParams = {
        {"groupNumX", &groupNumX}, {"groupNumY", &groupNumY}, {"groupNumZ", &groupNumZ},
        {"half_youngs_axis", &halfYoungsAxis},
        {"half_youngs_side", &halfYoungsSide},
        {"anchor_mode", &anchorMode},
        {"tet_volumeConstraintIterations", &tetVolumeConstraintIterations},
        {"agent_liveFileIntervalFrames", &agentLiveFileIntervalFrames},
        {"agent_vcSubsteps", &agentVcSubsteps},
        {"agent_collisionIterations", &agentCollisionIterations},
        {"agent_contactManifoldTriangles", &agentContactManifoldTriangles},
        {"left_hand_capsuleSamples", &leftHandCapsuleSamples},
        {"left_hand_vcSubsteps", &leftHandVcSubsteps},
        {"left_hand_collisionIterations", &leftHandCollisionIterations},
        {"left_hand_contactManifoldTriangles", &leftHandContactManifoldTriangles},
        {"exp3_settleSteps", &exp3SettleSteps},
        {"exp3_dragSteps", &exp3DragSteps},
        {"exp1_settleSteps", &exp1SettleSteps},
        {"exp1_dragSteps", &exp1DragSteps},
        {"exp1_holdSteps", &exp1HoldSteps},
        {"exp1_pbdIterationsFast", &exp1PbdIterationsFast},
        {"exp1_pbdIterationsReference", &exp1PbdIterationsReference},
        {"exp2_settleSteps", &exp2SettleSteps},
        {"exp2_dragSteps", &exp2DragSteps},
        {"exp2_holdSteps", &exp2HoldSteps},
        {"exp2_minRegionVertexCount", &exp2MinRegionVertexCount},
        {"exp2_pbdIterations", &exp2PbdIterations},
        {"exp4_warmupFrames", &exp4WarmupFrames},
        {"exp4_measureFrames", &exp4MeasureFrames},
        {"exp4_pbdIterations", &exp4PbdIterations},
        {"exp4_targetTets1", &exp4TargetTets1},
        {"exp4_targetTets2", &exp4TargetTets2},
        {"exp4_targetTets3", &exp4TargetTets3},
        {"exp4_targetTets4", &exp4TargetTets4},
        {"exp4_targetTets5", &exp4TargetTets5},
        {"exp4_tuneIters", &exp4TuneIters},
        {"exp4_thread1", &exp4Thread1},
        {"exp4_thread2", &exp4Thread2},
        {"exp4_thread3", &exp4Thread3},
        // Haptic params
        {"haptic_uart_motor_id", &haptic_uart_motor_id},
        {"haptic_uart_thumb_motor_id", &haptic_uart_thumb_motor_id},
        {"haptic_uart_middle_motor_id", &haptic_uart_middle_motor_id},
        {"haptic_uart_ring_motor_id", &haptic_uart_ring_motor_id}
    };

    std::unordered_map<std::string, std::string*> stringParams = {
        {"modelDir", &modelDir},
        {"stlFile", &stlFile}, {"tetgenArgs", &tetgenArgs}, 
        {"nodeFile", &nodeFile}, {"eleFile", &eleFile},
        // Haptic params
        {"haptic_uart_port", &haptic_uart_port}
    };
    
    std::unordered_map<std::string, bool*> boolParams = {
        {"useDirectLoading", &useDirectLoading},
        {"autoSaveMesh", &autoSaveMesh},
        {"half_youngs_enabled", &halfYoungsEnabled},
        {"tumor_youngs_enabled", &tumorYoungsEnabled},
        {"suspension_enabled", &suspensionEnabled},
        {"susp1_enabled", &susp1Enabled},
        {"susp2_enabled", &susp2Enabled},
        {"susp3_enabled", &susp3Enabled},
        {"tet_volumeConstraintEnabled", &tetVolumeConstraintEnabled},
        {"agent_enabled", &agentEnabled},
        {"agent_useSurfaceVertices", &agentUseSurfaceVertices},
        {"agent_useSurfaceTriangles", &agentUseSurfaceTriangles},
        {"agent_virtualCoupling", &agentVirtualCoupling},
        {"agent_vcAutoDamping", &agentVcAutoDamping},
        {"agent_gripEnabled", &agentGripEnabled},
        {"agent_writeLiveFile", &agentWriteLiveFile},
        {"wall_enabled", &wallEnabled},
        {"leap_enabled", &leapEnabled},
        {"leap_flipX", &leapFlipX},
        {"leap_flipY", &leapFlipY},
        {"leap_flipZ", &leapFlipZ},
        {"left_hand_enabled", &leftHandEnabled},
        {"exp3_overridePoisson", &exp3OverridePoisson},
        {"exp3_resetAfterFinish", &exp3ResetAfterFinish},
        {"exp1_resetAfterFinish", &exp1ResetAfterFinish},
        {"exp2_resetAfterFinish", &exp2ResetAfterFinish},
        // Haptic params
        {"haptic_uart_enabled", &haptic_uart_enabled},
        {"haptic_softclip_enabled", &haptic_softclip_enabled}
    };

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        if (floatParams.find(key) != floatParams.end()) {
            *floatParams[key] = std::stof(value);
        }
        else if (intParams.find(key) != intParams.end()) {
            *intParams[key] = std::stoi(value);
        }
        else if (stringParams.find(key) != stringParams.end()) {
            *stringParams[key] = value;
        }
        else if (boolParams.find(key) != boolParams.end()) {
            *boolParams[key] = (value == "true" || value == "1" || value == "True" || value == "TRUE");
        }
    }

    file.close();

    // Convenience: if modelDir is set, treat stlFile/nodeFile/eleFile as filenames unless they already contain a path.
    prefixModelDirIfNeeded(stlFile, modelDir);
    prefixModelDirIfNeeded(nodeFile, modelDir);
    prefixModelDirIfNeeded(eleFile, modelDir);
}
