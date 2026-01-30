#pragma once
#ifndef PARAMS_H
#define PARAMS_H

#include <string>

// Declare global variables
extern float youngs, youngs1, youngs2, youngs3, poisson, density;
extern int groupNum, groupNumX, groupNumY, groupNumZ;
extern const float PI;
extern float timeStep, dampingConst, Gravity, bindForce, bindVelocity, constraintHardness;
extern float dragInfluenceRadius, dragStiffness, dragMaxAccel, dragMaxDisplacement;

// Object anchoring (prevents whole-body drift; disable for full-body motion).
// 0 = none (free), 1 = fixed (hard pin), 2 = spring (soft attachment).
extern int anchorMode;
// Anchor region selection (based on initial bbox Z-depth).
extern float anchorBackSliceFrac;      // [0..1] thickness from bboxMin.z
extern float anchorRadiusDepthFrac;    // radius = depth * frac
extern float anchorCenterPushFrac;     // push anchor center towards +Z (into the body)
// Spring anchor parameters (only used when anchorMode==2). Units are "acceleration-space".
extern float anchorSpringK;            // 1/s^2
extern float anchorSpringDamping;      // 1/s
extern float anchorSpringMaxAccel;     // clamp (<=0 disables)

// Optional soft-body stabilization (helps avoid "hollow" compression / inverted tets).
extern bool tetVolumeConstraintEnabled;
// [0..1] position correction per iteration (smaller = safer/softer).
extern float tetVolumeConstraintCorrection;
// Iterations per frame (1-4 recommended).
extern int tetVolumeConstraintIterations;

// Agent sphere (finger proxy) contact parameters.
extern bool agentEnabled;
extern bool agentUseSurfaceVertices;
extern bool agentUseSurfaceTriangles;
extern bool agentVirtualCoupling;
extern float agentRadiusBboxScale;
extern float agentContactStiffness;
extern float agentContactDamping;
extern float agentMoveSpeedBboxPerSec;
extern float agentProxyMassFracOfObject;
extern float agentVcStiffnessNPerBbox;
extern float agentVcDampingNsPerBbox;
// Optional: higher VC damping while in contact (improves stability without slowing free-space motion).
extern float agentVcDampingNsPerBboxInContact;
// Optional: compute VC damping from critical damping (c = 2*zeta*sqrt(k*m)).
// If enabled, agent_vcDampingNsPerBbox* values are ignored.
extern bool agentVcAutoDamping;
extern float agentVcDampingRatioFree;
extern float agentVcDampingRatioContact;
extern float agentVcMaxDistanceRadiusFrac;
extern int agentVcSubsteps;
extern int agentCollisionIterations;
// Max number of simultaneous contact triangles to solve per proxy (1 = single triangle, >1 = manifold).
extern int agentContactManifoldTriangles;
extern float agentMaxPenetrationFrac;
extern float agentProxyPositionCorrection;
extern float agentInfluenceRadiusFrac;
extern float agentCollisionTangentialDamp;
// Scales proxy inverse mass used in contact response (0=push only tissue, 1=physical proxy mass).
// This does NOT change the VC integration mass; it only biases collision constraint corrections.
extern float agentContactProxyInvMassScale;
// Velocity feedback factor for contact position corrections (0 disables velocity injection, 1 full dp/dt).
// Lower values reduce high-frequency "buzzing" during strong press.
extern float agentContactVelocityRelaxation;
// Minimum relaxation used for large corrections (safety).
extern float agentContactVelocityRelaxationMin;
// Normal relative velocity damping while pressing (0..1). Helps eliminate contact chatter without
// requiring huge VC damping. Applied only when the proxy is driving into the surface.
extern float agentContactNormalDamp;
// Coulomb friction coefficient between proxy sphere and surface (0 disables).
extern float agentFrictionMu;
// Optional grip/stick mode (tangential spring) to help "grab" and drag the surface.
extern bool agentGripEnabled;
// Tangential correction fraction per frame (0..1).
extern float agentGripTangentCorrection;
// Maximum tangential correction step per frame (as fraction of proxy radius).
extern float agentGripMaxTangentStepFrac;
// Slip distance threshold to release grip (as fraction of proxy radius).
extern float agentGripSlipDistanceFrac;
// Minimum penetration required to engage grip (as fraction of proxy radius).
extern float agentGripMinPenetrationFrac;
// Optional low-pass filter for device force output (seconds; 0 disables).
extern float agentDeviceForceFilterTauSec;
// Optional low-pass filter for proxy contact force output (seconds; 0 disables).
// NOTE: The raw "contact force" derived from position corrections is inherently noisy in PBD;
// use this to make recorded/visualized force data usable.
extern float agentContactForceFilterTauSec;
// Optional smoothing for contact normal used for force decomposition (seconds; 0 disables).
// Helps reduce high-frequency force noise from triangle/manifold normal flipping.
extern float agentContactNormalFilterTauSec;
// Output scaling for haptic force (>=0). This does NOT affect simulation/contact, only the force you output.
extern float agentDeviceForceGain;
// Optional magnitude clamp for haptic force output (N; <=0 disables).
extern float agentDeviceForceMaxN;
extern bool agentWriteLiveFile;
extern int agentLiveFileIntervalFrames;

// Axis-aligned wall constraints (3 planes).
extern bool wallEnabled;
// Per-axis bbox extent fraction (0.05 => ~5% gap).
extern float wallMarginBboxScale;
extern float wallRestitution;
extern float wallTangentialDamp;

// Ultraleap Leap Motion (LeapC) input.
extern bool leapEnabled;
extern float leapWorkspaceXmm, leapWorkspaceYmm, leapWorkspaceZmm;
// Expands the bbox range used for mapping/clamping (fraction of bbox extents).
extern float leapWorldMargin;
// Extra multiplier on world mapping scale (1.0 = default).
extern float leapGain;
// Add a constant Y offset after mapping, as a fraction of bbox Y extent (negative lowers the hand).
extern float leapYOffsetBboxFrac;
// Extra multiplier on finger spread (relative offsets between fingertips).
extern float leapFingerSpreadGain;
// Exponential smoothing time constant in seconds (0 disables).
extern float leapSmoothingTime;
extern bool leapFlipX, leapFlipY, leapFlipZ;
// Experiment 3 (one-click) configuration (used only when EXP3 is started).
extern int exp3SettleSteps, exp3DragSteps;
extern float exp3ExOverEy;
extern bool exp3OverridePoisson;
extern float exp3PoissonOverride;
extern float exp3DragDistanceBboxScale, exp3DragDistanceMin, exp3DragDistanceMax;
extern bool exp3ResetAfterFinish;

// Experiment 1 (one-click) configuration.
extern int exp1SettleSteps, exp1DragSteps, exp1HoldSteps;
extern float exp1PullAccel, exp1ForceInfluenceRadius;
extern float exp1SweepAccel1, exp1SweepAccel2, exp1SweepAccel3;
extern int exp1PbdIterationsFast, exp1PbdIterationsReference;
extern bool exp1ResetAfterFinish;

// Experiment 2 (one-click) configuration.
extern int exp2SettleSteps, exp2DragSteps, exp2HoldSteps;
extern float exp2PoissonIncompressible;
extern float exp2DragDistanceBboxScale, exp2DragDistanceMin, exp2DragDistanceMax;
extern float exp2AnchorSliceFrac, exp2PullSliceFrac;
extern int exp2MinRegionVertexCount;
extern float exp2PullStiffness, exp2PullMaxAccel;
extern int exp2PbdIterations;
extern bool exp2ResetAfterFinish;

// Experiment 4 (one-click) configuration (performance benchmarking).
extern int exp4WarmupFrames, exp4MeasureFrames, exp4PbdIterations;
extern int exp4TargetTets1, exp4TargetTets2, exp4TargetTets3, exp4TargetTets4, exp4TargetTets5;
extern float exp4MaxVolumeStart, exp4MaxVolume1, exp4MaxVolume2, exp4MaxVolume3, exp4MaxVolume4, exp4MaxVolume5;
extern int exp4TuneIters;
extern int exp4Thread1, exp4Thread2, exp4Thread3;
// If set, relative asset filenames (e.g. stlFile/nodeFile/eleFile) are resolved under this directory.
extern std::string modelDir;
extern std::string stlFile, tetgenArgs, nodeFile, eleFile;
extern bool useDirectLoading;
extern bool autoSaveMesh;

// Haptic UART Interface
extern bool haptic_uart_enabled;
extern std::string haptic_uart_port;
extern int haptic_uart_motor_id;
extern float haptic_min_force_input;
extern float haptic_max_force_input;
extern float haptic_min_pwm_output;
extern float haptic_max_pwm_output;
extern float haptic_gamma;

// Function to load parameters
void loadParams(const std::string& filename);

#endif // PARAMS_H
