#pragma once
#ifndef PARAMS_H
#define PARAMS_H

#include <string>

// Declare global variables
extern float youngs, youngs1, youngs2, youngs3, poisson, density;
// Optional: split the model into halves and override Young's modulus on one side.
extern bool halfYoungsEnabled;
extern float halfYoungsValue;
// 0 = X, 1 = Y, 2 = Z
extern int halfYoungsAxis;
// 0 = lower (min side), 1 = upper (max side)
extern int halfYoungsSide;

// Local "tumor" stiffness patch (subset of groups).
extern bool tumorYoungsEnabled;
extern float tumorYoungsValue;
// Top slice thickness as a fraction of groupNumY (0..1).
extern float tumorTopFrac;
// Radius in the XZ plane as a fraction of min(groupNumX, groupNumZ) (0..1).
extern float tumorRadiusFrac;
// Center in X/Z as a fraction of [0..1] across groups (0.5 = middle).
extern float tumorCenterXFrac;
// Center in Y as a fraction of [0..1] across groups (0.5 = middle).
extern float tumorCenterYFrac;
extern float tumorCenterZFrac;
// Optional exact tumor center group override (set at runtime from picked point).
extern bool tumorCenterGroupOverrideEnabled;
extern int tumorCenterGroupX;
extern int tumorCenterGroupY;
extern int tumorCenterGroupZ;
// If true, use a 3D spherical tumor region (centerX/Y/Z + radiusFrac). If false, use legacy "top slice"
// XZ-cylinder (tumorTopFrac + centerX/Z + radiusFrac).
extern bool tumorUse3D;
extern int groupNum, groupNumX, groupNumY, groupNumZ;
extern const float PI;
extern float timeStep, dampingConst, Gravity, bindForce, bindVelocity, constraintHardness;
// Interface binding constraint stabilization: clamp per-constraint acceleration (<=0 disables).
extern float bind_maxAccel;
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

// Optional: 3-point "suspension" ligaments (springs from selected surface patches to box walls).
// Intended for organ supports: 1 posterior-superior (hard) + 2 diaphragm hangers (soft).
extern bool suspensionEnabled;
extern bool susp1Enabled, susp2Enabled, susp3Enabled;
// Patch selection (fractions of initial bbox extents).
extern float susp1_backSliceFrac;    // [0..1] thickness from bboxMin.z (posterior)
extern float susp1_topSliceFrac;     // [0..1] thickness from bboxMax.y (superior)
extern float susp_topSliceFrac;      // [0..1] thickness from bboxMax.y (superior) for susp2/3
extern float susp_sideFrac;          // [0..0.5] how far into left/right side to seed patches
extern float susp_patchRadiusBboxFrac; // radius = bboxDiag * frac
// Per-suspension spring parameters (acceleration-space, like drag/anchor springs).
extern float susp1_k, susp1_damping, susp1_maxAccel;
extern float susp2_k, susp2_damping, susp2_maxAccel;
extern float susp3_k, susp3_damping, susp3_maxAccel;

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
// Optional: scale haptic contact force by local material stiffness ratio (E/youngs)^exp.
// This affects only the contact force you output/record, not the collision resolution itself.
extern float agentContactForceMaterialExponent;
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
// Extra output gain applied when the proxy is over the "hard" half (Young's override side).
extern float agentDeviceForceHardGain;
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

// "Abdominal cavity" wall: a static, liver-shaped boundary derived from the liver surface (rest pose),
// inflated outward by a small gap. One side is left open (to model the surgical exposure).
extern bool cavity_enabled;
// If false: cavity is visual-only (no collision constraints applied to tissue).
extern bool cavity_collision_enabled;
extern float cavity_gap_bboxScale; // outward offset distance = bboxDiag * this
extern float cavity_open_frac;     // thickness as a fraction of bbox extent along cavity_open_axis
// 0=X, 1=Y, 2=Z
extern int cavity_open_axis;
// -1 = open near bboxMin[axis], +1 = open near bboxMax[axis]
extern int cavity_open_side;

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

// Leap left-hand capsule collision (non-haptic, strong interaction).
extern bool leftHandEnabled;
extern float leftHandCapsuleRadiusBboxScale;
extern float leftHandCapsuleLengthBboxScale;
extern int leftHandCapsuleSamples;
extern float leftHandExtraSmoothingTime;
extern float leftHandProxyMassFracOfObject;
extern float leftHandVcStiffnessNPerBbox;
extern float leftHandVcDampingNsPerBbox;
extern float leftHandVcMaxDistanceRadiusFrac;
extern int leftHandVcSubsteps;
extern int leftHandCollisionIterations;
extern int leftHandContactManifoldTriangles;
extern float leftHandMaxPenetrationFrac;
extern float leftHandProxyPositionCorrection;
extern float leftHandCollisionTangentialDamp;
extern float leftHandContactProxyInvMassScale;
extern float leftHandContactVelocityRelaxation;
extern float leftHandContactVelocityRelaxationMin;
extern float leftHandContactNormalDamp;
extern float leftHandFrictionMu;
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
// Rotate the loaded TetGen mesh around the Y axis (degrees, applied about the mesh bbox center).
// Use -90 for "clockwise 90 deg" when looking down +Y.
extern float model_rotateY_deg;
// Rotate the loaded TetGen mesh around the X axis (degrees, applied about the mesh bbox center).
// Positive uses the right-hand rule (Y->Z).
extern float model_rotateX_deg;
// Rotate the loaded TetGen mesh around the Z axis (degrees, applied about the mesh bbox center).
// Use -90 to map model -X to world +Y.
extern float model_rotateZ_deg;

// Haptic UART Interface
extern bool haptic_uart_enabled;
extern std::string haptic_uart_port;
extern int haptic_uart_motor_id;
extern int haptic_uart_thumb_motor_id;
extern int haptic_uart_middle_motor_id;
extern int haptic_uart_ring_motor_id;
extern float haptic_min_force_input;
extern float haptic_max_force_input;
extern float haptic_min_pwm_output;
extern float haptic_max_pwm_output;
extern float haptic_gamma;
extern bool haptic_softclip_enabled;
extern float haptic_softclip_knee;
extern bool haptic_slew_enabled;
extern float haptic_slew_up_pwm_per_sec;
extern float haptic_slew_down_pwm_per_sec;
// Tumor-specific "impact" vibration on contact (for 1DOF devices).
extern bool haptic_tumor_vib_enabled;
extern float haptic_tumor_vib_freq_hz;
extern float haptic_tumor_vib_amp;          // in force units (same units passed to sendForce)
extern float haptic_tumor_vib_duration_sec; // seconds

// Function to load parameters
void loadParams(const std::string& filename);

// Per-group effective Young's modulus helpers.
float effectiveYoungsForGroup(int groupIdx, float baseYoungs);
float effectiveYoungsScaleForGroup(int groupIdx);

#endif // PARAMS_H
