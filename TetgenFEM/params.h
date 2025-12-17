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
// If set, relative asset filenames (e.g. stlFile/nodeFile/eleFile) are resolved under this directory.
extern std::string modelDir;
extern std::string stlFile, tetgenArgs, nodeFile, eleFile;
extern bool useDirectLoading;

// Function to load parameters
void loadParams(const std::string& filename);

#endif // PARAMS_H
