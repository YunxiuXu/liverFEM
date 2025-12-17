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
// If set, relative asset filenames (e.g. stlFile/nodeFile/eleFile) are resolved under this directory.
extern std::string modelDir;
extern std::string stlFile, tetgenArgs, nodeFile, eleFile;
extern bool useDirectLoading;

// Function to load parameters
void loadParams(const std::string& filename);

#endif // PARAMS_H
