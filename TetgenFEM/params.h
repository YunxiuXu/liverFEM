#pragma once
#ifndef PARAMS_H
#define PARAMS_H

#include <string>

// Declare global variables
extern float youngs, youngs1, youngs2, youngs3, poisson, density;
extern int groupNum, groupNumX, groupNumY, groupNumZ;
extern const float PI;
extern float timeStep, dampingConst, Gravity, bindForce, bindVelocity, constraintHardness;
extern std::string stlFile, tetgenArgs;

// Function to load parameters
void loadParams(const std::string& filename);

#endif // PARAMS_H
