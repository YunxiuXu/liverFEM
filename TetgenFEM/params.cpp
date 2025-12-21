#include "params.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <string>

// Define global variables
float youngs, youngs1, youngs2, youngs3, poisson, density;
int groupNum, groupNumX, groupNumY, groupNumZ;
const float PI = 3.1415926535f; // This can be hardcoded as it won't change
float timeStep, dampingConst, Gravity, bindForce, bindVelocity, constraintHardness;
float dragInfluenceRadius = 0.6f;
float dragStiffness = 2500.0f;
float dragMaxAccel = 50000.0f;
float dragMaxDisplacement = 1.0f;
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

namespace {
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

void loadParams(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << "\n";
        return;
    }

	    std::unordered_map<std::string, float*> floatParams = {
	        {"youngs", &youngs}, {"youngs1", &youngs1}, {"youngs2", &youngs2},
	        {"youngs3", &youngs3}, {"poisson", &poisson}, {"density", &density},
	        {"timeStep", &timeStep}, {"dampingConst", &dampingConst},
	        {"Gravity", &Gravity}, {"bindForce", &bindForce}, {"bindVelocity", &bindVelocity},
	        {"constraintHardness", &constraintHardness},
	        {"dragInfluenceRadius", &dragInfluenceRadius},
	        {"dragStiffness", &dragStiffness},
	        {"dragMaxAccel", &dragMaxAccel},
	        {"dragMaxDisplacement", &dragMaxDisplacement},
	        {"exp3_exOverEy", &exp3ExOverEy},
	        {"exp3_poissonOverride", &exp3PoissonOverride},
	        {"exp3_dragDistanceBboxScale", &exp3DragDistanceBboxScale},
	        {"exp3_dragDistanceMin", &exp3DragDistanceMin},
	        {"exp3_dragDistanceMax", &exp3DragDistanceMax}
	        ,
	        {"exp1_pullAccel", &exp1PullAccel},
	        {"exp1_forceInfluenceRadius", &exp1ForceInfluenceRadius}
	        ,
	        {"exp1_sweepAccel1", &exp1SweepAccel1},
	        {"exp1_sweepAccel2", &exp1SweepAccel2},
	        {"exp1_sweepAccel3", &exp1SweepAccel3}
	        ,
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
	        {"exp4_maxVolume5", &exp4MaxVolume5}
	    };

    std::unordered_map<std::string, int*> intParams = {
        {"groupNumX", &groupNumX}, {"groupNumY", &groupNumY}, {"groupNumZ", &groupNumZ},
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
        {"exp4_thread3", &exp4Thread3}
    };

    std::unordered_map<std::string, std::string*> stringParams = {
        {"modelDir", &modelDir},
        {"stlFile", &stlFile}, {"tetgenArgs", &tetgenArgs}, 
        {"nodeFile", &nodeFile}, {"eleFile", &eleFile}
    };
    
    std::unordered_map<std::string, bool*> boolParams = {
        {"useDirectLoading", &useDirectLoading},
        {"autoSaveMesh", &autoSaveMesh},
        {"exp3_overridePoisson", &exp3OverridePoisson},
        {"exp3_resetAfterFinish", &exp3ResetAfterFinish},
        {"exp1_resetAfterFinish", &exp1ResetAfterFinish},
        {"exp2_resetAfterFinish", &exp2ResetAfterFinish}
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
