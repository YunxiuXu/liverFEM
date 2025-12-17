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
std::string modelDir;
std::string stlFile, tetgenArgs, nodeFile, eleFile;
bool useDirectLoading;

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
	        {"dragMaxDisplacement", &dragMaxDisplacement}
	    };

    std::unordered_map<std::string, int*> intParams = {
        {"groupNumX", &groupNumX}, {"groupNumY", &groupNumY}, {"groupNumZ", &groupNumZ}
    };

    std::unordered_map<std::string, std::string*> stringParams = {
        {"modelDir", &modelDir},
        {"stlFile", &stlFile}, {"tetgenArgs", &tetgenArgs}, 
        {"nodeFile", &nodeFile}, {"eleFile", &eleFile}
    };
    
    std::unordered_map<std::string, bool*> boolParams = {
        {"useDirectLoading", &useDirectLoading}
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
