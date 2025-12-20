#pragma once
#include <functional>
#include <string>
#include <vector>
#include "Demos/Common/DemoBase.h"
#include "Simulation/ParticleData.h"

namespace Exp4
{
	extern void (*resetFunc)();
	extern PBD::DemoBase *base;

	void init();
	void startExperiment4();
	void stopExperiment4();
	bool isRunning();
	void update();

	std::string status();
}
