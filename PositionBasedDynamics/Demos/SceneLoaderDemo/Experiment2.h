#pragma once
#include <functional>
#include <string>
#include <vector>
#include "Demos/Common/DemoBase.h"
#include "Simulation/ParticleData.h"

namespace Exp2
{
	// Assigned by SceneLoaderDemo.
	extern void (*resetFunc)();
	extern PBD::DemoBase *base;

	// Reset internal state (called on scene reload/reset).
	void init();

	// Button action: run volume preservation test with two Poisson ratios
	void startExperiment2();
	void stopExperiment2();
	bool isRunning();

	// Update function to be called each time step (handles state machine)
	void update();

	// Provide lambda for TimeStepController external acceleration hook.
	std::function<void(PBD::ParticleData&)> externalAccelFunc();

	// Status string for UI
	std::string status();
}

