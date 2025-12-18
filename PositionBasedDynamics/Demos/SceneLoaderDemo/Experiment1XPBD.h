#pragma once
#include <functional>
#include <string>
#include "Demos/Common/DemoBase.h"
#include "Simulation/ParticleData.h"

namespace Exp1XPBD
{
	// Assigned by SceneLoaderDemo.
	extern void (*resetFunc)();
	extern PBD::DemoBase *base;

	// Reset internal state (called on scene reload/reset).
	void init();

	// Button action: reset (like 'r'), unpause (like Space), then enable constant +X accel of |g|.
	void startPull();
	void stopPull();
	bool isPulling();

	// Advance state each simulation step (called from timeStep()).
	void step();

	// Provide lambda for TimeStepController external acceleration hook.
	std::function<void(PBD::ParticleData&)> externalAccelFunc();

	std::string status();
}
