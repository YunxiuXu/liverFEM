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

	// Button action: reset (like 'r'), unpause (like Space), then enable constant +X accel.
	void startPull();
	void stopPull();
	bool isPulling();

	// Provide lambda for TimeStepController external acceleration hook.
	std::function<void(PBD::ParticleData&)> externalAccelFunc();

	// Pull magnitude control (+X direction), in m/s^2. If <=0, falls back to |gravity|.
	void setPullAccel(float a);
	float getPullAccel();

	std::string status();
}
