#pragma once
#include <functional>
#include <string>
#include <vector>
#include "Demos/Common/DemoBase.h"
#include "Simulation/ParticleData.h"

namespace Exp1
{
	// Assigned by SceneLoaderDemo.
	extern void (*resetFunc)();
	extern PBD::DemoBase *base;

	// Reset internal state (called on scene reload/reset).
	void init();

	// Button action: reset, unpause, fix hilum region, then enable constant +X accel on edge vertices.
	void startExperiment1();
	void stopExperiment1();
	bool isRunning();

	// Provide lambda for TimeStepController external acceleration hook.
	std::function<void(PBD::ParticleData&)> externalAccelFunc();

	// Pull magnitude control (+X direction), in m/s^2. Medium: 800, High: 2000.
	void setPullAccel(float a);
	float getPullAccel();

	std::string status();
}
