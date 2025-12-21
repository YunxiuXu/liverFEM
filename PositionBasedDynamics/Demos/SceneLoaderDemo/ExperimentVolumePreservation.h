#pragma once
#include <functional>
#include <string>
#include "Demos/Common/DemoBase.h"
#include "Simulation/ParticleData.h"

namespace ExpVolumePreservation
{
	// Assigned by SceneLoaderDemo.
	extern void (*resetFunc)();
	extern PBD::DemoBase *base;

	// Reset internal state (called on scene reload/reset).
	void init();

	// Button action: toggle volume preservation visualization mode
	void startVolumePreservation();
	void stopVolumePreservation();
	bool isActive();

	// Update function to be called each time step (applies plane constraint)
	void update();

	// Status string for UI
	std::string status();
	
	// Get current volume ratio for visualization
	float getVolumeRatio();
	
	// Get plane constraint Y coordinate
	float getPlaneY();
}
