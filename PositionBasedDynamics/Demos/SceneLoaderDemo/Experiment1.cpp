#include "Experiment1.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Utils/Logger.h"
#include <algorithm>
#include <limits>

using namespace PBD;

namespace Exp1
{
	void (*resetFunc)() = nullptr;
	DemoBase *base = nullptr;

	namespace
	{
		static bool s_running = false;
		static float s_pullAccel = 800.0f; // m/s^2 along +X, Medium: 800, High: 2000
		static std::vector<unsigned int> s_fixedIndices; // Indices of fixed hilum region vertices
		static std::vector<unsigned int> s_edgeIndices; // Indices of edge vertices to apply force
		static Vector3r s_savedGravity; // Saved gravity value to restore later
	}

	void init()
	{
		// If experiment was running, restore gravity before resetting
		if (s_running)
		{
			Simulation *sim = Simulation::getCurrent();
			if (sim)
			{
				sim->setVecValue<Real>(Simulation::GRAVITATION, s_savedGravity.data());
			}
		}
		
		s_running = false;
		s_fixedIndices.clear();
		s_edgeIndices.clear();
	}

	// Identify and fix hilum region (back region, smallest Z coordinates)
	// Apply force to edge vertices (largest X coordinates)
	static void setupExperiment1()
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model)
			return;

		ParticleData &pd = model->getParticles();
		SimulationModel::TetModelVector &tetModels = model->getTetModels();

		if (tetModels.empty())
			return;

		// Use the first tet model
		TetModel *tm = tetModels[0];
		unsigned int offset = tm->getIndexOffset();
		unsigned int numVertices = tm->getParticleMesh().numVertices();

		// Find bounds
		float minX = std::numeric_limits<float>::max();
		float maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max();
		float maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max();
		float maxZ = std::numeric_limits<float>::lowest();

		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size())
				continue;

			const Vector3r &pos = pd.getPosition(idx);
			minX = std::min(minX, (float)pos[0]);
			maxX = std::max(maxX, (float)pos[0]);
			minY = std::min(minY, (float)pos[1]);
			maxY = std::max(maxY, (float)pos[1]);
			minZ = std::min(minZ, (float)pos[2]);
			maxZ = std::max(maxZ, (float)pos[2]);
		}

		const float depth = maxZ - minZ;
		const float backSliceZ = minZ + depth * 0.12f; // 最靠背面 12% 的区域作为肝门区域
		const float edgeSliceX = maxX - depth * 0.15f; // 最靠边缘 15% 的区域作为边缘区域

		s_fixedIndices.clear();
		s_edgeIndices.clear();

		// Fix hilum region (back region with smallest Z)
		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size())
				continue;

			const Vector3r &pos = pd.getPosition(idx);
			if (pos[2] <= backSliceZ)
			{
				s_fixedIndices.push_back(idx);
				pd.setMass(idx, 0.0); // Fix by setting mass to 0
			}
		}

		// Identify edge vertices (front region with largest X) for applying force
		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size())
				continue;

			const Vector3r &pos = pd.getPosition(idx);
			// Check if it's an edge vertex (large X coordinate) and not fixed
			if (pos[0] >= edgeSliceX && pd.getMass(idx) > 0.0)
			{
				s_edgeIndices.push_back(idx);
			}
		}

		LOG_INFO << "Experiment1: Fixed " << s_fixedIndices.size() << " hilum vertices, "
			<< s_edgeIndices.size() << " edge vertices for force application.";
	}

	void startExperiment1()
	{
		// Save current gravity and disable it
		Simulation *sim = Simulation::getCurrent();
		if (sim)
		{
			s_savedGravity = Vector3r(sim->getVecValue<Real>(Simulation::GRAVITATION));
			Vector3r zeroGravity(0.0, 0.0, 0.0);
			sim->setVecValue<Real>(Simulation::GRAVITATION, zeroGravity.data());
		}

		if (resetFunc)
			resetFunc(); // Reset simulation
		
		// Setup after reset
		setupExperiment1();

		if (base)
			base->setValue(DemoBase::PAUSE, false); // Unpause (run)
		
		s_running = true;
	}

	void stopExperiment1()
	{
		s_running = false;
		
		// Restore gravity
		Simulation *sim = Simulation::getCurrent();
		if (sim)
		{
			sim->setVecValue<Real>(Simulation::GRAVITATION, s_savedGravity.data());
		}
		
		// Restore masses (unfix vertices)
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (model)
		{
			ParticleData &pd = model->getParticles();
			for (unsigned int idx : s_fixedIndices)
			{
				if (idx < pd.size())
					pd.setMass(idx, 1.0); // Restore mass
			}
		}
		s_fixedIndices.clear();
		s_edgeIndices.clear();
	}

	bool isRunning()
	{
		return s_running;
	}

	void setPullAccel(float a)
	{
		s_pullAccel = a;
	}

	float getPullAccel()
	{
		return s_pullAccel;
	}

	std::function<void(ParticleData&)> externalAccelFunc()
	{
		return [](ParticleData &pd) {
			if (!s_running)
				return;

			// Apply constant acceleration along +X for edge vertices only.
			// Gravity is disabled during experiment 1, so no fallback needed.
			const Real a = static_cast<Real>(s_pullAccel);
			if (a <= static_cast<Real>(0.0))
				return;

			// Apply force only to edge vertices
			for (unsigned int idx : s_edgeIndices)
			{
				if (idx >= pd.size())
					continue;
				if (pd.getMass(idx) == 0.0)
					continue;
				Vector3r &acc = pd.getAcceleration(idx);
				acc[0] += a;
			}
		};
	}

	std::string status()
	{
		if (!s_running)
			return "Idle";
		return "Running Experiment1 (Pull: " + std::to_string(s_pullAccel) + " m/s²)";
	}
}
