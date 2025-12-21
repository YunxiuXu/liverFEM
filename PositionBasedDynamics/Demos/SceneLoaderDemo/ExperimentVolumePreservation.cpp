#include "ExperimentVolumePreservation.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Simulation/TimeStepController.h"
#include "Simulation/TetModel.h"
#include "Utils/Logger.h"
#include "Utils/FileSystem.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace PBD;

namespace ExpVolumePreservation
{
	void (*resetFunc)() = nullptr;
	DemoBase *base = nullptr;

	namespace
	{
		static bool s_active = false;
		static float s_planeConstraintY = 0.0f;
		static double s_initialVolume = 0.0;
		static double s_currentVolume = 0.0;
	}

	void init()
	{
		s_active = false;
		s_planeConstraintY = 0.0f;
		s_initialVolume = 0.0;
		s_currentVolume = 0.0;
	}

	static double computeTotalVolume()
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model) return 0.0;

		ParticleData &pd = model->getParticles();
		SimulationModel::TetModelVector &tetModels = model->getTetModels();
		if (tetModels.empty()) return 0.0;

		double totalVolume = 0.0;
		for (TetModel *tm : tetModels)
		{
			unsigned int offset = tm->getIndexOffset();
			const Utilities::IndexedTetMesh &mesh = tm->getParticleMesh();
			const unsigned int nTets = mesh.numTets();
			const unsigned int *tets = mesh.getTets().data();

			for (unsigned int i = 0; i < nTets; ++i)
			{
				unsigned int i0 = offset + tets[4 * i];
				unsigned int i1 = offset + tets[4 * i + 1];
				unsigned int i2 = offset + tets[4 * i + 2];
				unsigned int i3 = offset + tets[4 * i + 3];

				Vector3r &x0 = pd.getPosition(i0);
				Vector3r &x1 = pd.getPosition(i1);
				Vector3r &x2 = pd.getPosition(i2);
				Vector3r &x3 = pd.getPosition(i3);

				Vector3r v1 = x1 - x0;
				Vector3r v2 = x2 - x0;
				Vector3r v3 = x3 - x0;

				double vol = std::abs(v1.dot(v2.cross(v3))) / 6.0;
				totalVolume += vol;
			}
		}
		return totalVolume;
	}

	void startVolumePreservation()
	{
		if (s_active)
		{
			stopVolumePreservation();
			return;
		}

		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model)
		{
			LOG_ERR << "ExperimentVolumePreservation: No model available";
			return;
		}

		ParticleData &pd = model->getParticles();
		SimulationModel::TetModelVector &tetModels = model->getTetModels();
		if (tetModels.empty())
		{
			LOG_ERR << "ExperimentVolumePreservation: No tet models available";
			return;
		}

		// Find maximum Y coordinate (top of the liver)
		float maxY = -std::numeric_limits<float>::max();
		for (TetModel *tm : tetModels)
		{
			unsigned int offset = tm->getIndexOffset();
			const Utilities::IndexedTetMesh &mesh = tm->getParticleMesh();
			const unsigned int nVerts = mesh.numVertices();
			
			for (unsigned int i = 0; i < nVerts; ++i)
			{
				unsigned int idx = offset + i;
				Vector3r &pos = pd.getPosition(idx);
				maxY = std::max(maxY, (float)pos[1]);
			}
		}

		// Set plane constraint at 80% of max height (to create visible compression)
		s_planeConstraintY = maxY * 0.8f;

		// Calculate initial volume
		s_initialVolume = computeTotalVolume();
		s_currentVolume = s_initialVolume;

		s_active = true;

		LOG_INFO << "ExperimentVolumePreservation: Activated. Initial volume: " << s_initialVolume
		         << ", Plane constraint Y: " << s_planeConstraintY;
	}

	void stopVolumePreservation()
	{
		s_active = false;
		s_planeConstraintY = 0.0f;
		LOG_INFO << "ExperimentVolumePreservation: Deactivated";
	}

	bool isActive()
	{
		return s_active;
	}

	void update()
	{
		if (!s_active) return;

		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model) return;

		ParticleData &pd = model->getParticles();
		SimulationModel::TetModelVector &tetModels = model->getTetModels();

		// Apply plane constraint: project vertices that penetrate the plane
		for (TetModel *tm : tetModels)
		{
			unsigned int offset = tm->getIndexOffset();
			const Utilities::IndexedTetMesh &mesh = tm->getParticleMesh();
			const unsigned int nVerts = mesh.numVertices();

			for (unsigned int i = 0; i < nVerts; ++i)
			{
				unsigned int idx = offset + i;
				Vector3r &pos = pd.getPosition(idx);
				Vector3r &vel = pd.getVelocity(idx);

				// Check if vertex is fixed (don't constrain fixed vertices)
				if (pd.getInvMass(idx) == 0.0)
					continue;

				// Project vertex to plane if it penetrates
				if (pos[1] > s_planeConstraintY)
				{
					pos[1] = s_planeConstraintY;
					// Zero out Y velocity to prevent bouncing
					vel[1] = 0.0;
				}
			}
		}

		// Update current volume
		s_currentVolume = computeTotalVolume();
	}

	std::string status()
	{
		if (!s_active)
			return "Inactive";
		
		float ratio = getVolumeRatio();
		std::ostringstream oss;
		oss << "Active | Volume: " << std::fixed << std::setprecision(2) 
		    << (ratio * 100.0f) << "%";
		return oss.str();
	}

	float getVolumeRatio()
	{
		if (s_initialVolume < 1e-10)
			return 1.0f;
		return (float)(s_currentVolume / s_initialVolume);
	}

	float getPlaneY()
	{
		return s_planeConstraintY;
	}
}
