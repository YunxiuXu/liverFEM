#include "Experiment1.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Utils/Logger.h"
#include "Utils/FileSystem.h"
#include <algorithm>
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>

using namespace PBD;

namespace Exp1
{
	void (*resetFunc)() = nullptr;
	DemoBase *base = nullptr;

	namespace
	{
		enum class State
		{
			Idle,
			Settle,
			LoadRamp,
			HoldLoad,
			Capture,
			Finished
		};

		struct Snapshot
		{
			std::string runName;
			float pullAccel;
			unsigned int pbdIterations;
			std::vector<Vector3r> positions;
			std::vector<Vector3r> initPositions;
			std::vector<unsigned int> vertexIndices;
			float targetDisplacement;
		};

		static bool s_running = false;
		static State s_state = State::Idle;
		static float s_pullAccel = 800.0f; // m/s^2 along +X, Medium: 800, High: 2000
		static std::vector<unsigned int> s_fixedIndices; // Indices of fixed hilum region vertices
		static std::vector<unsigned int> s_edgeIndices; // Indices of edge vertices to apply force
		static std::vector<unsigned int> s_physicalIndices; // All physical vertices
		static std::vector<Vector3r> s_physicalInitPositions; // Initial positions of all physical vertices
		static Vector3r s_savedGravity; // Saved gravity value to restore later
		
		// State machine parameters
		static const int s_settleSteps = 120;
		static const int s_loadRampSteps = 240;
		static const int s_holdSteps = 240;
		static int s_stepInState = 0;
		static float s_currentLoadScale = 0.0f; // 0.0 = no force, 1.0 = full force

		static unsigned int s_targetVertexIndex = 0;
		static Vector3r s_targetInitPos;
		static Snapshot s_snapshot;
	}

	static std::string nowTimestamp()
	{
		using clock = std::chrono::system_clock;
		const auto now = clock::now();
		const std::time_t t = clock::to_time_t(now);
		std::tm tm{};
#ifdef _WIN32
		localtime_s(&tm, &t);
#else
		localtime_r(&t, &tm);
#endif
		std::ostringstream oss;
		oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
		return oss.str();
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
		s_state = State::Idle;
		s_stepInState = 0;
		s_currentLoadScale = 0.0f;
		s_fixedIndices.clear();
		s_edgeIndices.clear();
		s_physicalIndices.clear();
		s_physicalInitPositions.clear();
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
		s_physicalIndices.clear();
		s_physicalInitPositions.clear();

		// Record all physical vertices and their initial positions
		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size())
				continue;
			s_physicalIndices.push_back(idx);
			s_physicalInitPositions.push_back(pd.getPosition(idx));
		}

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
		float maxTargetX = std::numeric_limits<float>::lowest();
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
				if (pos[0] > maxTargetX)
				{
					maxTargetX = pos[0];
					s_targetVertexIndex = idx;
					s_targetInitPos = pos;
				}
			}
		}

		LOG_INFO << "Experiment1: Fixed " << s_fixedIndices.size() << " hilum vertices, "
			<< s_edgeIndices.size() << " edge vertices for force application. Target index: " << s_targetVertexIndex;
	}

	static void captureSnapshot()
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model)
			return;

		ParticleData &pd = model->getParticles();
		s_snapshot.runName = "fast"; // XPBD real-time run
		s_snapshot.pullAccel = s_pullAccel;
		s_snapshot.pbdIterations = 30; // Standard XPBD iterations for this experiment
		s_snapshot.positions.clear();
		s_snapshot.initPositions.clear();
		s_snapshot.vertexIndices.clear();
		
		// Capture all physical vertex positions
		for (size_t i = 0; i < s_physicalIndices.size(); ++i)
		{
			unsigned int idx = s_physicalIndices[i];
			if (idx < pd.size())
			{
				s_snapshot.vertexIndices.push_back(idx);
				s_snapshot.initPositions.push_back(s_physicalInitPositions[i]);
				s_snapshot.positions.push_back(pd.getPosition(idx));
			}
		}
		
		// Calculate target displacement
		if (s_targetVertexIndex < pd.size())
		{
			const Vector3r &currentPos = pd.getPosition(s_targetVertexIndex);
			s_snapshot.targetDisplacement = (currentPos - s_targetInitPos).norm();
		}
	}

	static void saveData()
	{
		std::string outputDir = "out/experiment1/" + nowTimestamp();
		Utilities::FileSystem::makeDirs(outputDir);

		// Save positions CSV using the SAME format as FEM
		const std::string csvPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_positions.csv");
		std::ofstream file(csvPath);
		if (file.is_open())
		{
			file << "run_index,run_name,pull_accel,pbd_iterations,target_displacement,vertex_list_index,vertex_index,initx,inity,initz,x,y,z\n";
			for (size_t i = 0; i < s_snapshot.positions.size(); ++i)
			{
				const Vector3r &p = s_snapshot.positions[i];
				const Vector3r &p0 = s_snapshot.initPositions[i];
				unsigned int vertexIdx = s_snapshot.vertexIndices[i];
				
				file << 0 << "," // run_index
					<< s_snapshot.runName << ","
					<< s_snapshot.pullAccel << ","
					<< s_snapshot.pbdIterations << ","
					<< s_snapshot.targetDisplacement << ","
					<< i << "," // vertex_list_index
					<< vertexIdx << ","
					<< p0[0] << "," << p0[1] << "," << p0[2] << ","
					<< p[0] << "," << p[1] << "," << p[2]
					<< "\n";
			}
			file.close();
		}

		// Save metadata
		const std::string metaPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_metadata.txt");
		std::ofstream meta(metaPath);
		if (meta.is_open())
		{
			meta << "target_vertex_index=" << s_targetVertexIndex << "\n";
			meta << "target_vertex_init=(" << s_targetInitPos[0] << "," << s_targetInitPos[1] << "," << s_targetInitPos[2] << ")\n";
			meta << "pull_accel=" << s_pullAccel << "\n";
			meta << "settleSteps=" << s_settleSteps << "\n";
			meta << "loadRampSteps=" << s_loadRampSteps << "\n";
			meta << "holdSteps=" << s_holdSteps << "\n";
			meta << "pbd_iterations=" << s_snapshot.pbdIterations << "\n";
			meta.close();
		}

		LOG_INFO << "Experiment1: Saved data to " << outputDir;
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
		
		// Disable gravity again after reset (reset may restore it)
		if (sim)
		{
			Vector3r zeroGravity(0.0, 0.0, 0.0);
			sim->setVecValue<Real>(Simulation::GRAVITATION, zeroGravity.data());
		}
		
		// Setup after reset
		setupExperiment1();

		// Initialize state machine
		s_state = State::Settle;
		s_stepInState = 0;
		s_currentLoadScale = 0.0f;

		if (base)
			base->setValue(DemoBase::PAUSE, false); // Unpause (run)
		
		s_running = true;
		LOG_INFO << "Experiment1: Started - Settle phase (120 steps)";
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

	void update()
	{
		if (!s_running)
			return;

		switch (s_state)
		{
			case State::Idle:
				return;

			case State::Settle:
			{
				s_currentLoadScale = 0.0f; // No force during settle
				++s_stepInState;
				if (s_stepInState >= s_settleSteps)
				{
					s_stepInState = 0;
					s_state = State::LoadRamp;
					LOG_INFO << "Experiment1: Settle complete - Starting LoadRamp phase (240 steps)";
				}
				return;
			}

			case State::LoadRamp:
			{
				// Linear ramp from 0.0 to 1.0 over loadRampSteps
				const float denom = static_cast<float>(std::max(1, s_loadRampSteps - 1));
				const float t = std::min(1.0f, static_cast<float>(s_stepInState) / denom);
				s_currentLoadScale = t;
				++s_stepInState;
				if (s_stepInState >= s_loadRampSteps)
				{
					s_stepInState = 0;
					s_state = State::HoldLoad;
					s_currentLoadScale = 1.0f; // Full force
					LOG_INFO << "Experiment1: LoadRamp complete - Starting HoldLoad phase (240 steps)";
				}
				return;
			}

			case State::HoldLoad:
			{
				s_currentLoadScale = 1.0f; // Full force
				++s_stepInState;
				if (s_stepInState >= s_holdSteps)
				{
					s_stepInState = 0;
					s_state = State::Capture;
					LOG_INFO << "Experiment1: HoldLoad complete - Capturing data";
				}
				return;
			}

			case State::Capture:
			{
				captureSnapshot();
				saveData();
				
				// Transition to finished
				s_state = State::Finished;
				return;
			}

			case State::Finished:
			{
				// Experiment complete - stop and restore
				s_currentLoadScale = 0.0f; // Stop applying force
				LOG_INFO << "Experiment1: Capture complete - Stopping and restoring state";
				
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
				
				s_running = false;
				s_state = State::Idle;
				LOG_INFO << "Experiment1: Finished - State restored";
				return;
			}

			default:
				return;
		}
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
			if (!s_running || s_state == State::Settle || s_state == State::Finished || s_state == State::Idle)
				return;

			// Apply acceleration scaled by currentLoadScale (0.0 during settle, ramps up, 1.0 during hold)
			const Real a = static_cast<Real>(s_pullAccel * s_currentLoadScale);
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
		
		std::string stateStr;
		switch (s_state)
		{
			case State::Settle: stateStr = "Settle"; break;
			case State::LoadRamp: stateStr = "LoadRamp"; break;
			case State::HoldLoad: stateStr = "HoldLoad"; break;
			case State::Capture: stateStr = "Capture"; break;
			case State::Finished: stateStr = "Finished"; break;
			default: stateStr = "Running"; break;
		}
		
		return "Exp1: " + stateStr + " (Pull: " + std::to_string(s_pullAccel) + " m/s², Scale: " + 
			std::to_string(s_currentLoadScale) + ")";
	}
}
