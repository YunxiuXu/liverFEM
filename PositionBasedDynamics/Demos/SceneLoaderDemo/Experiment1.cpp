#include "Experiment1.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Simulation/TimeStepController.h"
#include "Utils/Logger.h"
#include "Utils/FileSystem.h"
#include <algorithm>
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cmath>

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
			NextRun,
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
		static bool s_isSweep = false;
		static State s_state = State::Idle;
		static float s_pullAccel = 800.0f; 
		static std::vector<unsigned int> s_fixedIndices; 
		static std::vector<unsigned int> s_edgeIndices; 
		static std::vector<unsigned int> s_physicalIndices; 
		static std::vector<Vector3r> s_physicalInitPositions; 
		static Vector3r s_savedGravity; 
		
		static const int s_settleSteps = 120;
		static const int s_loadRampSteps = 240;
		static const int s_holdSteps = 240;
		static int s_stepInState = 0;
		static float s_currentLoadScale = 0.0f; 

		static unsigned int s_targetVertexIndex = 0;
		static Vector3r s_targetInitPos;
		
		// Sweep state: Only sweep accels now
		static std::vector<float> s_sweepAccels = { 800.0f, 1500.0f, 2000.0f };
		static int s_currentAccelIdx = 0;
		static std::vector<Snapshot> s_allSnapshots;
		static std::string s_currentOutputDir = "";
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
		// 如果正在扫频，保留扫频状态，只清理边界条件相关的数据
		bool wasSweeping = s_isSweep && s_running;
		
		if (s_running && !wasSweeping)
		{
			Simulation *sim = Simulation::getCurrent();
			if (sim)
			{
				sim->setVecValue<Real>(Simulation::GRAVITATION, s_savedGravity.data());
			}
		}
		
		// 如果正在扫频，不清空这些状态
		if (!wasSweeping)
		{
			s_running = false;
			s_isSweep = false;
			s_state = State::Idle;
			s_currentAccelIdx = 0;
		}
		
		// 总是清理这些（边界条件会在 setupExperiment1 中重新设置）
		s_stepInState = 0;
		s_currentLoadScale = 0.0f;
		s_fixedIndices.clear();
		s_edgeIndices.clear();
		s_physicalIndices.clear();
		s_physicalInitPositions.clear();
		// 注意：不清空 s_allSnapshots，保留之前采集的数据
	}

	static void setupExperiment1()
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model) return;

		ParticleData &pd = model->getParticles();
		SimulationModel::TetModelVector &tetModels = model->getTetModels();
		if (tetModels.empty()) return;

		TetModel *tm = tetModels[0];
		unsigned int offset = tm->getIndexOffset();
		unsigned int numVertices = tm->getParticleMesh().numVertices();

		float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
		float minY = std::numeric_limits<float>::max(), maxY = std::numeric_limits<float>::lowest();
		float minZ = std::numeric_limits<float>::max(), maxZ = std::numeric_limits<float>::lowest();

		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size()) continue;
			const Vector3r &pos = pd.getPosition(idx);
			minX = std::min(minX, (float)pos[0]); maxX = std::max(maxX, (float)pos[0]);
			minY = std::min(minY, (float)pos[1]); maxY = std::max(maxY, (float)pos[1]);
			minZ = std::min(minZ, (float)pos[2]); maxZ = std::max(maxZ, (float)pos[2]);
		}

		const float depth = maxZ - minZ;
		const float backSliceZ = minZ + depth * 0.12f;
		const float edgeSliceX = maxX - depth * 0.15f;

		s_fixedIndices.clear();
		s_edgeIndices.clear();
		s_physicalIndices.clear();
		s_physicalInitPositions.clear();

		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size()) continue;
			s_physicalIndices.push_back(idx);
			s_physicalInitPositions.push_back(pd.getPosition(idx));

			const Vector3r &pos = pd.getPosition(idx);
			if (pos[2] <= backSliceZ)
			{
				s_fixedIndices.push_back(idx);
				pd.setMass(idx, 0.0);
			}
		}

		float maxTargetX = std::numeric_limits<float>::lowest();
		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size() || pd.getMass(idx) == 0.0) continue;
			const Vector3r &pos = pd.getPosition(idx);
			if (pos[0] >= edgeSliceX)
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

		LOG_INFO << "Experiment1 Setup: Fixed=" << s_fixedIndices.size() << " Edge=" << s_edgeIndices.size() << " TargetIdx=" << s_targetVertexIndex;
	}

	static void captureSnapshot()
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		TimeStepController *tsc = dynamic_cast<TimeStepController*>(Simulation::getCurrent()->getTimeStep());
		if (!model || !tsc) return;

		ParticleData &pd = model->getParticles();
		Snapshot sn;
		sn.pbdIterations = tsc->getMaxIterations();
		sn.runName = "xpbd_run"; 
		sn.pullAccel = s_pullAccel;
		
		for (size_t i = 0; i < s_physicalIndices.size(); ++i)
		{
			unsigned int idx = s_physicalIndices[i];
			if (idx < pd.size())
			{
				sn.vertexIndices.push_back(idx);
				sn.initPositions.push_back(s_physicalInitPositions[i]);
				sn.positions.push_back(pd.getPosition(idx));
			}
		}
		
		if (s_targetVertexIndex < pd.size())
		{
			sn.targetDisplacement = (pd.getPosition(s_targetVertexIndex) - s_targetInitPos).norm();
		}
		
		s_allSnapshots.push_back(sn);
		LOG_INFO << "Captured snapshot: accel=" << sn.pullAccel << " iters=" << sn.pbdIterations;
	}

	static void saveData()
	{
		if (s_allSnapshots.empty()) return;

		std::string outputDir = s_currentOutputDir;
		if (outputDir.empty()) outputDir = "out/experiment1/" + nowTimestamp();
		Utilities::FileSystem::makeDirs(outputDir);

		const std::string csvPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_positions.csv");
		std::ofstream file(csvPath);
		if (file.is_open())
		{
			file << "run_index,run_name,pull_accel,pbd_iterations,target_displacement,vertex_list_index,vertex_index,initx,inity,initz,x,y,z\n";
			for (size_t r = 0; r < s_allSnapshots.size(); ++r)
			{
				const auto &sn = s_allSnapshots[r];
				for (size_t i = 0; i < sn.positions.size(); ++i)
				{
					const Vector3r &p = sn.positions[i];
					const Vector3r &p0 = sn.initPositions[i];
					file << r << "," << sn.runName << "," << sn.pullAccel << "," << sn.pbdIterations << "," << sn.targetDisplacement << ","
						 << i << "," << sn.vertexIndices[i] << "," << p0[0] << "," << p0[1] << "," << p0[2] << ","
						 << p[0] << "," << p[1] << "," << p[2] << "\n";
				}
			}
			file.close();
		}

		const std::string metaPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_metadata.txt");
		std::ofstream meta(metaPath);
		if (meta.is_open())
		{
			meta << "target_vertex_index=" << s_targetVertexIndex << "\n";
			meta << "target_vertex_init=(" << s_targetInitPos[0] << "," << s_targetInitPos[1] << "," << s_targetInitPos[2] << ")\n";
			meta << "pull_accels="; for(float a : s_sweepAccels) meta << a << " "; meta << "\n";
			meta << "settleSteps=" << s_settleSteps << "\n";
			meta << "loadRampSteps=" << s_loadRampSteps << "\n";
			meta << "holdSteps=" << s_holdSteps << "\n";
			meta.close();
		}

		LOG_INFO << "Experiment1: Saved data to " << outputDir;
	}

	void startExperiment1()
	{
		init(); 
		Simulation *sim = Simulation::getCurrent();
		if (sim) {
			s_savedGravity = Vector3r(sim->getVecValue<Real>(Simulation::GRAVITATION));
			Vector3r zero(0,0,0); sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data());
		}
		if (resetFunc) resetFunc();
		if (sim) { Vector3r zero(0,0,0); sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data()); }
		setupExperiment1();
		s_state = State::Settle; s_stepInState = 0; s_currentLoadScale = 0.0f;
		if (base) base->setValue(DemoBase::PAUSE, false);
		s_running = true;
	}

	void startSweep()
	{
		startExperiment1();
		s_isSweep = true;
		s_currentAccelIdx = 0;
		s_pullAccel = s_sweepAccels[0];
		s_currentOutputDir = "out/experiment1/" + nowTimestamp();
		LOG_INFO << "Started Experiment 1 Sweep (Accels only). Accel=" << s_pullAccel;
	}

	void stopExperiment1()
	{
		s_running = false;
		Simulation *sim = Simulation::getCurrent();
		if (sim) sim->setVecValue<Real>(Simulation::GRAVITATION, s_savedGravity.data());
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (model) {
			ParticleData &pd = model->getParticles();
			for (unsigned int idx : s_fixedIndices) if (idx < pd.size()) pd.setMass(idx, 1.0);
		}
		s_fixedIndices.clear(); s_edgeIndices.clear();
	}

	bool isRunning() { return s_running; }

	void update()
	{
		if (!s_running) return;

		switch (s_state)
		{
			case State::Settle:
				s_currentLoadScale = 0.0f;
				if (++s_stepInState >= s_settleSteps) { s_stepInState = 0; s_state = State::LoadRamp; }
				break;
			case State::LoadRamp:
			{
				const float denom = static_cast<float>(std::max(1, s_loadRampSteps - 1));
				s_currentLoadScale = std::min(1.0f, static_cast<float>(s_stepInState) / denom);
				if (++s_stepInState >= s_loadRampSteps) { s_stepInState = 0; s_state = State::HoldLoad; }
				break;
			}
			case State::HoldLoad:
				s_currentLoadScale = 1.0f;
				if (++s_stepInState >= s_holdSteps) { s_stepInState = 0; s_state = State::Capture; }
				break;
			case State::Capture:
				captureSnapshot();
				if (s_isSweep) {
					s_state = State::NextRun;
				} else {
					saveData();
					s_state = State::Finished;
				}
				break;
			case State::NextRun:
				s_currentAccelIdx++;
				if (s_currentAccelIdx < s_sweepAccels.size()) {
					// 保存扫频状态
					float nextAccel = s_sweepAccels[s_currentAccelIdx];
					bool wasSweeping = s_isSweep;
					int savedAccelIdx = s_currentAccelIdx;
					
					// 重置模拟
					if (resetFunc) resetFunc();
					
					// 立即恢复扫频状态
					s_isSweep = wasSweeping;
					s_running = true;  // 确保实验继续运行
					s_currentAccelIdx = savedAccelIdx;
					s_pullAccel = nextAccel;
					
					Simulation *sim = Simulation::getCurrent();
					if (sim) { Vector3r zero(0,0,0); sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data()); }
					setupExperiment1();
					s_state = State::Settle; s_stepInState = 0; s_currentLoadScale = 0.0f;
					
					if (base) base->setValue(DemoBase::PAUSE, false); // 确保继续运行
					LOG_INFO << "Sweep Next: Accel=" << s_pullAccel;
				} else {
					saveData();
					s_state = State::Finished;
				}
				break;
			case State::Finished:
				stopExperiment1();
				s_state = State::Idle;
				LOG_INFO << "Experiment 1 complete.";
				break;
			default: break;
		}
	}

	std::function<void(ParticleData&)> externalAccelFunc()
	{
		return [](ParticleData &pd) {
			if (!s_running || s_state == State::Settle || s_state == State::Finished || s_state == State::Idle) return;
			const Real a = static_cast<Real>(s_pullAccel * s_currentLoadScale);
			if (a <= static_cast<Real>(0.0)) return;
			for (unsigned int idx : s_edgeIndices) {
				if (idx >= pd.size() || pd.getMass(idx) == 0.0) continue;
				pd.getAcceleration(idx)[0] += a;
			}
		};
	}

	std::string status()
	{
		if (!s_running) return "Idle";
		std::string s;
		switch(s_state) {
			case State::Settle: s="Settle"; break;
			case State::LoadRamp: s="LoadRamp"; break;
			case State::HoldLoad: s="HoldLoad"; break;
			case State::Capture: s="Capture"; break;
			case State::NextRun: s="NextRun"; break;
			case State::Finished: s="Finished"; break;
			default: s="Running"; break;
		}
		return (s_isSweep ? "Sweep " : "Exp1 ") + s + " (A=" + std::to_string((int)s_pullAccel) + ")";
	}

	void setPullAccel(float a) { s_pullAccel = a; }
	float getPullAccel() { return s_pullAccel; }
}
