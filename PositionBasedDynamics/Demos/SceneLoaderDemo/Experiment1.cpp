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

#include "Utils/Timing.h"
#include <chrono>

using namespace PBD;
using namespace Utilities; // 确保可以访问 Timing

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
			unsigned int pbdSubsteps;    
			double avgStepTime;          // 新增：记录平均单步耗时(ms)
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
		
		// Sweep state: sweep both substeps and accels
		// XPBD 的正确对比方式：固定迭代次数为1，对比子步数（subSteps）
		static std::vector<float> s_sweepAccels = { 800.0f, 1500.0f, 2000.0f };
		static std::vector<unsigned int> s_sweepSubsteps = { 5, 50 }; // Fast (real-time) then Reference (converged)
		static int s_currentAccelIdx = 0;
		static int s_currentSubstepIdx = 0;
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

	static std::string findProjectRoot()
	{
		// Get executable path (usually build/pbd_user/bin/SceneLoaderDemo)
		std::string exePath = Utilities::FileSystem::getProgramPath();
		
		// Find "build" in the path
		size_t buildPos = exePath.find("/build/");
		if (buildPos == std::string::npos) {
			buildPos = exePath.find("\\build\\");
		}
		
		if (buildPos != std::string::npos) {
			// Extract path up to and including "build"
			std::string upToBuild = exePath.substr(0, buildPos);
			// Go up one level from "build" to get project root
			std::string root = Utilities::FileSystem::normalizePath(upToBuild);
			
			// Verify by checking if "out" directory exists
			std::string testOut = Utilities::FileSystem::normalizePath(root + "/out");
			if (Utilities::FileSystem::isDirectory(testOut)) {
				return root;
			}
		}
		
		// Fallback: try going up from executable directory
		std::string current = exePath;
		size_t lastSlash = current.find_last_of("/\\");
		if (lastSlash != std::string::npos) {
			current = current.substr(0, lastSlash);
		}
		
		// Go up 3 levels: bin -> pbd_user -> build -> project root
		std::string root = Utilities::FileSystem::normalizePath(current + "/../../..");
		return root;
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
			s_currentSubstepIdx = 0;
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
		sn.pbdIterations = tsc->getMaxIterations();  // 固定为1
		sn.pbdSubsteps = tsc->getSubSteps();         // 对比变量
		// 根据子步数设置 run_name
		sn.runName = (sn.pbdSubsteps >= 30) ? "reference" : "fast";
		sn.pullAccel = s_pullAccel;
		
		// 记录性能数据：获取最近的 SimStep 平均耗时
		sn.avgStepTime = 0.0;
		for (std::unordered_map<int, AverageTime>::iterator iter = Timing::m_averageTimes.begin(); iter != Timing::m_averageTimes.end(); iter++) {
			if (iter->second.name == "SimStep") {
				sn.avgStepTime = iter->second.totalTime / (double)iter->second.counter;
				break;
			}
		}
		
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
		if (outputDir.empty()) {
			std::string root = findProjectRoot();
			std::string timestamp = nowTimestamp();
			outputDir = Utilities::FileSystem::normalizePath(root + "/out/experiment1/" + timestamp + "_xpbd");
		}
		Utilities::FileSystem::makeDirs(outputDir);

		const std::string csvPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_positions.csv");
		std::ofstream file(csvPath);
		if (file.is_open())
		{
			file << "run_index,run_name,pull_accel,pbd_iterations,pbd_substeps,target_displacement,vertex_list_index,vertex_index,initx,inity,initz,x,y,z\n";
			for (size_t r = 0; r < s_allSnapshots.size(); ++r)
			{
				const auto &sn = s_allSnapshots[r];
				for (size_t i = 0; i < sn.positions.size(); ++i)
				{
					const Vector3r &p = sn.positions[i];
					const Vector3r &p0 = sn.initPositions[i];
					file << r << "," << sn.runName << "," << sn.pullAccel << "," << sn.pbdIterations << "," << sn.pbdSubsteps << "," << sn.targetDisplacement << ","
						 << i << "," << sn.vertexIndices[i] << "," << p0[0] << "," << p0[1] << "," << p0[2] << ","
						 << p[0] << "," << p[1] << "," << p[2] << "\n";
				}
			}
			file.close();
		}

		// Generate sweep_summary.csv if we have both fast and reference runs
		if (s_isSweep && s_allSnapshots.size() >= 2)
		{
			const std::string sweepPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_sweep_summary.csv");
			std::ofstream sweep(sweepPath);
			if (sweep.is_open())
			{
				sweep << "pull_accel,pbd_fast_substeps,pbd_reference_substeps,target_disp_fast,target_disp_reference,target_disp_ratio_fast_over_ref,target_disp_rel_error,rmse,max_error,time_fast_ms,time_reference_ms,speedup_ref_over_fast\n";
				
				// Calculate bounding box diagonal for normalization
				float diag = 0.5f;
				if (!s_allSnapshots.empty() && !s_allSnapshots[0].initPositions.empty()) {
					Vector3r bmin(1e9,1e9,1e9), bmax(-1e9,-1e9,-1e9);
					for(const auto& p : s_allSnapshots[0].initPositions) {
						for(int d=0; d<3; d++) { 
							bmin[d] = std::min(bmin[d], p[d]); 
							bmax[d] = std::max(bmax[d], p[d]); 
						}
					}
					diag = static_cast<float>((bmax - bmin).norm());
				}

				// For each acceleration, find fast and reference runs
				for (size_t a = 0; a < s_sweepAccels.size(); ++a)
				{
					float accel = s_sweepAccels[a];
					const Snapshot *fast = nullptr, *ref = nullptr;
					
					// Find fast (5 substeps) and reference (50 substeps) for this acceleration
					for(const auto& sn : s_allSnapshots) {
						if(std::abs(sn.pullAccel - accel) < 1.0f) {
							if(sn.runName == "fast" && sn.pbdSubsteps <= 10) {
								fast = &sn;
							} else if(sn.runName == "reference" && sn.pbdSubsteps >= 30) {
								ref = &sn;
							}
						}
					}
					
					if(fast && ref && fast->positions.size() == ref->positions.size() && fast->positions.size() > 0) {
						// Calculate RMSE and max error
						double sumSqErr = 0.0;
						float maxErr = 0.0f;
						for(size_t i=0; i<fast->positions.size(); ++i) {
							float dist = (fast->positions[i] - ref->positions[i]).norm();
							sumSqErr += dist * dist;
							maxErr = std::max(maxErr, dist);
						}
						float rmse = std::sqrt(sumSqErr / fast->positions.size());
						
						// Calculate displacement ratio and relative error
						float ratio = (ref->targetDisplacement > 1e-6f) ? (fast->targetDisplacement / ref->targetDisplacement) : 1.0f;
						float relErr = std::abs(ratio - 1.0f);
						
						sweep << accel << "," << fast->pbdSubsteps << "," << ref->pbdSubsteps << ","
							  << fast->targetDisplacement << "," << ref->targetDisplacement << ","
							  << ratio << "," << relErr << "," << rmse << "," << maxErr << ","
							  << fast->avgStepTime << "," << ref->avgStepTime << ","
							  << (fast->avgStepTime > 1e-6 ? ref->avgStepTime / fast->avgStepTime : 0) << "\n";
					}
				}
				sweep.close();
			}
		}

		const std::string metaPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment1_metadata.txt");
		std::ofstream meta(metaPath);
		if (meta.is_open())
		{
			meta << "target_vertex_index=" << s_targetVertexIndex << "\n";
			meta << "target_vertex_init=(" << s_targetInitPos[0] << "," << s_targetInitPos[1] << "," << s_targetInitPos[2] << ")\n";
			meta << "pull_accels="; for(float a : s_sweepAccels) meta << a << " "; meta << "\n";
			meta << "pbd_substeps="; for(unsigned int s : s_sweepSubsteps) meta << s << " "; meta << "\n";
			meta << "pbd_iterations=1 (fixed)\n";
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
		s_currentSubstepIdx = 0;
		s_pullAccel = s_sweepAccels[0];
		
		// Set initial substeps and fix iterations to 1
		TimeStepController *tsc = dynamic_cast<TimeStepController*>(Simulation::getCurrent()->getTimeStep());
		if (tsc) {
			tsc->setMaxIterations(1);  // 固定为1，这是XPBD的稳定配置
			tsc->setSubSteps(s_sweepSubsteps[0]);
		}
		
		std::string root = findProjectRoot();
		std::string timestamp = nowTimestamp();
		s_currentOutputDir = Utilities::FileSystem::normalizePath(root + "/out/experiment1/" + timestamp + "_xpbd");
		LOG_INFO << "Started Experiment 1 Sweep. Accel=" << s_pullAccel << " SubSteps=" << s_sweepSubsteps[0] << " (Iterations=1 fixed)";
		LOG_INFO << "Output directory: " << s_currentOutputDir;
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
					// 先遍历子步数，再遍历拉力
					s_currentSubstepIdx++;
					if (s_currentSubstepIdx < s_sweepSubsteps.size()) {
						// 同一个拉力，下一个子步数
						unsigned int nextSubsteps = s_sweepSubsteps[s_currentSubstepIdx];
						bool wasSweeping = s_isSweep;
						
						LOG_INFO << "Sweep: Captured accel=" << s_pullAccel << " substeps=" << s_sweepSubsteps[s_currentSubstepIdx-1] 
							<< ", next substeps=" << nextSubsteps;
						
						// 重置模拟
						if (resetFunc) {
							resetFunc();
						}
						
						// 恢复扫频状态
						s_isSweep = wasSweeping;
						s_running = true;
						
						// 设置新的子步数，迭代次数固定为1
						TimeStepController *tsc = dynamic_cast<TimeStepController*>(Simulation::getCurrent()->getTimeStep());
						if (tsc) {
							tsc->setMaxIterations(1);  // 固定为1
							tsc->setSubSteps(nextSubsteps);
						}
						
						Simulation *sim = Simulation::getCurrent();
						if (sim) { 
							Vector3r zero(0,0,0); 
							sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data()); 
						}
						setupExperiment1();
						s_state = State::Settle; 
						s_stepInState = 0; 
						s_currentLoadScale = 0.0f;
						
						if (base) base->setValue(DemoBase::PAUSE, false);
						LOG_INFO << "Sweep: Started next run with Accel=" << s_pullAccel << " SubSteps=" << nextSubsteps << " (Iterations=1)";
					} else {
						// 当前拉力的所有子步数都完成了，切换到下一个拉力
						s_currentSubstepIdx = 0; // 重置子步索引
						s_currentAccelIdx++;
						if (s_currentAccelIdx < s_sweepAccels.size()) {
							// 下一个拉力
							float nextAccel = s_sweepAccels[s_currentAccelIdx];
							unsigned int firstSubsteps = s_sweepSubsteps[0];
							bool wasSweeping = s_isSweep;
							
							LOG_INFO << "Sweep: Completed accel=" << s_pullAccel << ", next accel=" << nextAccel;
							
							// 重置模拟
							if (resetFunc) {
								resetFunc();
							}
							
							// 恢复扫频状态
							s_isSweep = wasSweeping;
							s_running = true;
							s_pullAccel = nextAccel;
							
							// 设置第一个子步数，迭代次数固定为1
							TimeStepController *tsc = dynamic_cast<TimeStepController*>(Simulation::getCurrent()->getTimeStep());
							if (tsc) {
								tsc->setMaxIterations(1);  // 固定为1
								tsc->setSubSteps(firstSubsteps);
							}
							
							Simulation *sim = Simulation::getCurrent();
							if (sim) { 
								Vector3r zero(0,0,0); 
								sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data()); 
							}
							setupExperiment1();
							s_state = State::Settle; 
							s_stepInState = 0; 
							s_currentLoadScale = 0.0f;
							
							if (base) base->setValue(DemoBase::PAUSE, false);
							LOG_INFO << "Sweep: Started next accel=" << s_pullAccel << " SubSteps=" << firstSubsteps << " (Iterations=1)";
						} else {
							// 所有拉力和子步数都完成了
							saveData();
							s_state = State::Finished;
						}
					}
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
