#include "Experiment2.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Simulation/TimeStepController.h"
#include "Simulation/TetModel.h"
#include "Utils/Logger.h"
#include "Utils/FileSystem.h"
#include <algorithm>
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cmath>
#include <chrono>

#include "Simulation/Constraints.h"
#include "Simulation/TimeStepController.h"

using namespace PBD;

namespace Exp2
{
	void (*resetFunc)() = nullptr;
	DemoBase *base = nullptr;

	namespace
	{
		enum class State
		{
			Idle,
			Settle,
			Drag,
			Hold,
			NextRun,
			SaveAndFinish,
			Finished
		};

		struct DataPoint
		{
			int runIndex;
			std::string runName;
			float poisson;
			std::string stage;
			float simTime;
			float imposedDisplacement;
			float actualDisplacement;
			double volume;
			double volumeRatio;
		};

		struct RunSpec
		{
			std::string name;
			float poisson;
		};

		static bool s_running = false;
		static State s_state = State::Idle;
		static std::vector<unsigned int> s_fixedIndices;  // Anchor region
		static std::vector<unsigned int> s_pullIndices;   // Pull region
		static std::vector<unsigned int> s_physicalIndices;
		static std::vector<Vector3r> s_physicalInitPositions;
		static Vector3r s_savedGravity;

		static const int s_settleSteps = 120;
		static const int s_dragSteps = 240;
		static const int s_holdSteps = 240;
		static int s_stepInState = 0;
		static float s_currentDragDistance = 0.0f;
		static float s_dragDistance = 0.5f;  // Will be computed from bbox

		static unsigned int s_targetVertexIndex = 0;
		static Vector3r s_targetInitPos;
		static Vector3r s_runStartPos;

		static std::vector<RunSpec> s_sequence = {
			{ "baseline", 0.28f },
			{ "incompressible", 0.47f }
		};
		static int s_currentRunIndex = 0;
		static std::vector<DataPoint> s_data;
		static double s_volume0 = 0.0;
		static std::string s_currentOutputDir = "";
		static float s_originalPoisson = 0.28f;
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
		std::string exePath = Utilities::FileSystem::getProgramPath();
		size_t buildPos = exePath.find("/build/");
		if (buildPos == std::string::npos) {
			buildPos = exePath.find("\\build\\");
		}
		if (buildPos != std::string::npos) {
			std::string upToBuild = exePath.substr(0, buildPos);
			std::string root = Utilities::FileSystem::normalizePath(upToBuild);
			std::string testOut = Utilities::FileSystem::normalizePath(root + "/out");
			if (Utilities::FileSystem::isDirectory(testOut)) {
				return root;
			}
		}
		std::string current = exePath;
		size_t lastSlash = current.find_last_of("/\\");
		if (lastSlash != std::string::npos) {
			current = current.substr(0, lastSlash);
		}
		std::string root = Utilities::FileSystem::normalizePath(current + "/../../..");
		return root;
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

				if (i0 >= pd.size() || i1 >= pd.size() || i2 >= pd.size() || i3 >= pd.size())
					continue;

				const Vector3r &p0 = pd.getPosition(i0);
				const Vector3r &p1 = pd.getPosition(i1);
				const Vector3r &p2 = pd.getPosition(i2);
				const Vector3r &p3 = pd.getPosition(i3);

				// Volume = (1/6) * |(p1-p0) x (p2-p0) · (p3-p0)|
				// Using signed volume and taking absolute value
				double vol = (1.0 / 6.0) * (p1 - p0).dot((p2 - p0).cross(p3 - p0));
				totalVolume += std::abs(vol);
			}
		}
		return totalVolume;
	}

	void init()
	{
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
		s_currentDragDistance = 0.0f;
		s_fixedIndices.clear();
		s_pullIndices.clear();
		s_physicalIndices.clear();
		s_physicalInitPositions.clear();
		s_currentRunIndex = 0;
		s_data.clear();
		s_volume0 = 0.0;
	}

	static void setupExperiment2()
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

		const float xRange = maxX - minX;
		const float anchorSliceX = minX + xRange * 0.05f;  // 5% from left
		const float pullSliceX = maxX - xRange * 0.12f;     // 12% from right

		s_fixedIndices.clear();
		s_pullIndices.clear();
		s_physicalIndices.clear();
		s_physicalInitPositions.clear();

		float maxTargetX = std::numeric_limits<float>::lowest();
		for (unsigned int i = 0; i < numVertices; ++i)
		{
			unsigned int idx = offset + i;
			if (idx >= pd.size()) continue;
			s_physicalIndices.push_back(idx);
			s_physicalInitPositions.push_back(pd.getPosition(idx));

			const Vector3r &pos = pd.getPosition(idx);
			if (pos[0] <= anchorSliceX)
			{
				s_fixedIndices.push_back(idx);
				pd.setMass(idx, 0.0);
			}
			else if (pos[0] >= pullSliceX)
			{
				s_pullIndices.push_back(idx);
				if (pos[0] > maxTargetX)
				{
					maxTargetX = pos[0];
					s_targetVertexIndex = idx;
					s_targetInitPos = pos;
				}
			}
		}

		// 强制统一拉伸位移为 0.5
		s_dragDistance = 0.5f;

		LOG_INFO << "Experiment2 Setup: Fixed=" << s_fixedIndices.size() << " Pull=" << s_pullIndices.size() 
		         << " TargetIdx=" << s_targetVertexIndex << " DragDistance=" << s_dragDistance;
	}

	static void applyPoissonRatio(float poisson)
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model) return;

		LOG_INFO << "Experiment2: Hard-resetting constraints with Poisson ratio " << poisson;
		
		// 1. 强制对齐杨氏模量为 1e6，保持变量控制
		const Real stiffness = 1000000.0;
		const int method = model->getValue<int>(SimulationModel::SOLID_SIMULATION_METHOD);
		const Real volumeStiffness = model->getValue<Real>(SimulationModel::SOLID_VOLUME_STIFFNESS);
		const bool normalizeStretch = model->getValue<bool>(SimulationModel::SOLID_NORMALIZE_STRETCH);
		const bool normalizeShear = model->getValue<bool>(SimulationModel::SOLID_NORMALIZE_SHEAR);

		// 2. 清除所有约束
		model->getConstraints().clear();
		model->getConstraintGroups().clear();

		// 3. 重新添加所有四面体模型的约束（使用新的泊松比）
		for (unsigned int i = 0; i < model->getTetModels().size(); i++)
		{
			model->addSolidConstraints(model->getTetModels()[i], method, stiffness, 
				static_cast<Real>(poisson), volumeStiffness, normalizeStretch, normalizeShear);
		}

		// 4. 重置子步数以保证稳定性 (参考计划书设置 Substeps=10)
		TimeStepController *tsc = dynamic_cast<TimeStepController*>(Simulation::getCurrent()->getTimeStep());
		if (tsc) {
			tsc->setSubSteps(5);
			tsc->setMaxIterations(1); // XPBD 经典设置
		}

		LOG_INFO << "Experiment2: Constraints rebuilt successfully.";
	}

	static void captureDataPoint()
	{
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (!model) return;

		ParticleData &pd = model->getParticles();
		if (s_targetVertexIndex >= pd.size()) return;

		const double vol = computeTotalVolume();
		if (s_volume0 <= 0.0)
		{
			s_volume0 = std::max(1e-12, vol);
		}
		const double ratio = vol / s_volume0;

		const float simTime = static_cast<float>(s_stepInState) * TimeManager::getCurrent()->getTimeStepSize();
		const Vector3r &curPos = pd.getPosition(s_targetVertexIndex);
		const float actualDisp = std::max(0.0f, (float)(curPos[0] - s_runStartPos[0]));

		const RunSpec &spec = s_sequence[s_currentRunIndex];
		std::string stage = (s_state == State::Drag) ? "drag" : "hold";

		s_data.push_back(DataPoint{
			s_currentRunIndex,
			spec.name,
			spec.poisson,
			stage,
			simTime,
			s_currentDragDistance,
			actualDisp,
			vol,
			ratio
		});
	}

	static void saveData()
	{
		if (s_data.empty()) return;

		std::string outputDir = s_currentOutputDir;
		if (outputDir.empty()) {
			std::string root = findProjectRoot();
			std::string timestamp = nowTimestamp();
			outputDir = Utilities::FileSystem::normalizePath(root + "/out/experiment2/" + timestamp + "_xpbd");
		}
		Utilities::FileSystem::makeDirs(outputDir);

		const std::string csvPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment2_volume.csv");
		std::ofstream file(csvPath);
		if (file.is_open())
		{
			file << "run_index,run_name,poisson,stage,sim_time,imposed_displacement,actual_displacement,volume,volume_ratio\n";
			for (const auto &dp : s_data)
			{
				file << dp.runIndex << "," << dp.runName << "," << dp.poisson << "," << dp.stage << ","
				     << dp.simTime << "," << dp.imposedDisplacement << "," << dp.actualDisplacement << ","
				     << dp.volume << "," << dp.volumeRatio << "\n";
			}
			file.close();
		}

		const std::string metaPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment2_metadata.txt");
		std::ofstream meta(metaPath);
		if (meta.is_open())
		{
			meta << "target_vertex_index=" << s_targetVertexIndex << "\n";
			meta << "target_vertex_init=(" << s_targetInitPos[0] << "," << s_targetInitPos[1] << "," << s_targetInitPos[2] << ")\n";
			meta << "poisson_original=" << s_originalPoisson << "\n";
			meta << "settleSteps=" << s_settleSteps << "\n";
			meta << "dragSteps=" << s_dragSteps << "\n";
			meta << "holdSteps=" << s_holdSteps << "\n";
			meta << "dragDistance=" << s_dragDistance << "\n";
			meta << "sequence=";
			for (size_t i = 0; i < s_sequence.size(); ++i)
			{
				meta << (i ? ";" : "") << s_sequence[i].name << ":nu" << s_sequence[i].poisson;
			}
			meta << "\n";
			meta.close();
		}

		LOG_INFO << "Experiment2: Saved " << s_data.size() << " data points to " << outputDir;
	}

	void startExperiment2()
	{
		init();
		Simulation *sim = Simulation::getCurrent();
		if (sim) {
			s_savedGravity = Vector3r(sim->getVecValue<Real>(Simulation::GRAVITATION));
			Vector3r zero(0,0,0); 
			sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data());
		}
		if (resetFunc) resetFunc();
		if (sim) { 
			Vector3r zero(0,0,0); 
			sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data()); 
		}
		
		// Get original Poisson ratio (this might need to be read from model parameters)
		s_originalPoisson = 0.28f;  // Default, should be read from actual model
		
		setupExperiment2();
		s_state = State::Settle; 
		s_stepInState = 0; 
		s_currentDragDistance = 0.0f;
		s_currentRunIndex = 0;
		s_volume0 = 0.0;
		
		std::string root = findProjectRoot();
		std::string timestamp = nowTimestamp();
		s_currentOutputDir = Utilities::FileSystem::normalizePath(root + "/out/experiment2/" + timestamp + "_xpbd");
		
		if (base) base->setValue(DemoBase::PAUSE, false);
		s_running = true;
		LOG_INFO << "Experiment2: Started. Output dir: " << s_currentOutputDir;
	}

	void stopExperiment2()
	{
		s_running = false;
		Simulation *sim = Simulation::getCurrent();
		if (sim) sim->setVecValue<Real>(Simulation::GRAVITATION, s_savedGravity.data());
		SimulationModel *model = Simulation::getCurrent()->getModel();
		if (model) {
			ParticleData &pd = model->getParticles();
			for (unsigned int idx : s_fixedIndices) 
				if (idx < pd.size()) pd.setMass(idx, 1.0);
		}
		s_fixedIndices.clear(); 
		s_pullIndices.clear();
	}

	bool isRunning() { return s_running; }

	void update()
	{
		if (!s_running) return;

		switch (s_state)
		{
			case State::Settle:
				s_currentDragDistance = 0.0f;
				if (++s_stepInState >= s_settleSteps) 
				{ 
					s_stepInState = 0; 
					// Get current position after settling
					SimulationModel *model = Simulation::getCurrent()->getModel();
					if (model && s_targetVertexIndex < model->getParticles().size())
					{
						s_runStartPos = model->getParticles().getPosition(s_targetVertexIndex);
					}
					else
					{
						// Fallback to initial position
						for (size_t i = 0; i < s_physicalIndices.size(); ++i)
						{
							if (s_physicalIndices[i] == s_targetVertexIndex)
							{
								s_runStartPos = s_physicalInitPositions[i];
								break;
							}
						}
					}
					s_volume0 = 0.0;  // Will be set on first drag sample
					s_state = State::Drag; 
				}
				break;
			case State::Drag:
			{
				const float denom = static_cast<float>(std::max(1, s_dragSteps - 1));
				s_currentDragDistance = std::min(1.0f, static_cast<float>(s_stepInState) / denom) * s_dragDistance;
				captureDataPoint();
				if (++s_stepInState >= s_dragSteps) 
				{ 
					s_stepInState = 0; 
					s_state = State::Hold; 
				}
				break;
			}
			case State::Hold:
				s_currentDragDistance = s_dragDistance;
				captureDataPoint();
				if (++s_stepInState >= s_holdSteps) 
				{ 
					s_stepInState = 0; 
					s_state = State::NextRun; 
				}
				break;
			case State::NextRun:
				s_currentRunIndex++;
				if (s_currentRunIndex < static_cast<int>(s_sequence.size()))
				{
					// Save running state before reset (resetFunc will call init() which clears state)
					bool wasRunning = s_running;
					int savedRunIndex = s_currentRunIndex;
					std::vector<DataPoint> savedData = s_data;  // Save collected data
					std::string savedOutputDir = s_currentOutputDir;  // Save output directory
					
					// Reset for next run
					if (resetFunc) resetFunc();
					
					// Restore running state after reset
					s_running = wasRunning;
					s_currentRunIndex = savedRunIndex;
					s_data = savedData;  // Restore collected data
					s_currentOutputDir = savedOutputDir;  // Restore output directory
					
					Simulation *sim = Simulation::getCurrent();
					if (sim) { 
						Vector3r zero(0,0,0); 
						sim->setVecValue<Real>(Simulation::GRAVITATION, zero.data()); 
					}
					setupExperiment2();
					applyPoissonRatio(s_sequence[s_currentRunIndex].poisson);
					s_state = State::Settle; 
					s_stepInState = 0; 
					s_currentDragDistance = 0.0f;
					s_volume0 = 0.0;
					if (base) base->setValue(DemoBase::PAUSE, false);
					LOG_INFO << "Experiment2: Starting run " << s_currentRunIndex << " (" << s_sequence[s_currentRunIndex].name << ", nu=" << s_sequence[s_currentRunIndex].poisson << ")";
				}
				else
				{
					s_state = State::SaveAndFinish;
				}
				break;
			case State::SaveAndFinish:
				saveData();
				s_state = State::Finished;
				break;
			case State::Finished:
				stopExperiment2();
				s_state = State::Idle;
				LOG_INFO << "Experiment2: Complete.";
				break;
			default: 
				break;
		}
	}

	std::function<void(ParticleData&)> externalAccelFunc()
	{
		return [](ParticleData &pd) {
			if (!s_running || s_state == State::Settle || s_state == State::Finished || s_state == State::Idle) return;
			if (s_pullIndices.empty()) return;
			
			// 降低追踪刚度，防止 XPBD 爆炸
			const float stiffness = 800.0f; 
			const Vector3r targetDir(1.0, 0.0, 0.0);
			
			for (unsigned int idx : s_pullIndices) {
				if (idx >= pd.size() || pd.getMass(idx) == 0.0) continue;
				
				// 假设拉伸区域整体相对于 Settle 后的起始点平移
				// 查找相对于 physicalIndices 的偏移
				size_t physIdx = 0;
				bool found = false;
				for (size_t i = 0; i < s_physicalIndices.size(); ++i) {
					if (s_physicalIndices[i] == idx) { physIdx = i; found = true; break; }
				}
				if (!found) continue;

				Vector3r pStart = s_physicalInitPositions[physIdx];
				Vector3r desiredPos = pStart + targetDir * s_currentDragDistance;
				Vector3r currentPos = pd.getPosition(idx);
				
				pd.getAcceleration(idx) += stiffness * (desiredPos - currentPos);
			}
		};
	}

	std::string status()
	{
		if (!s_running) return "Idle";
		std::string s;
		switch(s_state) {
			case State::Settle: s="Settle"; break;
			case State::Drag: s="Drag"; break;
			case State::Hold: s="Hold"; break;
			case State::NextRun: s="NextRun"; break;
			case State::SaveAndFinish: s="Saving"; break;
			case State::Finished: s="Finished"; break;
			default: s="Running"; break;
		}
		if (s_currentRunIndex < static_cast<int>(s_sequence.size()))
		{
			return "Exp2 " + s + " (" + s_sequence[s_currentRunIndex].name + ", nu=" + std::to_string(s_sequence[s_currentRunIndex].poisson) + ")";
		}
		return "Exp2 " + s;
	}
}

