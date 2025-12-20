#include "Experiment4.h"
#include "Simulation/Simulation.h"
#include "Simulation/SimulationModel.h"
#include "Simulation/TimeManager.h"
#include "Simulation/TimeStepController.h"
#include "Simulation/TetModel.h"
#include "Utils/Logger.h"
#include "Utils/FileSystem.h"
#include "Utils/Timing.h"
#include "Utils/SceneLoader.h"
#include "Demos/Common/DemoBase.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace PBD;
using namespace Utilities;

namespace Exp4
{
	void (*resetFunc)() = nullptr;
	DemoBase *base = nullptr;

	namespace
	{
		enum class State { Idle, Warmup, Measure, NextRun, Finished };
		
		struct RunResult {
			std::string meshName;
			int threads;
			int tetCount;
			double avgStepTime;
			double fps;
		};

		struct RunSpec {
			std::string meshName;
			std::string sceneFile;
			int threads;
		};

		static bool s_running = false;
		static State s_state = State::Idle;
		static int s_stepInState = 0;
		static const int s_warmupSteps = 50;
		static const int s_measureSteps = 100;

		static std::vector<RunSpec> s_sequence;
		static int s_currentRunIdx = 0;
		static std::vector<RunResult> s_results;
		static double s_totalMeasureTime = 0.0;
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
		std::string exePath = Utilities::FileSystem::getProgramPath();
		size_t buildPos = exePath.find("/build/");
		if (buildPos == std::string::npos) {
			buildPos = exePath.find("\\build\\");
		}
		if (buildPos != std::string::npos) {
			std::string upToBuild = exePath.substr(0, buildPos);
			return Utilities::FileSystem::normalizePath(upToBuild);
		}
		return Utilities::FileSystem::normalizePath(exePath + "/../../..");
	}

	void init()
	{
		s_running = false;
		s_state = State::Idle;
		s_stepInState = 0;
		s_currentRunIdx = 0;
		s_results.clear();
	}

	static void setNumThreads(int threads)
	{
#ifdef _OPENMP
		omp_set_num_threads(threads);
#else
		(void)threads; // OpenMP not available; fall back to single-thread
#endif
	}

	static void saveData()
	{
		if (s_results.empty()) return;

		std::string outputDir = s_currentOutputDir;
		if (outputDir.empty()) {
			std::string root = findProjectRoot();
			outputDir = Utilities::FileSystem::normalizePath(root + "/out/experiment4/" + nowTimestamp() + "_xpbd");
		}
		Utilities::FileSystem::makeDirs(outputDir);

		const std::string csvPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment4_performance.csv");
		std::ofstream file(csvPath);
		if (file.is_open())
		{
			file << "mesh_name,threads,tet_count,avg_step_time_ms,fps\n";
			for (const auto &r : s_results)
			{
				file << r.meshName << "," << r.threads << "," << r.tetCount << ","
				     << std::fixed << std::setprecision(3) << r.avgStepTime << ","
				     << std::fixed << std::setprecision(2) << r.fps << "\n";
			}
			file.close();
			LOG_INFO << "Experiment4: Results saved to " << csvPath;
		}
		else
		{
			LOG_ERR << "Experiment4: Failed to write CSV file: " << csvPath;
		}

		// Save metadata
		const std::string metaPath = Utilities::FileSystem::normalizePath(outputDir + "/experiment4_metadata.txt");
		std::ofstream meta(metaPath);
		if (meta.is_open())
		{
			meta << "experiment=Experiment4_Performance\n";
			meta << "warmup_steps=" << s_warmupSteps << "\n";
			meta << "measure_steps=" << s_measureSteps << "\n";
			meta << "total_runs=" << s_results.size() << "\n";
			meta << "timestamp=" << nowTimestamp() << "\n";
			meta << "sequence=\n";
			for (size_t i = 0; i < s_sequence.size(); ++i)
			{
				meta << "  " << i << ": " << s_sequence[i].meshName 
				     << " threads=" << s_sequence[i].threads 
				     << " scene=" << s_sequence[i].sceneFile << "\n";
			}
			meta.close();
			LOG_INFO << "Experiment4: Metadata saved to " << metaPath;
		}
	}

	void startExperiment4()
	{
		init();
		
		// Define benchmark sequence
		// Note: The scene files must exist in data/scenes/
		s_sequence = {
			{ "Liver_Low", "PositionBasedDynamics/data/scenes/Liver_Low_XPBD.json", 1 },
			{ "Liver_Low", "PositionBasedDynamics/data/scenes/Liver_Low_XPBD.json", 10 },
			{ "Liver_Mid", "PositionBasedDynamics/data/scenes/Liver_Mid_XPBD.json", 1 },
			{ "Liver_Mid", "PositionBasedDynamics/data/scenes/Liver_Mid_XPBD.json", 10 },
			{ "Liver_High", "PositionBasedDynamics/data/scenes/Liver_High_XPBD.json", 1 },
			{ "Liver_High", "PositionBasedDynamics/data/scenes/Liver_High_XPBD.json", 10 }
		};

		s_currentRunIdx = 0;
		s_running = true;
		s_state = State::NextRun;
		
		std::string root = findProjectRoot();
		s_currentOutputDir = Utilities::FileSystem::normalizePath(root + "/out/experiment4/" + nowTimestamp() + "_xpbd");

		LOG_INFO << "Experiment4: Started Performance Benchmark.";
	}

	void stopExperiment4()
	{
		s_running = false;
		s_state = State::Idle;
	}

	bool isRunning() { return s_running; }

	void update()
	{
		if (!s_running) return;

		switch (s_state)
		{
			case State::NextRun:
			{
				if (s_currentRunIdx >= s_sequence.size())
				{
					s_state = State::Finished;
					return;
				}

				const auto &spec = s_sequence[s_currentRunIdx];
				LOG_INFO << "Experiment4: Starting Run " << s_currentRunIdx << " - Mesh: " << spec.meshName << " Threads: " << spec.threads;

				// 1. Set threads
				setNumThreads(spec.threads);
#ifdef EIGEN_USE_MKL_ALL
				// If using MKL, we might want to set its threads too, but usually it follows OMP
#endif
				
				// 2. Load scene
				if (base)
				{
					std::string root = findProjectRoot();
					std::string scenePath = Utilities::FileSystem::normalizePath(root + "/" + spec.sceneFile);
					if (Utilities::FileSystem::fileExists(scenePath))
					{
						// Save running state before reset (resetFunc will call init() which clears state)
						bool wasRunning = s_running;
						int savedRunIdx = s_currentRunIdx;
						std::vector<RunResult> savedResults = s_results;
						std::string savedOutputDir = s_currentOutputDir;
						State savedState = s_state;
						
						// First, load the new scene data (before cleanup)
						Utilities::SceneLoader *loader = base->getSceneLoader();
						if (loader == nullptr)
							loader = new Utilities::SceneLoader();
						
						// Clear old model data definitions to prevent accumulation
						base->getSceneData().m_tetModelData.clear();
						base->getSceneData().m_rigidBodyData.clear();
						base->getSceneData().m_triangleModelData.clear();

						// Load scene data first (before cleanup clears it)
						loader->readScene(scenePath.c_str(), base->getSceneData());
						
						// Now cleanup and rebuild model using the new scene data
						// reset() will call readScene(false) which uses the already-loaded SceneData
						// and also call Exp4::init() which will clear our state
						if (resetFunc) resetFunc();
						
						// Restore running state after reset
						s_running = wasRunning;
						s_currentRunIdx = savedRunIdx;
						s_results = savedResults;
						s_currentOutputDir = savedOutputDir;
						s_state = savedState;

						// Fix particles using the same method as TetgenFEM: fix left side (initx < -0.619)
						SimulationModel *model = Simulation::getCurrent()->getModel();
						if (model) {
							ParticleData &pd = model->getParticles();
							for (auto *tm : model->getTetModels()) {
								unsigned int offset = tm->getIndexOffset();
								unsigned int nVert = tm->getParticleMesh().numVertices();
								
								// Use initial position (like TetgenFEM's initx, inity, initz)
								// Fix particles on the left side: initX < -0.619
								for (unsigned int i = 0; i < nVert; i++) {
									const Vector3r &initPos = pd.getPosition0(i + offset);
									if (initPos.x() < -0.619) {
										pd.setMass(i + offset, 0.0);
									}
								}
							}
						}
					}
					else
					{
						LOG_ERR << "Experiment4: Scene file not found: " << scenePath;
						s_currentRunIdx++;
						return;
					}
				}

				s_stepInState = 0;
				s_state = State::Warmup;
				if (base) base->setValue(DemoBase::PAUSE, false);
				break;
			}

			case State::Warmup:
				if (++s_stepInState >= s_warmupSteps)
				{
					s_stepInState = 0;
					s_totalMeasureTime = 0.0;
					s_state = State::Measure;
					// Clear timing for accurate measurement
					Utilities::Timing::reset();
				}
				break;

			case State::Measure:
			{
				// Simulation timing is handled in SceneLoaderDemo::timeStep() using START_TIMING("SimStep")
				// We can access it via Utilities::Timing::m_averageTimes
				if (++s_stepInState >= s_measureSteps)
				{
					// Capture results
					double avgMs = 0.0;
					for (auto iter = Timing::m_averageTimes.begin(); iter != Timing::m_averageTimes.end(); iter++) {
						if (iter->second.name == "SimStep") {
							avgMs = iter->second.totalTime / (double)iter->second.counter;
							break;
						}
					}

					// Read tet count directly from the scene file definition, not from loaded models
					// This avoids counting residual models from previous scenes
					int tets = 0;
					if (base) {
						Utilities::SceneLoader::SceneData &sceneData = base->getSceneData();
						const std::string basePath = Utilities::FileSystem::getFilePath(base->getSceneFile());
						
						for (const auto &tmd : sceneData.m_tetModelData) {
							// Get the .ele file path and normalize it
							std::string eleFilePath = tmd.m_modelFileElements;
							if (Utilities::FileSystem::isRelativePath(eleFilePath)) {
								eleFilePath = Utilities::FileSystem::normalizePath(basePath + "/" + eleFilePath);
							} else {
								eleFilePath = Utilities::FileSystem::normalizePath(eleFilePath);
							}
							
							// Read the .ele file to get the actual tet count
							std::ifstream eleFile(eleFilePath);
							if (eleFile.is_open()) {
								std::string line;
								// Skip comments and empty lines
								while (std::getline(eleFile, line)) {
									size_t first = line.find_first_not_of(" \t\r\n");
									if (first == std::string::npos) continue;
									if (line[first] == '#') continue;
									
									// First non-comment line contains: num_tets 4 0
									std::istringstream iss(line);
									int numTets;
									if (iss >> numTets) {
										tets += numTets;
										LOG_INFO << "Experiment4: Read " << numTets << " tets from " << eleFilePath;
										break;
									}
								}
								eleFile.close();
							} else {
								LOG_WARN << "Experiment4: Cannot open .ele file: " << eleFilePath;
							}
						}
					}

					RunResult res;
					res.meshName = s_sequence[s_currentRunIdx].meshName;
					res.threads = s_sequence[s_currentRunIdx].threads;
					res.tetCount = tets;
					res.avgStepTime = avgMs;
					res.fps = (avgMs > 0.0) ? (1000.0 / avgMs) : 0.0;
					s_results.push_back(res);

					LOG_INFO << "Experiment4: Run " << s_currentRunIdx << " Result - Avg Step: " << avgMs << "ms, FPS: " << res.fps;

					s_currentRunIdx++;
					s_state = State::NextRun;
				}
				break;
			}

			case State::Finished:
				saveData();
				s_running = false;
				s_state = State::Idle;
				LOG_INFO << "Experiment4: Benchmark Complete.";
				break;

			default:
				break;
		}
	}

	std::string status()
	{
		if (!s_running) return "Idle";
		std::string s;
		switch (s_state)
		{
			case State::Warmup: s = "Warmup"; break;
			case State::Measure: s = "Measure"; break;
			case State::NextRun: s = "Loading"; break;
			case State::Finished: s = "Finished"; break;
			default: s = "Running"; break;
		}
		if (s_currentRunIdx < s_sequence.size())
			return "Exp4 " + s + " (" + s_sequence[s_currentRunIdx].meshName + ", " + std::to_string(s_sequence[s_currentRunIdx].threads) + "T)";
		return "Exp4 " + s;
	}
}
