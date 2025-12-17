#include "Experiment4.h"
#include "Object.h"
#include "Group.h"
#include "ReadSTL.h"
#include "params.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

Experiment4& Experiment4::instance() {
    static Experiment4 inst;
    return inst;
}

static std::string nowTimestamp() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t t = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return oss.str();
}

static std::string stripTetgenAFlag(const std::string& args) {
    // Remove any 'a<...>' substring from a TetGen args string like "pq20a0.000083".
    // This is a heuristic but works for typical patterns.
    std::string out;
    out.reserve(args.size());
    for (size_t i = 0; i < args.size();) {
        if (args[i] == 'a' || args[i] == 'A') {
            ++i;
            while (i < args.size()) {
                const char c = args[i];
                const bool isNum = (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E';
                if (!isNum) break;
                ++i;
            }
            continue;
        }
        out.push_back(args[i]);
        ++i;
    }
    return out;
}

static void destroyObjectGraph(Object& object) {
    std::unordered_set<Vertex*> vertices;
    std::unordered_set<Edge*> edges;
    std::unordered_set<Tetrahedron*> tets;

    for (Group& g : object.groups) {
        for (Tetrahedron* t : g.tetrahedra) {
            if (!t) continue;
            tets.insert(t);
            for (int i = 0; i < 4; ++i) {
                if (t->vertices[i]) vertices.insert(t->vertices[i]);
            }
            for (int e = 0; e < 6; ++e) {
                if (t->edges[e]) edges.insert(t->edges[e]);
            }
        }
    }

    for (Edge* e : edges) delete e;
    for (Tetrahedron* t : tets) delete t;
    for (Vertex* v : vertices) delete v;
    object.groups.clear();
}

static void resetSimulationToInitial(Object& object) {
    std::unordered_set<Vertex*> visited;
    for (Group& g : object.groups) {
        for (const auto& kv : g.verticesMap) {
            Vertex* v = kv.second;
            if (!v) continue;
            if (!visited.insert(v).second) continue;
            v->x = v->initx;
            v->y = v->inity;
            v->z = v->initz;
            v->velx = 0.0f;
            v->vely = 0.0f;
            v->velz = 0.0f;
        }
    }

    for (Group& g : object.groups) {
        const int n = static_cast<int>(g.verticesVector.size());
        g.groupVelocity = Eigen::VectorXf::Zero(3 * n);
        g.groupVelocityFEM = Eigen::VectorXf::Zero(3 * n);
        g.gravityApplied = false;
        g.gravity = Eigen::VectorXf::Zero(3 * n);

        g.currentPosition = Eigen::VectorXf::Zero(3 * n);
        g.currentPositionFEM = Eigen::VectorXf::Zero(3 * n);
        for (Vertex* v : g.verticesVector) {
            const int base = 3 * v->localIndex;
            g.currentPosition.segment<3>(base) = Eigen::Vector3f(v->initx, v->inity, v->initz);
            g.currentPositionFEM.segment<3>(base) = Eigen::Vector3f(v->initx, v->inity, v->initz);
        }

        g.Fbind = Eigen::VectorXf::Zero(3 * n);
        g.prevFbind = Eigen::VectorXf::Zero(3 * n);
        g.deltaX = Eigen::VectorXf::Zero(3 * n);
        g.deltaXFEM = Eigen::VectorXf::Zero(3 * n);
        g.primeVec = g.currentPosition;
        g.prevPrimeVec.resize(0);
        g.rotate_matrix.setIdentity();
        g.lastRotationUpdate = -100;
        g.constraintLambdas.clear();
    }
}

static void applyDefaultFixRegion(Object& object) {
    std::unordered_set<Vertex*> visitedVertices;
    std::vector<Vertex*> uniqueVertices;
    uniqueVertices.reserve(10000);
    for (const auto& g : object.groups) {
        for (const auto& pair : g.verticesMap) {
            Vertex* v = pair.second;
            if (visitedVertices.insert(v).second) {
                uniqueVertices.push_back(v);
            }
        }
    }
    if (uniqueVertices.empty()) return;

    Eigen::Vector3f minBound(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    Eigen::Vector3f maxBound(
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::max());
    for (const auto* v : uniqueVertices) {
        minBound.x() = std::min(minBound.x(), v->initx);
        minBound.y() = std::min(minBound.y(), v->inity);
        minBound.z() = std::min(minBound.z(), v->initz);
        maxBound.x() = std::max(maxBound.x(), v->initx);
        maxBound.y() = std::max(maxBound.y(), v->inity);
        maxBound.z() = std::max(maxBound.z(), v->initz);
    }

    const float depth = maxBound.z() - minBound.z();
    const float backSliceZ = minBound.z() + depth * 0.12f;
    Eigen::Vector3f backCentroid(0.0f, 0.0f, 0.0f);
    int backCount = 0;
    for (const auto* v : uniqueVertices) {
        if (v->initz <= backSliceZ) {
            backCentroid.x() += v->initx;
            backCentroid.y() += v->inity;
            backCentroid.z() += v->initz;
            ++backCount;
        }
    }
    if (backCount > 0) {
        backCentroid /= static_cast<float>(backCount);
    } else {
        backCentroid = Eigen::Vector3f(
            0.5f * (minBound.x() + maxBound.x()),
            0.5f * (minBound.y() + maxBound.y()),
            minBound.z());
    }
    backCentroid.z() = std::min(backCentroid.z() + depth * 0.02f, maxBound.z());
    const float anchorRadius = std::max(depth * 0.2f, 0.001f);
    object.fixRegion(backCentroid, anchorRadius);
}

struct BuiltObject {
    Object object;
    int tetCount = 0;
    int pointCount = 0;
    float maxVolume = 0.0f;
    std::string meshSource;
    std::string tetgenArgsUsed;
};

static std::pair<int, int> meshCountsFromStlWithMaxVolume(const std::string& stlPath, const std::string& baseArgs, float maxVolume, std::string* argsUsedOut = nullptr) {
    tetgenio in, out;
    in.firstnumber = 1;
    readSTL(stlPath.c_str(), in);

    tetgenbehavior behavior;
    std::ostringstream oss;
    oss << baseArgs << "a" << std::setprecision(9) << maxVolume;
    std::string args = oss.str();
    if (argsUsedOut) *argsUsedOut = args;
    char* argsC = const_cast<char*>(args.c_str());
    behavior.parse_commandline(argsC);
    behavior.quiet = 1;

    tetrahedralize(&behavior, &in, &out);
    return { out.numberoftetrahedra, out.numberofpoints };
}

static BuiltObject buildFromStlWithMaxVolume(const std::string& stlPath, const std::string& baseArgs, float maxVolume) {
    tetgenio in, out;
    in.firstnumber = 1;
    readSTL(stlPath.c_str(), in);

    tetgenbehavior behavior;
    std::ostringstream oss;
    oss << baseArgs << "a" << std::setprecision(9) << maxVolume;
    std::string args = oss.str();
    char* argsC = const_cast<char*>(args.c_str());
    behavior.parse_commandline(argsC);
    behavior.quiet = 1;

    tetrahedralize(&behavior, &in, &out);

    BuiltObject built;
    built.tetCount = out.numberoftetrahedra;
    built.pointCount = out.numberofpoints;
    built.maxVolume = maxVolume;
    built.meshSource = stlPath;
    built.tetgenArgsUsed = args;

    groupNum = groupNumX * groupNumY * groupNumZ;
    built.object.groupNum = groupNum;
    built.object.groupNumX = groupNumX;
    built.object.groupNumY = groupNumY;
    built.object.groupNumZ = groupNumZ;
    divideIntoGroups(out, built.object, groupNumX, groupNumY, groupNumZ);

    built.object.updateIndices();
    built.object.assignLocalIndicesToAllGroups();
    built.object.generateUniqueVertices();
    built.object.updateAdjacentGroupIndices(groupNumX, groupNumY, groupNumZ);
    for (int i = 0; i < groupNum; ++i) {
        built.object.storeAdjacentGroupsCommonVertices(i);
    }
    for (int i = 0; i < groupNum; ++i) {
        Group& g = built.object.getGroup(i);
        const int n = static_cast<int>(g.verticesMap.size());
        g.LHS_I = Eigen::MatrixXf::Identity(3 * n, 3 * n);
    }

    applyDefaultFixRegion(built.object);

#pragma omp parallel for
    for (int i = 0; i < built.object.groupNum; ++i) {
        built.object.groups[i].calMassMatrix(density);
        built.object.groups[i].calDampingMatrix();
        built.object.groups[i].calCenterofMass();
        built.object.groups[i].calInitCOM();
        built.object.groups[i].calLocalPos();
        built.object.groups[i].calGroupK(youngs, poisson);
        built.object.groups[i].setVertexMassesFromMassMatrix();
        built.object.groups[i].calMassGroup();
        built.object.groups[i].calMassDistributionMatrix();
        built.object.groups[i].calLHS();
    }

    return built;
}

static float tuneMaxVolumeForTargetTets(const std::string& stlPath, const std::string& baseArgs, int targetTets, float startMaxVol, int maxIters) {
    float a = std::max(1e-9f, startMaxVol);
    for (int iter = 0; iter < std::max(1, maxIters); ++iter) {
        const auto [tets, _points] = meshCountsFromStlWithMaxVolume(stlPath, baseArgs, a);
        const int n = std::max(1, tets);

        const float ratio = static_cast<float>(n) / static_cast<float>(std::max(1, targetTets));
        if (std::abs(ratio - 1.0f) < 0.12f) {
            return a;
        }
        a = std::max(1e-9f, a * ratio);
    }
    return a;
}

struct BenchRow {
    std::string meshLabel;
    std::string meshSource;
    std::string tetgenArgsUsed;
    int targetTets = 0;
    int tetCount = 0;
    int pointCount = 0;
    float maxVolume = 0.0f;
    int threads = 0;
    int pbdIterations = 0;
    int warmupFrames = 0;
    int measureFrames = 0;
    double seconds = 0.0;
    double fps = 0.0;
    double msMean = 0.0;
    double msPrimeMean = 0.0;
    double msPbdMean = 0.0;
};

static BenchRow benchObject(Object& object, const BuiltObject& built, int threads, int pbdIterations, int warmupFrames, int measureFrames) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    Eigen::setNbThreads(threads);

    resetSimulationToInitial(object);

    Eigen::Vector3f externalForce = Eigen::Vector3f::Zero();
    const std::unordered_map<int, Eigen::Vector3f>& noForces = Group::emptyVertexForce;
    int frame = 1;
    for (int i = 0; i < std::max(0, warmupFrames); ++i) {
#pragma omp parallel for
        for (int g = 0; g < object.groupNum; ++g) {
            object.groups[g].calPrimeVec(externalForce, noForces);
            object.groups[g].calRotationMatrix(frame);
        }
        object.PBDLOOP(pbdIterations);
        ++frame;
    }

    using clock = std::chrono::steady_clock;
    double primeSec = 0.0;
    double pbdSec = 0.0;
    const auto tStart = clock::now();
    for (int i = 0; i < std::max(1, measureFrames); ++i) {
        const auto t0 = clock::now();
#pragma omp parallel for
        for (int g = 0; g < object.groupNum; ++g) {
            object.groups[g].calPrimeVec(externalForce, noForces);
            object.groups[g].calRotationMatrix(frame);
        }
        const auto t1 = clock::now();
        object.PBDLOOP(pbdIterations);
        const auto t2 = clock::now();
        primeSec += std::chrono::duration<double>(t1 - t0).count();
        pbdSec += std::chrono::duration<double>(t2 - t1).count();
        ++frame;
    }
    const auto tEnd = clock::now();
    const double seconds = std::chrono::duration<double>(tEnd - tStart).count();
    const int frames = std::max(1, measureFrames);

    BenchRow row;
    row.meshLabel = "";
    row.meshSource = built.meshSource;
    row.tetgenArgsUsed = built.tetgenArgsUsed;
    row.targetTets = 0;
    row.tetCount = built.tetCount;
    row.pointCount = built.pointCount;
    row.maxVolume = built.maxVolume;
    row.threads = threads;
    row.pbdIterations = pbdIterations;
    row.warmupFrames = warmupFrames;
    row.measureFrames = measureFrames;
    row.seconds = seconds;
    row.fps = frames / std::max(1e-9, seconds);
    row.msMean = (seconds * 1000.0) / frames;
    row.msPrimeMean = (primeSec * 1000.0) / frames;
    row.msPbdMean = (pbdSec * 1000.0) / frames;
    return row;
}

void Experiment4::requestStart() {
    if (state != State::Idle) return;
    startRequested = true;
}

bool Experiment4::isActive() const {
    return state != State::Idle;
}

std::string Experiment4::buttonLabel() const {
    if (state == State::Idle) return "START EXP4 (PERF)";
    return "EXP4 RUNNING...";
}

void Experiment4::update() {
    if (state == State::Idle) {
        if (startRequested) {
            startRequested = false;
            state = State::PendingStart;
        }
        return;
    }
    if (state == State::PendingStart) {
        state = State::Running;
        runBenchmarks();
        state = State::Idle;
        return;
    }
}

void Experiment4::runBenchmarks() {
    namespace fs = std::filesystem;
    const std::string outputDir = (fs::path("..") / "out" / "experiment4" / nowTimestamp()).string();
    std::error_code ec;
    fs::create_directories(outputDir, ec);
    if (ec) {
        std::cerr << "[Experiment4] Failed to create output dir: " << outputDir << " (" << ec.message() << ")\n";
        return;
    }

    const int pbdIterations = std::max(1, exp4PbdIterations);
    const int warmupFrames = std::max(0, exp4WarmupFrames);
    const int measureFrames = std::max(1, exp4MeasureFrames);

    std::vector<int> threads = { exp4Thread1, exp4Thread2, exp4Thread3 };
    const int maxThreads = std::max(1, omp_get_num_procs());
    threads.erase(std::remove_if(threads.begin(), threads.end(),
                    [&](int t) { return t <= 0 || t > maxThreads; }),
        threads.end());
    std::sort(threads.begin(), threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());
    if (threads.empty()) threads.push_back(std::max(1, std::min(1, maxThreads)));

    struct TargetSpec { int target; float fixedMaxVol; };
    std::vector<TargetSpec> targets = {
        { exp4TargetTets1, exp4MaxVolume1 },
        { exp4TargetTets2, exp4MaxVolume2 },
        { exp4TargetTets3, exp4MaxVolume3 },
    };
    targets.erase(std::remove_if(targets.begin(), targets.end(), [](const TargetSpec& s) { return s.target <= 0; }), targets.end());
    if (targets.empty()) {
        targets = { { 5000, 0.0f }, { 20000, 0.0f }, { 50000, 0.0f } };
    }

    const std::string stlPath = stlFile;
    std::string baseArgs = stripTetgenAFlag(tetgenArgs);
    if (baseArgs.empty()) baseArgs = "pq20";
    if (baseArgs.find('p') == std::string::npos && baseArgs.find('P') == std::string::npos) {
        baseArgs = "p" + baseArgs;
    }

    const int oldOmpThreads = omp_get_max_threads();
    const int oldEigenThreads = Eigen::nbThreads();

    std::vector<BenchRow> rows;
    std::cout << "[Experiment4] Start PERF benchmark. Output=" << outputDir << "\n";
    std::cout << "[Experiment4] STL=" << stlPath << ", baseArgs=" << baseArgs
              << ", targets=" << targets.size() << ", threads=" << threads.size()
              << ", maxProcs=" << maxThreads << "\n";

    float startMaxVol = exp4MaxVolumeStart;
    if (startMaxVol <= 0.0f) {
        startMaxVol = 1e-4f;
    }

    for (size_t mi = 0; mi < targets.size(); ++mi) {
        const int targetTets = targets[mi].target;
        float maxVol = targets[mi].fixedMaxVol;
        if (maxVol <= 0.0f) {
            maxVol = tuneMaxVolumeForTargetTets(stlPath, baseArgs, targetTets, startMaxVol, exp4TuneIters);
        }

        BuiltObject built = buildFromStlWithMaxVolume(stlPath, baseArgs, maxVol);
        const std::string meshLabel = "target" + std::to_string(targetTets);

        std::cout << "[Experiment4] Mesh " << (mi + 1) << "/" << targets.size()
                  << " " << meshLabel << ": tets=" << built.tetCount
                  << ", points=" << built.pointCount
                  << ", maxVolume=" << built.maxVolume
                  << ", args=" << built.tetgenArgsUsed << "\n";

        for (int t : threads) {
            BenchRow r = benchObject(built.object, built, t, pbdIterations, warmupFrames, measureFrames);
            r.meshLabel = meshLabel;
            r.targetTets = targetTets;
            rows.push_back(r);
            std::cout << "[Experiment4]  threads=" << t
                      << " fps=" << r.fps
                      << " ms(frame)=" << r.msMean
                      << " ms(prime)=" << r.msPrimeMean
                      << " ms(pbd)=" << r.msPbdMean
                      << "\n";
        }

        destroyObjectGraph(built.object);
        startMaxVol = maxVol;
    }

    omp_set_num_threads(oldOmpThreads);
    Eigen::setNbThreads(oldEigenThreads);

    // Derive speedup/efficiency relative to the smallest thread count row for each mesh.
    struct Baseline {
        int threads = 0;
        double fps = 0.0;
    };
    std::unordered_map<std::string, Baseline> baselineByMesh;
    for (const auto& r : rows) {
        auto& b = baselineByMesh[r.meshLabel];
        if (b.threads == 0 || r.threads < b.threads) {
            b.threads = r.threads;
            b.fps = r.fps;
        } else if (r.threads == b.threads) {
            // In case of duplicates, keep the higher FPS.
            b.fps = std::max(b.fps, r.fps);
        }
    }

    const fs::path csvPath = fs::path(outputDir) / "experiment4_performance.csv";
    std::ofstream csv(csvPath);
    if (!csv.is_open()) {
        std::cerr << "[Experiment4] Failed to write " << csvPath.string() << "\n";
        return;
    }
    csv << "mesh_label,mesh_source,tetgen_args,target_tets,tet_count,point_count,max_volume,threads,pbd_iterations,warmup_frames,measure_frames,seconds,fps,ms_mean,ms_prime_mean,ms_pbd_mean,baseline_threads,baseline_fps,speedup,efficiency\n";
    for (const auto& r : rows) {
        const auto itB = baselineByMesh.find(r.meshLabel);
        const int bThreads = (itB == baselineByMesh.end()) ? 0 : itB->second.threads;
        const double bFps = (itB == baselineByMesh.end()) ? 0.0 : itB->second.fps;
        const double speedup = (bFps > 0.0) ? (r.fps / bFps) : 0.0;
        const double denom = (bThreads > 0) ? (static_cast<double>(r.threads) / static_cast<double>(bThreads)) : 0.0;
        const double efficiency = (denom > 0.0) ? (speedup / denom) : 0.0;
        csv << r.meshLabel << ","
            << r.meshSource << ","
            << r.tetgenArgsUsed << ","
            << r.targetTets << ","
            << r.tetCount << ","
            << r.pointCount << ","
            << r.maxVolume << ","
            << r.threads << ","
            << r.pbdIterations << ","
            << r.warmupFrames << ","
            << r.measureFrames << ","
            << r.seconds << ","
            << r.fps << ","
            << r.msMean << ","
            << r.msPrimeMean << ","
            << r.msPbdMean << ","
            << bThreads << ","
            << bFps << ","
            << speedup << ","
            << efficiency
            << "\n";
    }
    csv.close();

    const fs::path metaPath = fs::path(outputDir) / "experiment4_metadata.txt";
    std::ofstream meta(metaPath);
    if (meta.is_open()) {
        meta << "timeStep=" << timeStep << "\n";
        meta << "youngs=" << youngs << "\n";
        meta << "poisson=" << poisson << "\n";
        meta << "groupNumX=" << groupNumX << "\n";
        meta << "groupNumY=" << groupNumY << "\n";
        meta << "groupNumZ=" << groupNumZ << "\n";
        meta << "pbdIterations=" << pbdIterations << "\n";
        meta << "warmupFrames=" << warmupFrames << "\n";
        meta << "measureFrames=" << measureFrames << "\n";
        meta << "maxProcs=" << maxThreads << "\n";
        meta << "threads=";
        for (size_t i = 0; i < threads.size(); ++i) meta << (i ? "," : "") << threads[i];
        meta << "\n";
        meta << "targets=";
        for (size_t i = 0; i < targets.size(); ++i) meta << (i ? "," : "") << targets[i].target;
        meta << "\n";
        meta << "stlFile=" << stlPath << "\n";
        meta << "tetgenArgsBase=" << baseArgs << "\n";
        meta.close();
    }

    std::cout << "[Experiment4] Saved " << rows.size() << " rows to " << csvPath.string() << "\n";
}
