#include "Experiment2.h"
#include "Object.h"
#include "Vertex.h"
#include "Tetrahedron.h"
#include "Group.h"
#include "params.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>

Experiment2& Experiment2::instance() {
    static Experiment2 inst;
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

void Experiment2::init(Object* obj, const std::vector<Vertex*>& physical) {
    object = obj;
    physicalVertices = physical;

    config.settleSteps = exp2SettleSteps;
    config.dragSteps = exp2DragSteps;
    config.holdSteps = exp2HoldSteps;
    config.poissonIncompressible = exp2PoissonIncompressible;
    config.dragDistanceBboxScale = exp2DragDistanceBboxScale;
    config.dragDistanceMin = exp2DragDistanceMin;
    config.dragDistanceMax = exp2DragDistanceMax;
    config.anchorSliceFrac = exp2AnchorSliceFrac;
    config.pullSliceFrac = exp2PullSliceFrac;
    config.minRegionVertexCount = exp2MinRegionVertexCount;
    config.pullStiffness = exp2PullStiffness;
    config.pullMaxAccel = exp2PullMaxAccel;
    config.pbdIterations = exp2PbdIterations;
    config.resetAfterFinish = exp2ResetAfterFinish;

    allVertices.clear();
    allTetrahedra.clear();
    verticesByIndex.clear();
    if (object) {
        std::unordered_set<Vertex*> visited;
        for (Group& g : object->groups) {
            for (const auto& kv : g.verticesMap) {
                Vertex* v = kv.second;
                if (!v) continue;
                if (visited.insert(v).second) {
                    allVertices.push_back(v);
                }
                verticesByIndex[v->index].push_back(v);
            }
            for (Tetrahedron* t : g.tetrahedra) {
                if (t) allTetrahedra.push_back(t);
            }
        }
    }

    // Default sequence: baseline nu (from parameters) + incompressible nu.
    // Baseline nu will be filled on start from current poisson.
    sequence = {
        { "baseline", poisson },
        { "incompressible", config.poissonIncompressible },
    };
}

void Experiment2::requestStart() {
    if (!object) {
        std::cerr << "[Experiment2] Cannot start: Object not initialized.\n";
        return;
    }
    if (state != State::Idle) return;
    state = State::PendingStart;
}

bool Experiment2::isActive() const {
    return state != State::Idle;
}

bool Experiment2::wantsDrag() const {
    // EXP2 uses per-vertex deterministic forces; no mouse-like single-point drag needed.
    return false;
}

std::string Experiment2::buttonLabel() const {
    if (state == State::Idle) return "START EXP2 (UNIAXIAL)";
    if (state == State::SaveAndFinish) return "SAVE EXP2";
    return "EXP2 (UNIAXIAL)";
}

Vertex* Experiment2::targetVertex() const {
    return selectedTarget;
}

Eigen::Vector3f Experiment2::desiredTargetPosition() const {
    return currentDesiredPos;
}

const std::vector<Vertex*>& Experiment2::forceVertices() const {
    return allVertices;
}

int Experiment2::pbdIterationsThisFrame(int defaultIterations) const {
    if (!isActive()) return defaultIterations;
    return 10;
}

Vertex* Experiment2::pickDeterministicTarget() const {
    if (physicalVertices.empty()) return nullptr;

    Eigen::Vector3f minBound(std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
    Eigen::Vector3f maxBound(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
    for (const Vertex* v : physicalVertices) {
        minBound.x() = std::min(minBound.x(), v->initx);
        minBound.y() = std::min(minBound.y(), v->inity);
        minBound.z() = std::min(minBound.z(), v->initz);
        maxBound.x() = std::max(maxBound.x(), v->initx);
        maxBound.y() = std::max(maxBound.y(), v->inity);
        maxBound.z() = std::max(maxBound.z(), v->initz);
    }

    const float zRange = std::max(1e-6f, maxBound.z() - minBound.z());
    const float frontThresholdZ = minBound.z() + 0.80f * zRange;

    Vertex* best = nullptr;
    auto consider = [&](Vertex* v) {
        if (!v) return;
        if (v->isFixed) return;
        if (!best || v->initx > best->initx) best = v;
    };
    for (Vertex* v : physicalVertices) {
        if (v->initz >= frontThresholdZ) consider(v);
    }
    if (best) return best;
    for (Vertex* v : physicalVertices) consider(v);
    return best;
}

void Experiment2::pickRegionsDeterministic() {
    anchorIndices.clear();
    pullIndices.clear();
    selectedTarget = nullptr;

    if (verticesByIndex.empty()) return;

    std::vector<Vertex*> reps;
    reps.reserve(verticesByIndex.size());
    for (const auto& kv : verticesByIndex) {
        if (!kv.second.empty() && kv.second.front()) reps.push_back(kv.second.front());
    }
    std::sort(reps.begin(), reps.end(), [](const Vertex* a, const Vertex* b) {
        return a->index < b->index;
    });
    if (reps.empty()) return;

    Eigen::Vector3f minBound(std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
    Eigen::Vector3f maxBound(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
    for (const Vertex* v : reps) {
        minBound.x() = std::min(minBound.x(), v->initx);
        minBound.y() = std::min(minBound.y(), v->inity);
        minBound.z() = std::min(minBound.z(), v->initz);
        maxBound.x() = std::max(maxBound.x(), v->initx);
        maxBound.y() = std::max(maxBound.y(), v->inity);
        maxBound.z() = std::max(maxBound.z(), v->initz);
    }

    const float xRange = std::max(1e-6f, maxBound.x() - minBound.x());
    float anchorFrac = std::clamp(config.anchorSliceFrac, 0.0f, 0.45f);
    float pullFrac = std::clamp(config.pullSliceFrac, 0.0f, 0.45f);
    if (anchorFrac + pullFrac > 0.90f) {
        const float scale = 0.90f / std::max(1e-6f, anchorFrac + pullFrac);
        anchorFrac *= scale;
        pullFrac *= scale;
    }

    auto selectSlice = [&](bool isAnchor, float frac) {
        std::unordered_set<int> indices;
        const float xThresh = isAnchor ? (minBound.x() + frac * xRange) : (maxBound.x() - frac * xRange);
        for (Vertex* v : reps) {
            if (!v) continue;
            if (isAnchor) {
                if (v->initx <= xThresh) indices.insert(v->index);
            } else {
                if (v->initx >= xThresh) indices.insert(v->index);
            }
        }
        return indices;
    };

    std::unordered_set<int> anchorSet;
    std::unordered_set<int> pullSet;
    const int minCount = std::max(1, config.minRegionVertexCount);
    for (int iter = 0; iter < 6; ++iter) {
        anchorSet = selectSlice(true, anchorFrac);
        if (static_cast<int>(anchorSet.size()) >= minCount) break;
        anchorFrac = std::min(0.45f, anchorFrac * 1.5f);
    }
    for (int iter = 0; iter < 6; ++iter) {
        pullSet = selectSlice(false, pullFrac);
        if (static_cast<int>(pullSet.size()) >= minCount) break;
        pullFrac = std::min(0.45f, pullFrac * 1.5f);
    }

    // Ensure non-empty fallbacks.
    if (anchorSet.empty()) {
        Vertex* minX = nullptr;
        for (Vertex* v : reps) {
            if (!v) continue;
            if (!minX || v->initx < minX->initx) minX = v;
        }
        if (minX) anchorSet.insert(minX->index);
    }
    if (pullSet.empty()) {
        Vertex* maxX = nullptr;
        for (Vertex* v : reps) {
            if (!v) continue;
            if (!maxX || v->initx > maxX->initx) maxX = v;
        }
        if (maxX) pullSet.insert(maxX->index);
    }

    // Avoid overlap (can happen with tiny meshes or extreme fractions).
    for (int idx : anchorSet) pullSet.erase(idx);

    anchorIndices.assign(anchorSet.begin(), anchorSet.end());
    pullIndices.assign(pullSet.begin(), pullSet.end());
    std::sort(anchorIndices.begin(), anchorIndices.end());
    std::sort(pullIndices.begin(), pullIndices.end());

    // Representative target for metadata/debug: pick a pull-slice vertex near face center (y/z).
    Eigen::Vector3f centerYZ(0.0f, 0.0f, 0.0f);
    int centerCount = 0;
    for (Vertex* v : reps) {
        if (!v) continue;
        if (pullSet.find(v->index) == pullSet.end()) continue;
        centerYZ.y() += v->inity;
        centerYZ.z() += v->initz;
        ++centerCount;
    }
    if (centerCount > 0) {
        centerYZ.y() /= static_cast<float>(centerCount);
        centerYZ.z() /= static_cast<float>(centerCount);
    } else {
        centerYZ.y() = 0.5f * (minBound.y() + maxBound.y());
        centerYZ.z() = 0.5f * (minBound.z() + maxBound.z());
    }

    float bestScore = std::numeric_limits<float>::infinity();
    for (Vertex* v : reps) {
        if (!v) continue;
        if (pullSet.find(v->index) == pullSet.end()) continue;
        if (v->isFixed) continue;
        const float dy = v->inity - centerYZ.y();
        const float dz = v->initz - centerYZ.z();
        const float score = 10.0f * std::abs(v->initx - maxBound.x()) + std::sqrt(dy * dy + dz * dz);
        if (score < bestScore) {
            bestScore = score;
            selectedTarget = v;
        }
    }
    if (!selectedTarget) {
        selectedTarget = pickDeterministicTarget();
    }
}

void Experiment2::computeDragDistance() {
    if (physicalVertices.empty()) {
        dragDistance = std::min(config.dragDistanceMax, std::max(config.dragDistanceMin, 0.5f));
        return;
    }
    Eigen::Vector3f minBound(std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
    Eigen::Vector3f maxBound(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
    for (const Vertex* v : physicalVertices) {
        minBound.x() = std::min(minBound.x(), v->initx);
        minBound.y() = std::min(minBound.y(), v->inity);
        minBound.z() = std::min(minBound.z(), v->initz);
        maxBound.x() = std::max(maxBound.x(), v->initx);
        maxBound.y() = std::max(maxBound.y(), v->inity);
        maxBound.z() = std::max(maxBound.z(), v->initz);
    }
    const float xRange = std::max(1e-6f, maxBound.x() - minBound.x());
    float dist = std::max(config.dragDistanceMin, xRange * config.dragDistanceBboxScale);
    dist = std::min(dist, config.dragDistanceMax);
    dist = std::min(dist, std::max(0.0f, dragMaxDisplacement) * 0.95f);
    dragDistance = std::max(config.dragDistanceMin, dist);
}

void Experiment2::snapshotFixedFlags() {
    fixedSnapshot.clear();
    fixedSnapshot.reserve(allVertices.size());
    for (Vertex* v : allVertices) {
        if (!v) continue;
        fixedSnapshot.emplace_back(v, v->isFixed);
    }
}

void Experiment2::restoreFixedFlags() {
    for (const auto& p : fixedSnapshot) {
        if (p.first) p.first->isFixed = p.second;
    }
}

void Experiment2::applyAnchorFixing() {
    if (anchorIndices.empty()) return;

    std::unordered_set<int> anchorSet(anchorIndices.begin(), anchorIndices.end());
    std::unordered_set<int> pullSet(pullIndices.begin(), pullIndices.end());
    for (int idx : anchorIndices) {
        auto it = verticesByIndex.find(idx);
        if (it == verticesByIndex.end()) continue;
        for (Vertex* v : it->second) {
            if (v) v->isFixed = true;
        }
    }
    for (int idx : pullIndices) {
        auto it = verticesByIndex.find(idx);
        if (it == verticesByIndex.end()) continue;
        for (Vertex* v : it->second) {
            if (v) v->isFixed = false;
        }
    }
}

void Experiment2::resetSimulationToInitial() {
    if (!object) return;

    std::unordered_set<Vertex*> visited;
    visited.reserve(allVertices.size() * 2);
    for (Vertex* v : allVertices) {
        if (!v) continue;
        if (!visited.insert(v).second) continue;
        v->x = v->initx;
        v->y = v->inity;
        v->z = v->initz;
        v->velx = 0.0f;
        v->vely = 0.0f;
        v->velz = 0.0f;
    }

    for (Group& g : object->groups) {
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

void Experiment2::applyMaterialForRun(const RunSpec& spec) {
    if (!object) return;

    poisson = std::min(0.49f, std::max(0.0f, spec.poisson));
    youngs1 = youngs;
    youngs2 = youngs;
    youngs3 = youngs;

#pragma omp parallel for
    for (int i = 0; i < object->groupNum; ++i) {
        object->groups[i].calGroupK(youngs, poisson);
        object->groups[i].calLHS();
    }
}

void Experiment2::restoreMaterial() {
    if (!object) return;
    if (!hasOldMaterial) return;

    poisson = poissonOriginal;
    youngs1 = oldYoungs1;
    youngs2 = oldYoungs2;
    youngs3 = oldYoungs3;

#pragma omp parallel for
    for (int i = 0; i < object->groupNum; ++i) {
        if (std::abs(youngs1 - youngs2) > 1e-1f || std::abs(youngs1 - youngs3) > 1e-1f) {
            object->groups[i].calGroupKAni(youngs1, youngs2, youngs3, poisson);
        } else {
            object->groups[i].calGroupK(youngs, poisson);
        }
        object->groups[i].calLHS();
    }
    hasOldMaterial = false;
}

double Experiment2::computeTotalVolume() const {
    double sum = 0.0;
    for (Tetrahedron* t : allTetrahedra) {
        if (!t) continue;
        sum += static_cast<double>(t->calVolumeTetra());
    }
    return sum;
}

void Experiment2::saveData() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outputDir, ec);
    if (ec) {
        std::cerr << "[Experiment2] Failed to create output dir: " << outputDir << " (" << ec.message() << ")\n";
        return;
    }

    const fs::path csvPath = fs::path(outputDir) / "experiment2_volume.csv";
    std::ofstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[Experiment2] Failed to open " << csvPath.string() << " for writing\n";
        return;
    }
    file << "run_index,run_name,poisson,stage,sim_time,imposed_displacement,actual_displacement,volume,volume_ratio,vertex_index"
         << ",pull_set_size,anchor_set_size,actual_displacement_mean,actual_displacement_rms,actual_displacement_min,actual_displacement_max\n";
    for (const auto& p : data) {
        const RunSpec& spec = sequence[static_cast<size_t>(p.runIndex)];
        file << p.runIndex << ","
             << spec.name << ","
             << spec.poisson << ","
             << p.stage << ","
             << p.simTime << ","
             << p.imposedDisplacement << ","
             << p.actualDisplacement << ","
             << p.volume << ","
             << p.volumeRatio << ","
             << (selectedTarget ? selectedTarget->index : -1)
             << ","
             << pullIndices.size() << ","
             << anchorIndices.size() << ","
             << p.actualDisplacementMean << ","
             << p.actualDisplacementRms << ","
             << p.actualDisplacementMin << ","
             << p.actualDisplacementMax
             << "\n";
    }
    file.close();

    const fs::path metaPath = fs::path(outputDir) / "experiment2_metadata.txt";
    std::ofstream meta(metaPath);
    if (meta.is_open()) {
        meta << "vertex_index=" << (selectedTarget ? selectedTarget->index : -1) << "\n";
        if (selectedTarget) {
            meta << "vertex_init=(" << selectedTarget->initx << "," << selectedTarget->inity << "," << selectedTarget->initz << ")\n";
        }
        meta << "timeStep=" << timeStep << "\n";
        meta << "youngs=" << youngs << "\n";
        meta << "poisson_original=" << poissonOriginal << "\n";
        meta << "anchorSliceFrac=" << config.anchorSliceFrac << "\n";
        meta << "pullSliceFrac=" << config.pullSliceFrac << "\n";
        meta << "minRegionVertexCount=" << config.minRegionVertexCount << "\n";
        meta << "anchor_set_size=" << anchorIndices.size() << "\n";
        meta << "pull_set_size=" << pullIndices.size() << "\n";
        meta << "pullStiffness=" << config.pullStiffness << "\n";
        meta << "pullMaxAccel=" << config.pullMaxAccel << "\n";
        meta << "dragInfluenceRadius=" << dragInfluenceRadius << "\n";
        meta << "dragStiffness=" << dragStiffness << "\n";
        meta << "dragMaxAccel=" << dragMaxAccel << "\n";
        meta << "dragMaxDisplacement=" << dragMaxDisplacement << "\n";
        meta << "dragDistance=" << dragDistance << "\n";
        meta << "settleSteps=" << config.settleSteps << "\n";
        meta << "dragSteps=" << config.dragSteps << "\n";
        meta << "holdSteps=" << config.holdSteps << "\n";
        meta << "pbdIterations=" << config.pbdIterations << "\n";
        meta << "sequence=";
        for (size_t i = 0; i < sequence.size(); ++i) {
            meta << (i ? ";" : "");
            meta << sequence[i].name << ":nu" << sequence[i].poisson;
        }
        meta << "\n";
        meta.close();
    }

    std::cout << "[Experiment2] Saved " << data.size() << " samples to " << csvPath.string() << "\n";
}

void Experiment2::appendVertexForces(std::unordered_map<int, Eigen::Vector3f>& dragForces) const {
    if (!object) return;
    if (state != State::Drag && state != State::Hold) return;
    if (pullStartPosByIndex.empty()) return;

    const Eigen::Vector3f dir(1.0f, 0.0f, 0.0f);
    const float stiffness = std::max(0.0f, config.pullStiffness);
    const float maxAccel = std::max(0.0f, config.pullMaxAccel);

    for (int idx : pullIndices) {
        auto itStart = pullStartPosByIndex.find(idx);
        if (itStart == pullStartPosByIndex.end()) continue;
        auto itV = verticesByIndex.find(idx);
        if (itV == verticesByIndex.end() || itV->second.empty()) continue;
        Eigen::Vector3f cur = Eigen::Vector3f::Zero();
        int count = 0;
        bool anyFixed = false;
        for (const Vertex* v : itV->second) {
            if (!v) continue;
            if (v->isFixed) anyFixed = true;
            cur += Eigen::Vector3f(v->x, v->y, v->z);
            ++count;
        }
        if (count == 0) continue;
        if (anyFixed) continue;
        cur /= static_cast<float>(count);

        const Eigen::Vector3f desired = itStart->second + dir * currentImposedDistance;
        Eigen::Vector3f accel = (desired - cur) * stiffness;
        const float n = accel.norm();
        if (maxAccel > 0.0f && n > maxAccel) {
            accel *= (maxAccel / std::max(1e-12f, n));
        }
        dragForces[idx] += accel;
    }
}

void Experiment2::update() {
    if (!object) return;

    switch (state) {
        case State::Idle:
            return;
        case State::PendingStart: {
            pickRegionsDeterministic();
            if (!selectedTarget) {
                std::cerr << "[Experiment2] Cannot start: no selectable target vertex.\n";
                state = State::Idle;
                return;
            }

            poissonOriginal = poisson;
            oldYoungs1 = youngs1;
            oldYoungs2 = youngs2;
            oldYoungs3 = youngs3;
            hasOldMaterial = true;
            snapshotFixedFlags();

            // Fill baseline from current poisson (whatever demo is using).
            if (!sequence.empty()) sequence[0].poisson = poissonOriginal;
            computeDragDistance();

            outputDir = (std::filesystem::path("..") / "out" / "experiment2" / nowTimestamp()).string();
            data.clear();
            runIndex = 0;
            stepInState = 0;
            lastSampleStep = 0;
            currentImposedDistance = 0.0f;

            std::cout << "[Experiment2] Start (one-click). Vertex=" << selectedTarget->index
                      << ", dragDistance=" << dragDistance
                      << ", steps(settle/drag/hold)=" << config.settleSteps << "/" << config.dragSteps << "/" << config.holdSteps
                      << ", pbdIterations=" << config.pbdIterations
                      << ", anchor/pull=" << anchorIndices.size() << "/" << pullIndices.size()
                      << ", runs=" << sequence.size() << "\n";

            state = State::ApplyRunAndReset;
            return;
        }
        case State::ApplyRunAndReset: {
            if (runIndex >= static_cast<int>(sequence.size())) {
                state = State::SaveAndFinish;
                return;
            }
            resetSimulationToInitial();
            restoreFixedFlags();
            applyAnchorFixing();
            applyMaterialForRun(sequence[static_cast<size_t>(runIndex)]);
            currentDesiredPos = Eigen::Vector3f(selectedTarget->x, selectedTarget->y, selectedTarget->z);
            stepInState = 0;
            lastSampleStep = 0;
            currentImposedDistance = 0.0f;
            pullStartPosByIndex.clear();
            volume0 = 0.0;
            state = State::Settle;
            return;
        }
        case State::Settle: {
            currentDesiredPos = Eigen::Vector3f(selectedTarget->x, selectedTarget->y, selectedTarget->z);
            ++stepInState;
            if (stepInState >= std::max(1, config.settleSteps)) {
                runStartPos = Eigen::Vector3f(selectedTarget->x, selectedTarget->y, selectedTarget->z);
                for (int idx : pullIndices) {
                    auto itV = verticesByIndex.find(idx);
                    if (itV == verticesByIndex.end() || itV->second.empty()) continue;
                    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
                    int count = 0;
                    for (const Vertex* v : itV->second) {
                        if (!v) continue;
                        mean += Eigen::Vector3f(v->x, v->y, v->z);
                        ++count;
                    }
                    if (count <= 0) continue;
                    pullStartPosByIndex[idx] = mean / static_cast<float>(count);
                }
                stepInState = 0;
                lastSampleStep = 0;
                volume0 = 0.0; // will be set by onAfterPhysics at first drag sample
                state = State::Drag;
            }
            return;
        }
        case State::Drag: {
            const int steps = std::max(1, config.dragSteps);
            const float denom = static_cast<float>(std::max(1, steps - 1));
            const float t = std::min(1.0f, static_cast<float>(stepInState) / denom);
            currentImposedDistance = dragDistance * t;
            currentDesiredPos = runStartPos + Eigen::Vector3f(1.0f, 0.0f, 0.0f) * currentImposedDistance;
            lastSampleStep = stepInState;
            ++stepInState;
            if (stepInState >= steps) {
                stepInState = 0;
                state = State::Hold;
            }
            return;
        }
        case State::Hold: {
            currentImposedDistance = dragDistance;
            currentDesiredPos = runStartPos + Eigen::Vector3f(1.0f, 0.0f, 0.0f) * currentImposedDistance;
            lastSampleStep = config.dragSteps + stepInState;
            ++stepInState;
            if (stepInState >= std::max(1, config.holdSteps)) {
                ++runIndex;
                state = State::ApplyRunAndReset;
            }
            return;
        }
        case State::SaveAndFinish: {
            saveData();
            restoreMaterial();
            restoreFixedFlags();
            if (config.resetAfterFinish) {
                resetSimulationToInitial();
            }
            std::cout << "[Experiment2] Finished.\n";
            state = State::Idle;
            return;
        }
        default:
            return;
    }
}

void Experiment2::onAfterPhysics() {
    if (!object) return;
    if (!selectedTarget) return;
    if (state != State::Drag && state != State::Hold) return;

    const double vol = computeTotalVolume();
    if (volume0 <= 0.0) {
        // First sample after settle becomes the reference volume.
        volume0 = std::max(1e-12, vol);
    }
    const double ratio = vol / volume0;

    const float simTime = static_cast<float>(lastSampleStep) * timeStep;
    const Eigen::Vector3f dir(1.0f, 0.0f, 0.0f);
    const float imposed = std::max(0.0f, currentImposedDistance);
    const Eigen::Vector3f cur(selectedTarget->x, selectedTarget->y, selectedTarget->z);
    const float actual = std::max(0.0f, (cur - runStartPos).dot(dir));

    float meanDisp = 0.0f;
    float rmsDisp = 0.0f;
    float minDisp = 0.0f;
    float maxDisp = 0.0f;
    int count = 0;
    for (int idx : pullIndices) {
        auto itStart = pullStartPosByIndex.find(idx);
        auto itV = verticesByIndex.find(idx);
        if (itStart == pullStartPosByIndex.end()) continue;
        if (itV == verticesByIndex.end() || itV->second.empty()) continue;

        Eigen::Vector3f curV = Eigen::Vector3f::Zero();
        int vCount = 0;
        for (const Vertex* v : itV->second) {
            if (!v) continue;
            curV += Eigen::Vector3f(v->x, v->y, v->z);
            ++vCount;
        }
        if (vCount <= 0) continue;
        curV /= static_cast<float>(vCount);

        const float d = (curV - itStart->second).dot(dir);
        if (count == 0) {
            minDisp = d;
            maxDisp = d;
        } else {
            minDisp = std::min(minDisp, d);
            maxDisp = std::max(maxDisp, d);
        }
        meanDisp += d;
        rmsDisp += d * d;
        ++count;
    }
    if (count > 0) {
        meanDisp /= static_cast<float>(count);
        rmsDisp = std::sqrt(rmsDisp / static_cast<float>(count));
    } else {
        meanDisp = actual;
        rmsDisp = actual;
        minDisp = actual;
        maxDisp = actual;
    }

    data.push_back(DataPoint{
        runIndex,
        simTime,
        imposed,
        actual,
        meanDisp,
        rmsDisp,
        minDisp,
        maxDisp,
        vol,
        ratio,
        (state == State::Drag) ? "drag" : "hold",
    });
}
