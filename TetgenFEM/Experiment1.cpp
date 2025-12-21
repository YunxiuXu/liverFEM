#include "Experiment1.h"
#include "Object.h"
#include "Vertex.h"
#include "params.h"
#include "Group.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

Experiment1& Experiment1::instance() {
    static Experiment1 instance;
    return instance;
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

void Experiment1::init(Object* obj, const std::vector<Vertex*>& physical) {
    object = obj;
    physicalVertices = physical;

    config.settleSteps = exp1SettleSteps;
    config.loadRampSteps = exp1DragSteps;
    config.holdSteps = exp1HoldSteps;
    config.pullAccel = exp1PullAccel;
    config.influenceRadius = exp1ForceInfluenceRadius;
    config.pbdIterationsFast = exp1PbdIterationsFast;
    config.pbdIterationsReference = exp1PbdIterationsReference;
    config.resetAfterFinish = exp1ResetAfterFinish;

    runs = {
        { "fast", std::max(1, config.pbdIterationsFast) },
        { "reference", std::max(1, config.pbdIterationsReference) },
    };
    accelSweep = {
        { std::max(0.0f, exp1SweepAccel1) },
        { std::max(0.0f, exp1SweepAccel2) },
        { std::max(0.0f, exp1SweepAccel3) },
    };

    allVertices.clear();
    if (object) {
        std::unordered_set<Vertex*> visited;
        visited.reserve(static_cast<size_t>(object->groupNum) * 1024);
        for (Group& g : object->groups) {
            for (const auto& kv : g.verticesMap) {
                Vertex* v = kv.second;
                if (!v) continue;
                if (visited.insert(v).second) {
                    allVertices.push_back(v);
                }
            }
        }
    }
}

void Experiment1::requestStart() {
    if (!object) {
        std::cerr << "[Experiment1] Cannot start: Object not initialized.\n";
        return;
    }
    if (state != State::Idle) return;
    state = State::PendingStart;
}

bool Experiment1::isActive() const {
    return state != State::Idle;
}

std::string Experiment1::buttonLabel() const {
    if (state == State::Idle) return "START EXP1";
    if (state == State::SaveAndFinish) return "SAVE EXP1";
    return "EXP1";
}

Vertex* Experiment1::targetVertex() const {
    return selectedTarget;
}

int Experiment1::pbdIterationsThisFrame(int defaultIterations) const {
    if (state == State::Idle || runs.empty()) return defaultIterations;
    const int idx = std::max(0, std::min(withinAccelRunIndex, static_cast<int>(runs.size()) - 1));
    return std::max(1, runs[static_cast<size_t>(idx)].pbdIterations);
}

void Experiment1::appendVertexForces(std::vector<Eigen::Vector3f>& forcesOut) const {
    if (state != State::LoadRamp && state != State::HoldLoad) return;
    if (!selectedTarget) return;
    if (forceRegionVertices.empty()) return;

    Eigen::Vector3f dir = config.pullDirection;
    if (dir.squaredNorm() < 1e-12f) return;
    dir.normalize();

    const float accelMag = std::max(0.0f, config.pullAccel) * std::max(0.0f, currentLoadScale);
    if (accelMag <= 0.0f) return;

    const float r = std::max(1e-6f, config.influenceRadius);
    // Optimization: forcesOut is now a vector, ensuring fast indexing
    for (Vertex* v : forceRegionVertices) {
        if (!v) continue;
        if (v->isFixed) continue;
        const Eigen::Vector3f p(v->initx, v->inity, v->initz);
        const float dist = (p - targetInitPos).norm();
        if (dist > r) continue;

        float falloff = std::max(0.0f, 1.0f - dist / r);
        if (falloff <= 0.0f) continue;
        if (v->index == selectedTarget->index) falloff *= 1.5f;

        if (v->index < (int)forcesOut.size()) {
            forcesOut[v->index] += (accelMag * falloff) * dir;
        }
    }
}

Vertex* Experiment1::pickDeterministicTarget() const {
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

void Experiment1::resetSimulationToInitial() {
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

void Experiment1::applyIsotropicMaterial() {
    if (!object) return;

    youngs1 = youngs;
    youngs2 = youngs;
    youngs3 = youngs;

#pragma omp parallel for
    for (int i = 0; i < object->groupNum; ++i) {
        object->groups[i].calGroupK(youngs, poisson);
        object->groups[i].calLHS();
    }
}

void Experiment1::restoreMaterial() {
    if (!object) return;
    if (!hasOldMaterial) return;

    youngs1 = oldYoungs1;
    youngs2 = oldYoungs2;
    youngs3 = oldYoungs3;
    if (hasOldPoisson) {
        poisson = oldPoisson;
        hasOldPoisson = false;
    }

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

void Experiment1::captureSnapshot(const RunSpec& spec) {
    Snapshot snap;
    snap.runName = spec.name;
    snap.pullAccel = config.pullAccel;
    snap.pbdIterations = spec.pbdIterations;
    snap.positions.reserve(physicalVertices.size());
    for (Vertex* v : physicalVertices) {
        snap.positions.emplace_back(v->x, v->y, v->z);
    }
    if (selectedTarget) {
        const Eigen::Vector3f init(selectedTarget->initx, selectedTarget->inity, selectedTarget->initz);
        const Eigen::Vector3f cur(selectedTarget->x, selectedTarget->y, selectedTarget->z);
        snap.targetDisplacement = (cur - init).norm();
    }
    snapshots.push_back(std::move(snap));
}

void Experiment1::saveData() const {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outputDir, ec);
    if (ec) {
        std::cerr << "[Experiment1] Failed to create output dir: " << outputDir << " (" << ec.message() << ")\n";
        return;
    }

    const fs::path csvPath = fs::path(outputDir) / "experiment1_positions.csv";
    std::ofstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[Experiment1] Failed to open " << csvPath.string() << " for writing\n";
        return;
    }

    file << "run_index,run_name,pull_accel,pbd_iterations,target_displacement,vertex_list_index,vertex_index,initx,inity,initz,x,y,z\n";
    for (size_t r = 0; r < snapshots.size(); ++r) {
        const auto& snap = snapshots[r];
        for (size_t i = 0; i < physicalVertices.size(); ++i) {
            const Vertex* v = physicalVertices[i];
            const Eigen::Vector3f& p = snap.positions[i];
            file << r << ","
                 << snap.runName << ","
                 << snap.pullAccel << ","
                 << snap.pbdIterations << ","
                 << snap.targetDisplacement << ","
                 << i << ","
                 << (v ? v->index : -1) << ","
                 << (v ? v->initx : 0.0f) << ","
                 << (v ? v->inity : 0.0f) << ","
                 << (v ? v->initz : 0.0f) << ","
                 << p.x() << "," << p.y() << "," << p.z()
                 << "\n";
        }
    }
    file.close();

    const fs::path sweepPath = fs::path(outputDir) / "experiment1_sweep_summary.csv";
    std::ofstream sweep(sweepPath);
    if (sweep.is_open()) {
        // BBox diag for normalization.
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
        const double diag = (maxBound - minBound).cast<double>().norm();

        sweep << "pull_accel,pbd_fast,pbd_reference,target_disp_fast,target_disp_reference,target_disp_ratio_fast_over_ref,target_disp_rel_error,rmse,max_error,rmse_over_diag,max_over_diag,rmse_over_target_disp_ref,max_over_target_disp_ref\n";
        for (const auto& row : summary) {
            const double rmseOver = (diag > 0.0) ? row.rmse / diag : 0.0;
            const double maxOver = (diag > 0.0) ? row.maxErr / diag : 0.0;
            const double tdRef = std::max(1e-12, static_cast<double>(row.targetDispRef));
            const double tdRatio = static_cast<double>(row.targetDispFast) / tdRef;
            const double tdRelErr = tdRatio - 1.0;
            const double rmseOverTd = row.rmse / tdRef;
            const double maxOverTd = row.maxErr / tdRef;
            sweep << row.pullAccel << ","
                  << row.pbdFast << ","
                  << row.pbdRef << ","
                  << row.targetDispFast << ","
                  << row.targetDispRef << ","
                  << tdRatio << ","
                  << tdRelErr << ","
                  << row.rmse << ","
                  << row.maxErr << ","
                  << rmseOver << ","
                  << maxOver << ","
                  << rmseOverTd << ","
                  << maxOverTd << "\n";
        }
        sweep.close();
    }

    const fs::path metaPath = fs::path(outputDir) / "experiment1_metadata.txt";
    std::ofstream meta(metaPath);
    if (meta.is_open()) {
        meta << "vertex_index=" << (selectedTarget ? selectedTarget->index : -1) << "\n";
        if (selectedTarget) {
            meta << "vertex_init=(" << selectedTarget->initx << "," << selectedTarget->inity << "," << selectedTarget->initz << ")\n";
        }
        meta << "timeStep=" << timeStep << "\n";
        meta << "youngs=" << youngs << "\n";
        meta << "poisson=" << poisson << "\n";
        if (hasOldPoisson) meta << "poisson_original=" << oldPoisson << "\n";
        meta << "dragInfluenceRadius=" << dragInfluenceRadius << "\n";
        meta << "dragStiffness=" << dragStiffness << "\n";
        meta << "dragMaxAccel=" << dragMaxAccel << "\n";
        meta << "dragMaxDisplacement=" << dragMaxDisplacement << "\n";
        meta << "pullAccel=" << config.pullAccel << "\n";
        meta << "forceInfluenceRadius=" << config.influenceRadius << "\n";
        meta << "sweepAccels=" << exp1SweepAccel1 << "," << exp1SweepAccel2 << "," << exp1SweepAccel3 << "\n";
        meta << "settleSteps=" << config.settleSteps << "\n";
        meta << "loadRampSteps=" << config.loadRampSteps << "\n";
        meta << "holdSteps=" << config.holdSteps << "\n";
        meta << "runs=";
        for (size_t i = 0; i < runs.size(); ++i) {
            meta << (i ? ";" : "");
            meta << runs[i].name << ":pbd" << runs[i].pbdIterations;
        }
        meta << "\n";
        meta.close();
    }

    // Backward-compatible: write the last sweep row as experiment1_summary.txt.
    if (!summary.empty()) {
        const auto& last = summary.back();
        const fs::path summaryPath = fs::path(outputDir) / "experiment1_summary.txt";
        std::ofstream summaryFile(summaryPath);
        if (summaryFile.is_open()) {
            summaryFile << "rmse_fast_vs_reference=" << last.rmse << "\n";
            summaryFile << "max_error_fast_vs_reference=" << last.maxErr << "\n";
            summaryFile.close();
        }
    }

    std::cout << "[Experiment1] Saved snapshots=" << snapshots.size()
              << " to " << csvPath.string() << "\n";
}

void Experiment1::update() {
    if (!object) return;

    switch (state) {
        case State::Idle:
            return;
        case State::PendingStart: {
            selectedTarget = pickDeterministicTarget();
            if (!selectedTarget) {
                std::cerr << "[Experiment1] Cannot start: no selectable target vertex.\n";
                state = State::Idle;
                return;
            }
            targetInitPos = Eigen::Vector3f(selectedTarget->initx, selectedTarget->inity, selectedTarget->initz);

            oldYoungs1 = youngs1;
            oldYoungs2 = youngs2;
            oldYoungs3 = youngs3;
            oldPoisson = poisson;
            hasOldMaterial = true;
            hasOldPoisson = true;

            outputDir = (std::filesystem::path("..") / "out" / "experiment1" / nowTimestamp()).string();
            snapshots.clear();
            summary.clear();
            accelIndex = 0;
            withinAccelRunIndex = 0;
            stepInState = 0;
            currentLoadScale = 0.0f;

            forceRegionVertices = allVertices;

            std::cout << "[Experiment1] Start (one-click). Vertex=" << selectedTarget->index
                      << ", sweepAccels=" << accelSweep.size()
                      << ", steps(settle/ramp/hold)=" << config.settleSteps << "/" << config.loadRampSteps << "/" << config.holdSteps
                      << ", runsPerAccel=" << runs.size() << "\n";

            state = State::ApplyRunAndReset;
            return;
        }
        case State::ApplyRunAndReset: {
            if (accelSweep.empty()) {
                accelSweep = { { config.pullAccel } };
            }
            if (accelIndex >= static_cast<int>(accelSweep.size())) {
                state = State::SaveAndFinish;
                return;
            }
            if (withinAccelRunIndex >= static_cast<int>(runs.size())) {
                withinAccelRunIndex = 0;
                ++accelIndex;
                state = State::ApplyRunAndReset;
                return;
            }

            config.pullAccel = accelSweep[static_cast<size_t>(accelIndex)].pullAccel;
            resetSimulationToInitial();
            applyIsotropicMaterial();
            stepInState = 0;
            currentLoadScale = 0.0f;
            state = State::Settle;
            return;
        }
        case State::Settle: {
            ++stepInState;
            if (stepInState >= std::max(1, config.settleSteps)) {
                stepInState = 0;
                state = State::LoadRamp;
            }
            return;
        }
        case State::LoadRamp: {
            const int steps = std::max(1, config.loadRampSteps);
            const float denom = static_cast<float>(std::max(1, steps - 1));
            const float t = std::min(1.0f, static_cast<float>(stepInState) / denom);
            currentLoadScale = t;
            ++stepInState;
            if (stepInState >= steps) {
                stepInState = 0;
                state = State::HoldLoad;
            }
            return;
        }
        case State::HoldLoad: {
            currentLoadScale = 1.0f;
            ++stepInState;
            if (stepInState >= std::max(1, config.holdSteps)) {
                state = State::Capture;
            }
            return;
        }
        case State::Capture: {
            const int idx = std::max(0, std::min(withinAccelRunIndex, static_cast<int>(runs.size()) - 1));
            captureSnapshot(runs[static_cast<size_t>(idx)]);
            ++withinAccelRunIndex;

            // After reference snapshot (assumed second), compute RMSE for this accel.
            if (withinAccelRunIndex >= static_cast<int>(runs.size()) && snapshots.size() >= 2) {
                const Snapshot& fastSnap = snapshots[snapshots.size() - 2];
                const Snapshot& refSnap = snapshots[snapshots.size() - 1];
                if (fastSnap.positions.size() == refSnap.positions.size() && !fastSnap.positions.empty()) {
                    double sumSq = 0.0;
                    double maxErr = 0.0;
                    for (size_t i = 0; i < fastSnap.positions.size(); ++i) {
                        const double d = (fastSnap.positions[i] - refSnap.positions[i]).norm();
                        sumSq += d * d;
                        maxErr = std::max(maxErr, d);
                    }
                    const double rmse = std::sqrt(sumSq / static_cast<double>(fastSnap.positions.size()));
                    summary.push_back(SummaryRow{
                        config.pullAccel,
                        fastSnap.pbdIterations,
                        refSnap.pbdIterations,
                        fastSnap.targetDisplacement,
                        refSnap.targetDisplacement,
                        rmse,
                        maxErr,
                    });
                }
            }

            state = State::ApplyRunAndReset;
            return;
        }
        case State::SaveAndFinish: {
            saveData();
            restoreMaterial();
            if (config.resetAfterFinish) {
                resetSimulationToInitial();
            }
            std::cout << "[Experiment1] Finished.\n";
            state = State::Idle;
            return;
        }
        default:
            return;
    }
}
