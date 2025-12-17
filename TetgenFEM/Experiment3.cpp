#include "Experiment3.h"
#include "Object.h"
#include "Vertex.h"
#include "params.h" 
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

Experiment3& Experiment3::instance() {
    static Experiment3 instance;
    return instance;
}

void Experiment3::init(Object* obj, const std::vector<Vertex*>& vertices) {
    object = obj;
    uniqueVertices = vertices;
    sequence = {
        { RunSpec::Material::Isotropic, 0 },
        { RunSpec::Material::Isotropic, 1 },
        { RunSpec::Material::Anisotropic, 0 },
        { RunSpec::Material::Anisotropic, 1 },
    };
}

void Experiment3::setConfig(const Config& cfg) {
    config = cfg;
}

void Experiment3::requestStart() {
    if (!object) {
        std::cerr << "[Experiment3] Cannot start: Object not initialized.\n";
        return;
    }
    if (state != State::Idle) {
        return;
    }
    state = State::PendingStart;
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

Vertex* Experiment3::pickDeterministicTarget() const {
    if (uniqueVertices.empty()) return nullptr;

    Eigen::Vector3f minBound(std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
    Eigen::Vector3f maxBound(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
    for (const Vertex* v : uniqueVertices) {
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

    // Prefer front-facing +X extreme among non-fixed vertices.
    for (Vertex* v : uniqueVertices) {
        if (v->initz >= frontThresholdZ) {
            consider(v);
        }
    }
    if (best) return best;

    for (Vertex* v : uniqueVertices) {
        consider(v);
    }
    return best;
}

void Experiment3::computeDragDistance() {
    if (uniqueVertices.empty()) {
        dragDistance = std::min(config.dragDistanceMax, std::max(config.dragDistanceMin, 0.5f));
        return;
    }
    Eigen::Vector3f minBound(std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
    Eigen::Vector3f maxBound(-std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());
    for (const Vertex* v : uniqueVertices) {
        minBound.x() = std::min(minBound.x(), v->initx);
        minBound.y() = std::min(minBound.y(), v->inity);
        minBound.z() = std::min(minBound.z(), v->initz);
        maxBound.x() = std::max(maxBound.x(), v->initx);
        maxBound.y() = std::max(maxBound.y(), v->inity);
        maxBound.z() = std::max(maxBound.z(), v->initz);
    }

    const float diag = (maxBound - minBound).norm();
    float dist = std::max(config.dragDistanceMin, diag * config.dragDistanceBboxScale);
    dist = std::min(dist, config.dragDistanceMax);
    dist = std::min(dist, std::max(0.0f, dragMaxDisplacement) * 0.8f);
    dragDistance = std::max(config.dragDistanceMin, dist);
}

void Experiment3::resetSimulationToInitial() {
    if (!object) return;

    std::unordered_set<Vertex*> visited;
    visited.reserve(uniqueVertices.size() * 2);
    for (Vertex* v : uniqueVertices) {
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

void Experiment3::applyMaterial(const RunSpec& spec) {
    if (!object) return;

    const float baseE = youngs;
    if (spec.material == RunSpec::Material::Isotropic) {
        youngs1 = baseE;
        youngs2 = baseE;
        youngs3 = baseE;
    } else {
        youngs1 = config.exOverEy * baseE;
        youngs2 = baseE;
        youngs3 = baseE;
    }

#pragma omp parallel for
    for (int i = 0; i < object->groupNum; ++i) {
        if (spec.material == RunSpec::Material::Isotropic) {
            object->groups[i].calGroupK(youngs, poisson);
        } else {
            object->groups[i].calGroupKAni(youngs1, youngs2, youngs3, poisson);
        }
        object->groups[i].calLHS();
    }
}

void Experiment3::restoreMaterial() {
    if (!object) return;
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
}

void Experiment3::update() {
    if (!object) return;

    switch (state) {
        case State::Idle:
            return;
        case State::PendingStart: {
            selectedTarget = pickDeterministicTarget();
            if (!selectedTarget) {
                std::cerr << "[Experiment3] Cannot start: no selectable target vertex.\n";
                state = State::Idle;
                return;
            }

            oldYoungs1 = youngs1;
            oldYoungs2 = youngs2;
            oldYoungs3 = youngs3;

            computeDragDistance();

            outputDir = (std::filesystem::path("..") / "out" / "experiment3" / nowTimestamp()).string();
            data.clear();
            runIndex = 0;
            stepInState = 0;

            std::cout << "[Experiment3] Start (one-click). Vertex=" << selectedTarget->index
                      << ", dragDistance=" << dragDistance
                      << ", steps(settle/drag)=" << config.settleSteps << "/" << config.dragSteps
                      << ", sequenceRuns=" << sequence.size() << "\n";

            state = State::ApplyMaterialAndReset;
            return;
        }
        case State::ApplyMaterialAndReset: {
            if (runIndex >= static_cast<int>(sequence.size())) {
                state = State::SaveAndFinish;
                return;
            }
            applyMaterial(sequence[runIndex]);
            resetSimulationToInitial();
            currentDesiredPos = Eigen::Vector3f(selectedTarget->x, selectedTarget->y, selectedTarget->z);
            stepInState = 0;
            advanceToNextRunPending = false;
            state = State::Settle;
            return;
        }
        case State::Settle: {
            // No drag during settle; just let physics reach a consistent baseline.
            currentDesiredPos = Eigen::Vector3f(selectedTarget->x, selectedTarget->y, selectedTarget->z);
            ++stepInState;
            if (stepInState >= std::max(1, config.settleSteps)) {
                runStartPos = Eigen::Vector3f(selectedTarget->x, selectedTarget->y, selectedTarget->z);
                stepInState = 0;
                state = State::Drag;
            }
            return;
        }
        case State::Drag: {
            if (advanceToNextRunPending) {
                advanceToNextRunPending = false;
                ++runIndex;
                state = State::ApplyMaterialAndReset;
                return;
            }

            const int steps = std::max(1, config.dragSteps);
            const float denom = static_cast<float>(std::max(1, steps - 1));
            const float t = std::min(1.0f, static_cast<float>(stepInState) / denom);
            Eigen::Vector3f axis = Eigen::Vector3f::Zero();
            axis[sequence[runIndex].axis] = 1.0f;
            currentDesiredPos = runStartPos + axis * (dragDistance * t);

            lastDragStep = stepInState;
            ++stepInState;
            if (stepInState >= steps) {
                advanceToNextRunPending = true;
            }
            return;
        }
        case State::SaveAndFinish: {
            saveData();
            restoreMaterial();
            std::cout << "[Experiment3] Finished.\n";
            state = State::Idle;
            return;
        }
        default:
            return;
    }
}

bool Experiment3::isActive() const {
    return state != State::Idle;
}

bool Experiment3::wantsDrag() const {
    return state == State::Drag;
}

std::string Experiment3::buttonLabel() const {
    if (state == State::Idle) return "START EXP3";
    if (state == State::SaveAndFinish) return "SAVE EXP3";
    return "EXP3";
}

Vertex* Experiment3::targetVertex() const {
    return selectedTarget;
}

Eigen::Vector3f Experiment3::desiredTargetPosition() const {
    return currentDesiredPos;
}

void Experiment3::saveData() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(outputDir, ec);
    if (ec) {
        std::cerr << "[Experiment3] Failed to create output dir: " << outputDir << " (" << ec.message() << ")\n";
        return;
    }

    const fs::path csvPath = fs::path(outputDir) / "experiment3_force_displacement.csv";
    std::ofstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "[Experiment3] Failed to open " << csvPath.string() << " for writing\n";
        return;
    }
    file << "run_index,material,axis,sim_time,imposed_displacement,actual_displacement,force_total,force_target,vertex_index\n";

    for (const auto& p : data) {
        const RunSpec& spec = sequence[static_cast<size_t>(p.runIndex)];
        const char* material = (spec.material == RunSpec::Material::Isotropic) ? "isotropic" : "anisotropic";
        const char* axis = (spec.axis == 0) ? "x" : "y";
        file << p.runIndex << ","
             << material << ","
             << axis << ","
             << p.simTime << ","
             << p.imposedDisplacement << ","
             << p.actualDisplacement << ","
             << p.totalForce << ","
             << p.targetForce << ","
             << (selectedTarget ? selectedTarget->index : -1)
             << "\n";
    }
    file.close();

    const fs::path metaPath = fs::path(outputDir) / "experiment3_metadata.txt";
    std::ofstream meta(metaPath);
    if (meta.is_open()) {
        meta << "vertex_index=" << (selectedTarget ? selectedTarget->index : -1) << "\n";
        if (selectedTarget) {
            meta << "vertex_init=(" << selectedTarget->initx << "," << selectedTarget->inity << "," << selectedTarget->initz << ")\n";
        }
        meta << "timeStep=" << timeStep << "\n";
        meta << "youngs=" << youngs << "\n";
        meta << "poisson=" << poisson << "\n";
        meta << "dragInfluenceRadius=" << dragInfluenceRadius << "\n";
        meta << "dragStiffness=" << dragStiffness << "\n";
        meta << "dragMaxAccel=" << dragMaxAccel << "\n";
        meta << "dragMaxDisplacement=" << dragMaxDisplacement << "\n";
        meta << "exOverEy=" << config.exOverEy << "\n";
        meta << "dragDistance=" << dragDistance << "\n";
        meta << "settleSteps=" << config.settleSteps << "\n";
        meta << "dragSteps=" << config.dragSteps << "\n";
        meta << "sequence=";
        for (size_t i = 0; i < sequence.size(); ++i) {
            const auto& spec = sequence[i];
            meta << (i ? ";" : "");
            meta << ((spec.material == RunSpec::Material::Isotropic) ? "iso" : "ani") << ":" << ((spec.axis == 0) ? "x" : "y");
        }
        meta << "\n";
        meta.close();
    }

    std::cout << "[Experiment3] Saved " << data.size() << " samples to " << csvPath.string() << "\n";
}

void Experiment3::onDragForces(float totalForceMagnitude, float targetForceMagnitude) {
    if (state != State::Drag) return;
    if (!selectedTarget) return;

    const float simTime = static_cast<float>(lastDragStep) * timeStep;
    const float imposed = (currentDesiredPos - runStartPos).norm();
    const Eigen::Vector3f currentPos(selectedTarget->x, selectedTarget->y, selectedTarget->z);
    const float actual = (currentPos - runStartPos).norm();

    data.push_back(DataPoint{
        runIndex,
        simTime,
        imposed,
        actual,
        totalForceMagnitude,
        targetForceMagnitude,
    });
}
