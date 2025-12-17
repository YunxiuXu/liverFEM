#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

class Object;
class Vertex;

class Experiment1 {
public:
    static Experiment1& instance();

    struct Config {
        // Deterministic, step-based timings (independent of wall-clock FPS).
        int settleSteps = 120;
        int loadRampSteps = 240;
        int holdSteps = 240;

        // Constant load (force) applied on a region near a deterministic boundary vertex.
        float pullAccel = 800.0f;
        float influenceRadius = 0.6f;
        Eigen::Vector3f pullDirection = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

        // Per-run solver iterations (controls accuracy/cost).
        int pbdIterationsFast = 10;
        int pbdIterationsReference = 60;

        bool resetAfterFinish = true;
    };

    void init(Object* obj, const std::vector<Vertex*>& physicalVertices);
    void requestStart();
    void update();

    bool isActive() const;
    std::string buttonLabel() const;

    Vertex* targetVertex() const;
    int pbdIterationsThisFrame(int defaultIterations) const;
    void appendVertexForces(std::unordered_map<int, Eigen::Vector3f>& forcesOut) const;

private:
    Experiment1() = default;

    enum class State {
        Idle,
        PendingStart,
        ApplyRunAndReset,
        Settle,
        LoadRamp,
        HoldLoad,
        Capture,
        SaveAndFinish
    };

    struct RunSpec {
        std::string name;
        int pbdIterations = 10;
    };

    struct AccelSpec {
        float pullAccel = 0.0f;
    };

    State state = State::Idle;
    Object* object = nullptr;
    std::vector<Vertex*> physicalVertices;
    std::vector<Vertex*> allVertices;
    Vertex* selectedTarget = nullptr;

    Config config{};
    std::vector<RunSpec> runs;
    std::vector<AccelSpec> accelSweep;
    int accelIndex = 0;
    int withinAccelRunIndex = 0; // 0:fast, 1:reference
    int stepInState = 0;

    // Drag control
    Eigen::Vector3f targetInitPos = Eigen::Vector3f::Zero();
    std::vector<Vertex*> forceRegionVertices;
    float currentLoadScale = 0.0f;

    // Saved original material params (restore after experiment).
    float oldYoungs1 = 0.0f;
    float oldYoungs2 = 0.0f;
    float oldYoungs3 = 0.0f;
    float oldPoisson = 0.0f;
    bool hasOldMaterial = false;
    bool hasOldPoisson = false;

    struct Snapshot {
        std::string runName;
        float pullAccel = 0.0f;
        int pbdIterations = 0;
        std::vector<Eigen::Vector3f> positions;
        float targetDisplacement = 0.0f;
    };
    std::vector<Snapshot> snapshots;

    struct SummaryRow {
        float pullAccel = 0.0f;
        int pbdFast = 0;
        int pbdRef = 0;
        float targetDispFast = 0.0f;
        float targetDispRef = 0.0f;
        double rmse = 0.0;
        double maxErr = 0.0;
    };
    std::vector<SummaryRow> summary;

    std::string outputDir;

    Vertex* pickDeterministicTarget() const;
    void resetSimulationToInitial();
    void applyIsotropicMaterial();
    void restoreMaterial();
    void captureSnapshot(const RunSpec& spec);
    void saveData() const;
};
