#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

class Object;
class Vertex;

class Experiment3 {
public:
    static Experiment3& instance();

    struct Config {
        // Deterministic, step-based timings (independent of wall-clock FPS).
        int settleSteps = 120; // ~1.2s if timeStep=0.01
        int dragSteps = 240;   // ~2.4s if timeStep=0.01
        float exOverEy = 5.0f; // Ex = exOverEy * Ey
        // For Experiment 3, reducing nu weakens coupling and makes anisotropy response clearer.
        bool overridePoisson = true;
        float poissonOverride = 0.08f;
        // Drag distance is chosen automatically from model bbox and clamped by dragMaxDisplacement.
        float dragDistanceBboxScale = 0.15f; // fraction of bbox diagonal
        float dragDistanceMin = 0.15f;
        float dragDistanceMax = 0.8f; // additional clamp besides dragMaxDisplacement
        bool resetAfterFinish = true; // return to normal demo state after saving
    };

    void init(Object* obj, const std::vector<Vertex*>& uniqueVertices);
    void setConfig(const Config& cfg);

    // UI-triggered (one-click) start; actual start happens on next update() to keep frame order stable.
    void requestStart();

    // Called once per frame (before computing drag forces).
    void update();

    // State queries for main loop.
    bool isActive() const;
    bool wantsDrag() const;
    std::string buttonLabel() const;

    // Getters for physics/drag driver.
    Vertex* targetVertex() const;
    Eigen::Vector3f desiredTargetPosition() const;
    const std::vector<Vertex*>& forceVertices() const;

    // Input from drag driver (computed this frame).
    void onDragForces(float totalForceMagnitude, float targetForceMagnitude);

private:
    Experiment3() = default;
    
    enum class State {
        Idle,
        PendingStart,
        ApplyMaterialAndReset,
        Settle,
        Drag,
        SaveAndFinish
    };

    State state = State::Idle;
    Object* object = nullptr;
    std::vector<Vertex*> uniqueVertices;
    std::vector<Vertex*> allVertices;

    Vertex* selectedTarget = nullptr;
    
    struct RunSpec {
        enum class Material { Isotropic, Anisotropic } material;
        int dragAxis; // 0:X, 1:Y, 2:Z
        int hardAxis; // for anisotropic only: 0:X, 1:Y, 2:Z
    };

    Config config{};
    std::vector<RunSpec> sequence;
    int runIndex = 0;
    int stepInState = 0;
    int lastDragStep = 0;
    bool advanceToNextRunPending = false;

    // Per-run reference data.
    Eigen::Vector3f runStartPos = Eigen::Vector3f::Zero();
    Eigen::Vector3f currentDesiredPos = Eigen::Vector3f::Zero();
    float dragDistance = 0.5f;

    // Saved original material params (restore after experiment).
    float oldYoungs1 = 0.0f;
    float oldYoungs2 = 0.0f;
    float oldYoungs3 = 0.0f;
    bool hasOldMaterial = false;

    float oldPoisson = 0.0f;
    bool hasOldPoisson = false;

    std::string outputDir;

    struct DataPoint {
        int runIndex;
        float simTime;
        float imposedDisplacement;
        float actualDisplacement;
        float totalForce;
        float targetForce;
    };
    
    std::vector<DataPoint> data;

    Vertex* pickDeterministicTarget() const;
    void computeDragDistance();
    void resetSimulationToInitial();
    void applyMaterial(const RunSpec& spec);
    void restoreMaterial();
    void saveData();
};
