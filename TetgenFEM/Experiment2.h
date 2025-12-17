#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

class Object;
class Vertex;
class Tetrahedron;

class Experiment2 {
public:
    static Experiment2& instance();

    struct Config {
        int settleSteps = 120;
        int dragSteps = 240;
        int holdSteps = 240;

        float poissonIncompressible = 0.49f;

        float dragDistanceBboxScale = 0.35f;
        float dragDistanceMin = 0.30f;
        float dragDistanceMax = 0.90f;

        float anchorSliceFrac = 0.05f;
        float pullSliceFrac = 0.05f;
        int minRegionVertexCount = 24;

        float pullStiffness = 3500.0f;
        float pullMaxAccel = 50000.0f;

        int pbdIterations = 60;
        bool resetAfterFinish = true;
    };

    void init(Object* obj, const std::vector<Vertex*>& physicalVertices);
    void requestStart();
    void update();
    void onAfterPhysics();
    void appendVertexForces(std::unordered_map<int, Eigen::Vector3f>& dragForces) const;

    bool isActive() const;
    bool wantsDrag() const;
    std::string buttonLabel() const;

    Vertex* targetVertex() const;
    Eigen::Vector3f desiredTargetPosition() const;
    const std::vector<Vertex*>& forceVertices() const;
    int pbdIterationsThisFrame(int defaultIterations) const;

private:
    Experiment2() = default;

    enum class State {
        Idle,
        PendingStart,
        ApplyRunAndReset,
        Settle,
        Drag,
        Hold,
        SaveAndFinish
    };

    struct RunSpec {
        std::string name;
        float poisson = 0.0f;
    };

    struct DataPoint {
        int runIndex = 0;
        float simTime = 0.0f;
        float imposedDisplacement = 0.0f;
        float actualDisplacement = 0.0f;
        float actualDisplacementMean = 0.0f;
        float actualDisplacementRms = 0.0f;
        float actualDisplacementMin = 0.0f;
        float actualDisplacementMax = 0.0f;
        double volume = 0.0;
        double volumeRatio = 1.0;
        std::string stage;
    };

    State state = State::Idle;
    Object* object = nullptr;
    std::vector<Vertex*> physicalVertices;
    std::vector<Vertex*> allVertices;
    std::unordered_map<int, std::vector<Vertex*>> verticesByIndex;
    std::vector<Tetrahedron*> allTetrahedra;

    Vertex* selectedTarget = nullptr;
    std::vector<int> anchorIndices;
    std::vector<int> pullIndices;
    std::unordered_map<int, Eigen::Vector3f> pullStartPosByIndex;
    std::vector<std::pair<Vertex*, bool>> fixedSnapshot;
    Config config{};
    std::vector<RunSpec> sequence;
    int runIndex = 0;
    int stepInState = 0;

    Eigen::Vector3f runStartPos = Eigen::Vector3f::Zero();
    Eigen::Vector3f currentDesiredPos = Eigen::Vector3f::Zero();
    float currentImposedDistance = 0.0f;
    float dragDistance = 0.5f;
    int lastSampleStep = 0;

    double volume0 = 0.0;
    float poissonOriginal = 0.0f;
    float oldYoungs1 = 0.0f;
    float oldYoungs2 = 0.0f;
    float oldYoungs3 = 0.0f;
    bool hasOldMaterial = false;

    std::string outputDir;
    std::vector<DataPoint> data;

    Vertex* pickDeterministicTarget() const;
    void pickRegionsDeterministic();
    void computeDragDistance();
    void resetSimulationToInitial();
    void snapshotFixedFlags();
    void restoreFixedFlags();
    void applyAnchorFixing();
    void applyMaterialForRun(const RunSpec& spec);
    void restoreMaterial();
    double computeTotalVolume() const;
    void saveData() const;
};
