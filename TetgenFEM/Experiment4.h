#pragma once
#include <string>

class Experiment4 {
public:
    static Experiment4& instance();

    void requestStart();
    void update();

    bool isActive() const;
    std::string buttonLabel() const;

private:
    Experiment4() = default;

    enum class State {
        Idle,
        PendingStart,
        Running
    };

    State state = State::Idle;
    bool startRequested = false;

    void runBenchmarks();
};

