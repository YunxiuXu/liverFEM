#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <array>
#include <chrono>
#include <thread>

class HapticInterface {
public:
    HapticInterface();
    ~HapticInterface();

    // Initialize serial port
    bool init(const std::string& portName, int baudRate = 2000000);
    void close();

    // Publish the latest force for one motor. The simulation thread calls this;
    // a 1 kHz thread samples the latest values and writes one serial packet.
    // force: normalized or raw force value.
    // motorId: 0-7
    void sendForce(int motorId, float force);
    void sendForce(int motorId, float force, bool bypassSlew);

    // One-shot: the output thread plays this burst at 1 kHz. A burst already
    // in progress is left running.
    void requestVibration(int motorId, float freqHz, float amplitude, float durationSec);
    void cancelVibration(int motorId);

    // Force 0 and cancel any burst. Used when the simulation stops publishing.
    void clearForces();
    
    // Set parameters
    void setParameters(float minForceIn, float maxForceIn, float minPwmOut, float maxPwmOut, float gamma = 1.0f);

    // Slew limiter on PWM output to avoid "snap"/"pop" sensations on cable-driven 1DOF devices.
    // Rates are in PWM units per second (the same units passed to the device protocol).
    void setSlewLimiter(bool enabled, float upPwmPerSec, float downPwmPerSec);

    bool isOpen() const { return connected; }

private:
    std::string portName;
    int fd; // file descriptor
    std::atomic<bool> connected;

    // Parameters for mapping
    float minForceInput = 0.0f;
    float maxForceInput = 10.0f;
    float minPwmOutput = 0.0f;
    float maxPwmOutput = 255.0f; 
    float gamma = 1.0f;

    // Optional output slew limiter state (per motor). Touched only by the output thread
    // after init(), except for the reset in setSlewLimiter() before the thread starts.
    bool slewEnabled = false;
    float slewUpPwmPerSec = 0.0f;
    float slewDownPwmPerSec = 0.0f;
    std::array<int, 8> lastPwm_{};
    std::array<bool, 8> lastPwmValid_{};

    struct ForceCommand {
        float force = 0.0f;
        bool bypassSlew = false;
        bool vibStart = false;
        bool vibCancel = false;
        float vibFreqHz = 0.0f;
        float vibAmp = 0.0f;
        float vibDurationSec = 0.0f;
    };
    std::array<ForceCommand, 8> pending_{};
    std::array<float, 8> vibTimeLeft_{};
    std::array<float, 8> vibPhase_{};

    // Buffer for all motors: 8 motors * 2 bytes = 16 bytes
    // Plus padding as per protocol.
    // Protocol: 0x31, val[0]..val[15], 0...
    uint8_t motorValues[16];
    std::mutex mutex;

    std::atomic<bool> outputRunning_{false};
    std::thread outputThread_;
    std::chrono::steady_clock::time_point lastOutputTime_{};
    bool lastOutputTimeValid_ = false;
    bool writeErrorLogged_ = false;

    void startOutputThread();
    void stopOutputThread();
    void outputLoop();
    int mapForceToPwm(int motorId, float force, bool bypassSlew, float dtSec);
    void sendPacket();
    std::vector<uint8_t> intToHexProtocol(int num);
};
