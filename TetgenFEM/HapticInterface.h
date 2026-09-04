#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <array>
#include <chrono>

class HapticInterface {
public:
    HapticInterface();
    ~HapticInterface();

    // Initialize serial port
    bool init(const std::string& portName, int baudRate = 2000000);
    void close();

    // Send force to a specific motor.
    // force: normalized or raw force value.
    // motorId: 0-7
    void sendForce(int motorId, float force);
    void sendForce(int motorId, float force, bool bypassSlew);
    
    // Set parameters
    void setParameters(float minForceIn, float maxForceIn, float minPwmOut, float maxPwmOut, float gamma = 1.0f);

    // Slew limiter on PWM output to avoid "snap"/"pop" sensations on cable-driven 1DOF devices.
    // Rates are in PWM units per second (the same units passed to the device protocol).
    void setSlewLimiter(bool enabled, float upPwmPerSec, float downPwmPerSec);

    bool isOpen() const { return connected; }

private:
    std::string portName;
#ifdef _WIN32
    void* serialHandle = nullptr;
#else
    int fd; // file descriptor
#endif
    std::atomic<bool> connected;

    // Parameters for mapping
    float minForceInput = 0.0f;
    float maxForceInput = 10.0f;
    float minPwmOutput = 0.0f;
    float maxPwmOutput = 255.0f; 
    float gamma = 1.0f;

    // Optional output slew limiter state (per motor).
    bool slewEnabled = false;
    float slewUpPwmPerSec = 0.0f;
    float slewDownPwmPerSec = 0.0f;
    std::array<int, 8> lastPwm_{};
    std::array<bool, 8> lastPwmValid_{};
    std::array<std::chrono::steady_clock::time_point, 8> lastTime_{};
    std::array<bool, 8> lastTimeValid_{};

    // Buffer for all motors: 8 motors * 2 bytes = 16 bytes
    // Plus padding as per protocol.
    // Protocol: 0x31, val[0]..val[15], 0...
    uint8_t motorValues[16]; 
    std::mutex mutex;

    void sendPacket();
    std::vector<uint8_t> intToHexProtocol(int num);
};
