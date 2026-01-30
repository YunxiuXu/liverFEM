#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <atomic>

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
    
    // Set parameters
    void setParameters(float minForceIn, float maxForceIn, float minPwmOut, float maxPwmOut, float gamma = 1.0f);

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

    // Buffer for all motors: 8 motors * 2 bytes = 16 bytes
    // Plus padding as per protocol.
    // Protocol: 0x31, val[0]..val[15], 0...
    uint8_t motorValues[16]; 
    std::mutex mutex;

    void sendPacket();
    std::vector<uint8_t> intToHexProtocol(int num);
};
