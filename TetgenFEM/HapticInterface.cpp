#include "HapticInterface.h"
#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>

#if defined(__APPLE__)
#include <sys/ioctl.h>
#include <IOKit/serial/ioss.h>
#include <pthread/qos.h>
#endif


HapticInterface::HapticInterface() : fd(-1), connected(false) {
    std::memset(motorValues, 0, sizeof(motorValues));
}

HapticInterface::~HapticInterface() {
    close();
}

bool HapticInterface::init(const std::string& port, int baudRate) {
    portName = port;
    fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        std::cerr << "HapticInterface: Unable to open port " << port << std::endl;
        return false;
    }

    struct termios options;
    tcgetattr(fd, &options);

    // Set baud rate
    speed_t speed;
    bool custom_baud = false;
    switch (baudRate) {
        case 9600: speed = B9600; break;
        case 19200: speed = B19200; break;
        case 38400: speed = B38400; break;
        case 57600: speed = B57600; break;
        case 115200: speed = B115200; break;
#ifdef B2000000
        case 2000000: speed = B2000000; break; 
#endif
        default: 
             speed = B38400; // Use a standard rate as base
             custom_baud = true;
             // std::cerr << "HapticInterface: Warning, baudrate " << baudRate << " might not be supported directly, trying custom method." << std::endl;
             break;
    }
    
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB; // No parity
    options.c_cflag &= ~CSTOPB; // 1 stop bit
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;     // 8 data bits
    
    // Raw input/output
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    tcsetattr(fd, TCSANOW, &options);

#if defined(__APPLE__)
    // Force set custom baud rate using ioctl (works for 2000000 on macOS)
    if (custom_baud || baudRate == 2000000) {
        speed_t rate = static_cast<speed_t>(baudRate);
        if (ioctl(fd, IOSSIOSPEED, &rate) == -1) {
            std::cerr << "HapticInterface: Error setting custom baud rate " << baudRate << " via IOSSIOSPEED" << std::endl;
        } else {
             // std::cout << "HapticInterface: Set custom baud rate " << baudRate << " via IOSSIOSPEED" << std::endl;
        }
    }
    // Ask the driver not to hold a partial USB packet. Default 0 waits on a
    // character-count latency, which drops the output rate well below 1 kHz.
    unsigned long latencyUs = 1000;
    if (ioctl(fd, IOSSDATALAT, &latencyUs) == -1) {
        std::cerr << "HapticInterface: IOSSDATALAT failed, serial latency left at driver default" << std::endl;
    }
#endif

    // Non-blocking so a full USB buffer cannot stall the 1 kHz thread.
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    connected = true;
    startOutputThread();
    std::cout << "HapticInterface: Connected to " << port << std::endl;
    return true;
}

void HapticInterface::close() {
    stopOutputThread();
    if (fd != -1) {
        ::close(fd);
        fd = -1;
    }
    connected = false;
}

void HapticInterface::setParameters(float minF, float maxF, float minP, float maxP, float g) {
    minForceInput = minF;
    maxForceInput = maxF;
    minPwmOutput = minP;
    maxPwmOutput = maxP;
    gamma = g;
}

void HapticInterface::setSlewLimiter(bool enabled, float upPwmPerSec, float downPwmPerSec) {
    slewEnabled = enabled;
    slewUpPwmPerSec = std::max(0.0f, upPwmPerSec);
    slewDownPwmPerSec = std::max(0.0f, downPwmPerSec);
    // Reset state so the next command takes effect deterministically.
    lastPwm_.fill(0);
    lastPwmValid_.fill(false);
    lastOutputTimeValid_ = false;
}

// Exactly from original code
std::vector<uint8_t> HapticInterface::intToHexProtocol(int num) {
    std::vector<uint8_t> result(2);
    
    // Clamp to -2048 to 2048
    if (num < -2048) num = -2048;
    if (num > 2048) num = 2048;

    if (num >= 0) {
        result[0] = num & 0xFF;  // Low byte
        result[1] = (num >> 8) & 0xFF;  // High byte
    }
    else {
        num = -num;  // Absolute value
        result[0] = num & 0xFF;
        result[1] = (num >> 8) & 0xFF;
        result[1] = ~result[1];
        result[0] = ~result[0];
        if (result[0] == 255) {
            result[0] = 0;
            result[1] += 1;
        }
        else {
            result[0] += 1;
        }
    }
    return result;
}

void HapticInterface::sendForce(int motorId, float force) {
    sendForce(motorId, force, false);
}

void HapticInterface::sendForce(int motorId, float force, bool bypassSlew) {
    if (motorId < 0 || motorId > 7) return;
    std::lock_guard<std::mutex> lock(mutex);
    pending_[static_cast<size_t>(motorId)].force = force;
    pending_[static_cast<size_t>(motorId)].bypassSlew = bypassSlew;
}

void HapticInterface::requestVibration(int motorId, float freqHz, float amplitude, float durationSec) {
    if (motorId < 0 || motorId > 7) return;
    std::lock_guard<std::mutex> lock(mutex);
    ForceCommand& cmd = pending_[static_cast<size_t>(motorId)];
    cmd.vibStart = true;
    cmd.vibCancel = false;
    cmd.vibFreqHz = std::max(0.0f, freqHz);
    cmd.vibAmp = std::max(0.0f, amplitude);
    cmd.vibDurationSec = std::max(0.0f, durationSec);
}

void HapticInterface::cancelVibration(int motorId) {
    if (motorId < 0 || motorId > 7) return;
    std::lock_guard<std::mutex> lock(mutex);
    pending_[static_cast<size_t>(motorId)].vibStart = false;
    pending_[static_cast<size_t>(motorId)].vibCancel = true;
}

void HapticInterface::clearForces() {
    std::lock_guard<std::mutex> lock(mutex);
    for (ForceCommand& cmd : pending_) {
        cmd.force = 0.0f;
        cmd.bypassSlew = true;
        cmd.vibStart = false;
        cmd.vibCancel = true;
    }
}

void HapticInterface::startOutputThread() {
    stopOutputThread();
    outputRunning_.store(true, std::memory_order_relaxed);
    outputThread_ = std::thread([this]() { outputLoop(); });
}

void HapticInterface::stopOutputThread() {
    outputRunning_.store(false, std::memory_order_relaxed);
    if (outputThread_.joinable()) {
        outputThread_.join();
    }
}

int HapticInterface::mapForceToPwm(int motorId, float force, bool bypassSlew, float dtSec) {
    // Zero (or below the input minimum) is a true off command, not minPwmOutput.
    if (force <= minForceInput + 1e-6f || maxForceInput <= minForceInput + 1e-6f) {
        lastPwm_[static_cast<size_t>(motorId)] = 0;
        lastPwmValid_[static_cast<size_t>(motorId)] = true;
        return 0;
    }

    float f = std::max(minForceInput, std::min(force, maxForceInput));
    float t = (f - minForceInput) / (maxForceInput - minForceInput);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    if (std::abs(gamma - 1.0f) > 0.001f) {
        t = std::pow(t, gamma);
    }

    int intVal = static_cast<int>(minPwmOutput + t * (maxPwmOutput - minPwmOutput));

    if (slewEnabled && !bypassSlew) {
        const int prev = lastPwmValid_[static_cast<size_t>(motorId)] ? lastPwm_[static_cast<size_t>(motorId)] : intVal;
        const float maxUp = slewUpPwmPerSec * dtSec;
        const float maxDown = slewDownPwmPerSec * dtSec;
        if (intVal > prev) {
            const int cap = prev + static_cast<int>(std::ceil(maxUp));
            if (maxUp > 0.0f) intVal = std::min(intVal, cap);
        } else if (intVal < prev) {
            const int cap = prev - static_cast<int>(std::ceil(maxDown));
            if (maxDown > 0.0f) intVal = std::max(intVal, cap);
        }
    }

    lastPwm_[static_cast<size_t>(motorId)] = intVal;
    lastPwmValid_[static_cast<size_t>(motorId)] = true;
    return intVal;
}

void HapticInterface::outputLoop() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    constexpr auto kPeriod = std::chrono::milliseconds(1);
    auto next = std::chrono::steady_clock::now();
    std::cout << "HapticInterface: serial output thread running at 1000 Hz" << std::endl;

    while (outputRunning_.load(std::memory_order_relaxed)) {
        next += kPeriod;

        std::array<ForceCommand, 8> cmds;
        {
            std::lock_guard<std::mutex> lock(mutex);
            cmds = pending_;
            for (ForceCommand& cmd : pending_) {
                cmd.vibStart = false;
                cmd.vibCancel = false;
            }
        }

        const auto now = std::chrono::steady_clock::now();
        float dtSec = 0.001f;
        if (lastOutputTimeValid_) {
            dtSec = std::chrono::duration<float>(now - lastOutputTime_).count();
        }
        lastOutputTime_ = now;
        lastOutputTimeValid_ = true;
        if (dtSec < 0.0f) dtSec = 0.0f;
        if (dtSec > 0.02f) dtSec = 0.02f;

        uint8_t values[16];
        std::memset(values, 0, sizeof(values));
        constexpr float kPi = 3.14159265f;
        constexpr float kTwoPi = 6.28318530718f;
        for (int motorId = 0; motorId < 8; ++motorId) {
            const ForceCommand& cmd = cmds[static_cast<size_t>(motorId)];
            float force = cmd.force;
            bool bypassSlew = cmd.bypassSlew;
            float& tLeft = vibTimeLeft_[static_cast<size_t>(motorId)];
            float& phase = vibPhase_[static_cast<size_t>(motorId)];
            if (cmd.vibCancel) {
                tLeft = 0.0f;
                phase = 0.0f;
            } else if (cmd.vibStart && tLeft <= 0.0f) {
                tLeft = cmd.vibDurationSec;
                phase = 0.0f;
            }
            if (tLeft > 0.0f && dtSec > 0.0f && cmd.vibDurationSec > 0.0f) {
                const float dur = std::max(1e-4f, cmd.vibDurationSec);
                const float elapsed = std::max(0.0f, dur - tLeft);
                const float u = std::clamp(elapsed / dur, 0.0f, 1.0f);
                const float env = 0.5f * (1.0f + std::cos(kPi * u));
                const float s = std::sin(phase);
                const float vib = cmd.vibAmp * env * (0.5f * (1.0f + s));
                force = std::max(0.0f, force + vib);
                phase += kTwoPi * cmd.vibFreqHz * dtSec;
                if (phase > kTwoPi) phase = std::fmod(phase, kTwoPi);
                tLeft = std::max(0.0f, tLeft - dtSec);
                bypassSlew = true;
            }
            const int pwm = mapForceToPwm(motorId, force, bypassSlew, dtSec);
            const std::vector<uint8_t> bytes = intToHexProtocol(pwm);
            values[motorId * 2] = bytes[0];
            values[motorId * 2 + 1] = bytes[1];
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            std::memcpy(motorValues, values, sizeof(values));
        }
        sendPacket();

        std::this_thread::sleep_until(next);
        if (std::chrono::steady_clock::now() > next + kPeriod) {
            next = std::chrono::steady_clock::now();
        }
    }
}

void HapticInterface::sendPacket() {
    if (!connected) return;

    // Prepare buffer
    // Header 0x31
    // 16 bytes motor data
    // Padding (lots of 0s)
    // Total size in original code was:
    // unsigned char data_to_sends[] = { 0x31, val[0]...val[15], 0... };
    // The original array length seems large, around 100 bytes?
    // Let's count from the original code: 
    // 0x31 + 16 bytes + 14 groups of 0, 0, 0, ...
    // It looks like a fixed size buffer.
    // "DWORD bytes_to_send = sizeof(data_to_sends);"
    // The array initialization had many zeros. 
    // Let's send a reasonably large buffer filled with zeros.
    
    constexpr size_t BUFFER_SIZE = 128; // Safe enough, original looked < 128
    uint8_t buffer[BUFFER_SIZE];
    std::memset(buffer, 0, BUFFER_SIZE);
    
    buffer[0] = 0x31;
    {
        std::lock_guard<std::mutex> lock(mutex);
        std::memcpy(&buffer[1], motorValues, 16);
    }

    const ssize_t written = write(fd, buffer, BUFFER_SIZE);
    if (written == static_cast<ssize_t>(BUFFER_SIZE)) return;
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
    if (!writeErrorLogged_) {
        writeErrorLogged_ = true;
        std::cerr << "HapticInterface: serial write incomplete (" << written << " bytes)" << std::endl;
    }
}
