#include "HapticInterface.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/ioctl.h>
#include <IOKit/serial/ioss.h>
#endif
#endif


#ifdef _WIN32
HapticInterface::HapticInterface() : serialHandle(INVALID_HANDLE_VALUE), connected(false) {
    std::memset(motorValues, 0, sizeof(motorValues));
}
#else
HapticInterface::HapticInterface() : fd(-1), connected(false) {
    std::memset(motorValues, 0, sizeof(motorValues));
}
#endif

HapticInterface::~HapticInterface() {
    close();
}

#ifdef _WIN32
static std::string winSerialDevicePath(const std::string& port) {
    if (port.rfind("\\\\.\\", 0) == 0) {
        return port;
    }
    return "\\\\.\\" + port;
}

bool HapticInterface::init(const std::string& port, int baudRate) {
    portName = port;
    const std::string device = winSerialDevicePath(port);
    HANDLE handle = CreateFileA(
        device.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        nullptr,
        OPEN_EXISTING,
        0,
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        std::cerr << "HapticInterface: Unable to open port " << port
                  << " (WinError " << GetLastError() << ")" << std::endl;
        return false;
    }

    DCB dcb{};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(handle, &dcb)) {
        std::cerr << "HapticInterface: GetCommState failed" << std::endl;
        CloseHandle(handle);
        return false;
    }
    dcb.BaudRate = static_cast<DWORD>(baudRate);
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    if (!SetCommState(handle, &dcb)) {
        std::cerr << "HapticInterface: SetCommState failed for baud " << baudRate
                  << " (WinError " << GetLastError() << ")" << std::endl;
        CloseHandle(handle);
        return false;
    }

    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    SetCommTimeouts(handle, &timeouts);

    serialHandle = handle;
    connected = true;
    std::cout << "HapticInterface: Connected to " << port << std::endl;
    return true;
}

void HapticInterface::close() {
    HANDLE handle = static_cast<HANDLE>(serialHandle);
    if (connected && handle != nullptr && handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
        serialHandle = INVALID_HANDLE_VALUE;
        connected = false;
    }
}
#else
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
        }
    }
#endif

    // Blocking read with timeout (optional, we mainly write)
    fcntl(fd, F_SETFL, 0);

    connected = true;
    std::cout << "HapticInterface: Connected to " << port << std::endl;
    return true;
}

void HapticInterface::close() {
    if (connected && fd != -1) {
        ::close(fd);
        fd = -1;
        connected = false;
    }
}
#endif

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
    lastTimeValid_.fill(false);
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
    if (!connected || motorId < 0 || motorId > 7) return;

    const auto now = std::chrono::steady_clock::now();

    // Important: if the requested force is zero (or below the input minimum),
    // output a true "off" command (0 PWM), regardless of configured minPwmOutput.
    // This prevents a lingering baseline force after the fingertip leaves contact.
    if (force <= minForceInput + 1e-6f || maxForceInput <= minForceInput + 1e-6f) {
        if (slewEnabled) {
            lastPwm_[motorId] = 0;
            lastPwmValid_[motorId] = true;
            lastTime_[motorId] = now;
            lastTimeValid_[motorId] = true;
        }
        std::vector<uint8_t> bytes = intToHexProtocol(0);
        {
            std::lock_guard<std::mutex> lock(mutex);
            motorValues[motorId * 2] = bytes[0];
            motorValues[motorId * 2 + 1] = bytes[1];
        }
        sendPacket();
        return;
    }

    // Map force to output range
    // Clamp input
    float f = std::max(minForceInput, std::min(force, maxForceInput));
    
    // Normalize 0-1
    float t = (f - minForceInput) / (maxForceInput - minForceInput);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    // Apply Gamma Correction (Weber's Law / Psychophysics)
    // t_out = t_in ^ gamma
    // gamma < 1.0 boosts small forces (makes them more noticeable)
    // gamma > 1.0 suppresses small forces
    if (std::abs(gamma - 1.0f) > 0.001f) {
        t = std::pow(t, gamma);
    }
    
    // Map to output
    float outVal = minPwmOutput + t * (maxPwmOutput - minPwmOutput);
    int intVal = static_cast<int>(outVal);

    if (slewEnabled && !bypassSlew) {
        float dt = 0.0f;
        if (lastTimeValid_[motorId]) {
            dt = std::chrono::duration<float>(now - lastTime_[motorId]).count();
        }
        lastTime_[motorId] = now;
        lastTimeValid_[motorId] = true;

        const int prev = lastPwmValid_[motorId] ? lastPwm_[motorId] : intVal;
        const float maxUp = slewUpPwmPerSec * dt;
        const float maxDown = slewDownPwmPerSec * dt;
        int limited = intVal;
        if (limited > prev) {
            const int cap = prev + static_cast<int>(std::ceil(maxUp));
            if (maxUp > 0.0f) limited = std::min(limited, cap);
        } else if (limited < prev) {
            const int cap = prev - static_cast<int>(std::ceil(maxDown));
            if (maxDown > 0.0f) limited = std::max(limited, cap);
        }
        intVal = limited;
        lastPwm_[motorId] = intVal;
        lastPwmValid_[motorId] = true;
    } else if (slewEnabled) {
        // Keep state consistent even when bypassing.
        lastTime_[motorId] = now;
        lastTimeValid_[motorId] = true;
        lastPwm_[motorId] = intVal;
        lastPwmValid_[motorId] = true;
    }

    // Convert to protocol bytes
    std::vector<uint8_t> bytes = intToHexProtocol(intVal);

    {
        std::lock_guard<std::mutex> lock(mutex);
        motorValues[motorId * 2] = bytes[0];     // Low
        motorValues[motorId * 2 + 1] = bytes[1]; // High
    }

    sendPacket();
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

#ifdef _WIN32
    HANDLE handle = static_cast<HANDLE>(serialHandle);
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    if (!WriteFile(handle, buffer, static_cast<DWORD>(BUFFER_SIZE), &written, nullptr)) {
        std::cerr << "HapticInterface: Write error (WinError " << GetLastError() << ")" << std::endl;
    }
#else
    ssize_t written = write(fd, buffer, BUFFER_SIZE);
    if (written < 0) {
        std::cerr << "HapticInterface: Write error" << std::endl;
    }
#endif
}
