#include "serial.h"

SerialPort::SerialPort(const std::string& portName, int baudRate)
{
    hSerial = CreateFileA(portName.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

    if (hSerial == INVALID_HANDLE_VALUE) {
        if (GetLastError() == ERROR_FILE_NOT_FOUND) {
            std::cerr << "ERROR: Serial port doesn't exist.\n";
        }
        else {
            std::cerr << "ERROR: Unable to open serial port.\n";
        }
    }
    else {
        // 增加缓冲区大小
        if (!SetupComm(hSerial, 16384, 16384)) { // 4096
            std::cerr << "ERROR: Unable to setup serial port buffer size.\n";
        }

        DCB dcbSerialParams = { 0 };
        dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

        if (!GetCommState(hSerial, &dcbSerialParams)) {
            std::cerr << "ERROR: Unable to get current serial port state.\n";
        }

        dcbSerialParams.BaudRate = baudRate;
        dcbSerialParams.ByteSize = 8;
        dcbSerialParams.StopBits = ONESTOPBIT;
        dcbSerialParams.Parity = NOPARITY;

        if (!SetCommState(hSerial, &dcbSerialParams)) {
            std::cerr << "ERROR: Unable to configure serial port.\n";
        }
    }
}

SerialPort::~SerialPort()
{
    if (isOpen()) {
        CloseHandle(hSerial);
    }
}

bool SerialPort::isOpen() const
{
    return (hSerial != INVALID_HANDLE_VALUE);
}

bool SerialPort::writeData(const unsigned char* data, DWORD size)
{
    if (!isOpen()) {
        return false;
    }

    DWORD bytes_written = 0;
    return WriteFile(hSerial, data, size, &bytes_written, NULL) && (bytes_written == size);
}

std::vector<unsigned char> intToHexProtocol(int num) {
    std::vector<unsigned char> result(2);
    // Check if the number is in range -2048 to 2048
    if (num < -2048 || num > 2048) {
        //throw std::invalid_argument("The number must be in range -2048 to 2048.");
        std::cout << "The number must be in range -2048 to 2048." << std::endl;
        num = 2048;
    }

    

    if (num >= 0) {
        // If the number is non-negative, it is represented directly.
        result[0] = num & 0xFF;  // Low byte
        result[1] = (num >> 8) & 0xFF;  // High byte
    }
    else {
        // If the number is negative, it is represented in two's complement form.
        num = -num;  // Take the absolute value
        result[0] = num & 0xFF;  // Low byte
        result[1] = (num >> 8) & 0xFF;  // High byte
        result[1] = ~result[1];  // Bitwise NOT operation
        result[0] = ~result[0];  // Bitwise NOT operation
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