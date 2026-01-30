#pragma once

#include <windows.h>
#include <iostream>
#include <vector>
#include <stdexcept>



std::vector<unsigned char> intToHexProtocol(int num);

class SerialPort {
public:
    SerialPort(const std::string& portName, int baudRate);
    ~SerialPort();

    bool isOpen() const;
    bool writeData(const unsigned char* data, DWORD size);

private:
    HANDLE hSerial;
};
