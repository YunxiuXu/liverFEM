#include "TetgenFEM/params.h"
#include <iostream>

int main() {
    loadParams("TetgenFEM/parameters.txt");
    
    std::cout << "Loaded parameters:" << std::endl;
    std::cout << "youngs: " << youngs << std::endl;
    std::cout << "inputFile: " << inputFile << std::endl;
    std::cout << "tetgenArgs: " << tetgenArgs << std::endl;
    std::cout << "outputFilePrefix: " << outputFilePrefix << std::endl;
    std::cout << "groupNumX: " << groupNumX << std::endl;
    
    return 0;
}