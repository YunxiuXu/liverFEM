#include "TetgenFEM/params.h"
#include <iostream>
loadParams("TetgenFEM/parameters_armadillo.txt");

int main() {
    loadParams("TetgenFEM/parameters.txt");
    
    std::cout << "=== Parameter Loading Test ===" << std::endl;
    std::cout << "useDirectLoading: " << (useDirectLoading ? "true" : "false") << std::endl;
    std::cout << "nodeFile: " << nodeFile << std::endl;
    std::cout << "eleFile: " << eleFile << std::endl;
    std::cout << "stlFile: " << stlFile << std::endl;
    std::cout << "tetgenArgs: " << tetgenArgs << std::endl;
    
    return 0;
}