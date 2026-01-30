#include <iostream>
#include <cmath>
#include <mutex>
#include "signalGenerator.h"


//std::vector<HapticFunctionCall> functionCalls;//所有的震动事件写在这里
std::mutex mtx;
std::vector<std::vector<float>> functionPoolVector;

double calculateExponential(double base, double exponent, double t) {
    double result = std::pow(base, exponent * t);
    return result;
}

double calculateSin(double amplitude, double frequency, double phaseShift, double t) {
    double radians = 2.0 * 3.1415926 * frequency * t + phaseShift;
    return amplitude * std::sin(radians);
}

std::vector<float> basicCollision(float t0, float L, float B, float freq, float t) {
    float lifeTime = 0.2f;
    t = t - t0;
    float result = L * std::exp(-B * t) * std::sin(2 * 3.14159 * freq * t);
    std::vector<float> vec;
    vec.push_back(result);
    if (t > lifeTime) // if life over
        vec.push_back(0);
    else
        vec.push_back(1);
    return vec;
}


//void addFunctionCall(const std::function<double(double, const std::tuple<float, double, double, double>&)>& function, int t0, double a, double b, double c) {
//    std::lock_guard<std::mutex> lock(mtx);  // 在这个作用域中锁定互斥量
//    functionCalls.push_back(HapticFunctionCall{ function, std::make_tuple(t0, a, b, c) });
//}  // 退出作用域时，析构函数自动解锁互斥量
//
