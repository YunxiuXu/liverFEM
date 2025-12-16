#pragma once
#include "GLFW/glfw3.h"
#include "tetgen.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Object.h"



extern Eigen::Quaternionf rotation;
extern float zoomFactor;
extern float aspectRatio;
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void applyProjectionMatrix();
Eigen::Matrix4f buildProjectionMatrix(float nearVal = -3.0f, float farVal = 3.0f);
std::string createEdgeId(Vertex* vertex1, Vertex* vertex2);
void drawEdge(Vertex* vertex1, Vertex* vertex2, float r, float g, float b);
void drawAxis(float length);
void drawAxis1(float length, const Eigen::Matrix3f& rotationMatrix);
void XPrintString(const char* s);
void initFontData();
void hsvToRgb(float h, float s, float v, float& r, float& g, float& b);
float getRotationAngleZ(const Eigen::Matrix3f& rotationmatrix);
