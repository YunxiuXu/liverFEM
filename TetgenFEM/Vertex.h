#pragma once
#ifndef VERTEX_H
#define VERTEX_H

class Vertex {
public:
    float x, y, z;
    const float initx, inity, initz; // Initialization after declaration
    int index;  // Global index
    int localIndex; // Local index within a group
	float vertexMass; // Mass of the vertex
	float velx, vely, velz; // Velocity components
	bool isFixed;
	float lastStress = 0.0f; // Added for smooth stress visualization
	int connectedTets = 0;   // Count for vertex stress averaging

    Vertex(float x, float y, float z, int index);
    void setFixedIfBelowThreshold();

};

#endif // VERTEX_H