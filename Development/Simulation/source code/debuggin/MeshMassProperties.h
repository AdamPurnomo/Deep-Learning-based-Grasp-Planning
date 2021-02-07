//
//  MeshMassProperties.h
//
// Utility for computing volume, center of mass, and inertia of a closed mesh.
//
// Written by Andrew Meadows, 2015.05.24 for the public domain.  Feel free to relicense.

#ifndef MESH_MASS_PROPERTIES_H
#define MESH_MASS_PROPERTIES_H

#include <vector>

#include <btBulletDynamicsCommon.h>

typedef std::vector<btVector3> VectorOfPoints;
typedef std::vector<uint32_t> VectorOfIndices;

#define EXPOSE_HELPER_FUNCTIONS_FOR_UNIT_TEST
#ifdef EXPOSE_HELPER_FUNCTIONS_FOR_UNIT_TEST
void computeBoxInertia(btScalar mass, const btVector3& diagonal, btMatrix3x3& I);

// mass = input mass of tetrahedron
// points = input array of four points with center of mass at origin
// I = output inertia of tetrahedron about its center of mass
void computeTetrahedronInertia(btScalar mass, btVector3* points, btMatrix3x3& I);
void computeTetrahedronInertiaByBruteForce(btVector3* points, btMatrix3x3& I);

btScalar computeTetrahedronVolume(btVector3* points);

void applyParallelAxisTheorem(btMatrix3x3& inertia, const btVector3& shift, btScalar mass);
#endif // EXPOSE_HELPER_FUNCTIONS_FOR_UNIT_TEST

// Given a closed mesh with right-hand triangles a MeshMassProperties instance will compute
// its mass properties:
//
// volume
// center-of-mass
// normalized interia tensor about center of mass
//
class MeshMassProperties {
public:
    
    // the mass properties calculation is done in the constructor, so if the mesh is complex
    // then the construction could be computationally expensiuve.
    MeshMassProperties(const VectorOfPoints& points, const VectorOfIndices& triangleIndices);

    // compute the mass properties of a new mesh
    void computeMassProperties(const VectorOfPoints& points, const VectorOfIndices& triangleIndices);

    // harveste the mass properties from these public data members
    btScalar m_volume = 1.0;
    btVector3 m_centerOfMass = btVector3(0.0, 0.0, 0.0);
    btMatrix3x3 m_inertia = btMatrix3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
};

#endif // MESH_MASS_PROPERTIES_H
