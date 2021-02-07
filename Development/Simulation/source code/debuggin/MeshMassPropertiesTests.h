//
// MeshMassPropertiesTests.h
//
// Unit tests for MeshMassProperties class which computes the volume, center of mass,
// and inertia tensor for a closed mesh of arbitrary shape.
//
// Written by Andrew Meadows, 2015.05.24 for the public domain.  Feel free to relicense.
//

#ifndef MESH_MASS_PROPERTIES_H
#define MESH_MASS_PROPERTIES_H
namespace MeshInfoTests{
    void testParallelAxisTheorem();
    void testTetrahedron();
    void testOpenTetrahedonMesh();
	void testClosedTetrahedronMesh();
    void testBoxAsMesh();
    void runAllTests();
}
#endif // MESH_MASS_PROPERTIES_H
