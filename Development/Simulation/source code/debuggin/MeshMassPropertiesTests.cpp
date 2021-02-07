//
// MeshMassPropertiesTests.h
//
// Unit tests for MeshMassProperties class which computes the volume, center of mass,
// and inertia tensor for a closed mesh of arbitrary shape.
//
// Written by Andrew Meadows, 2015.05.24 for the public domain.  Feel free to relicense.
//

#include <iostream>

#include "MeshMassProperties.h"
#include "MeshInfoTests.h"

#define EXPOSE_HELPER_FUNCTIONS_FOR_UNIT_TEST
#define VERBOSE_UNIT_TESTS

const btScalar acceptableRelativeError(1.0e-5);
const btScalar acceptableAbsoluteError(1.0e-4);

void printMatrix(const std::string& name, const btMatrix3x3& matrix) {
    std::cout << name << " = [" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "    " << matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

void MeshInfoTests::testParallelAxisTheorem() {
#ifdef EXPOSE_HELPER_FUNCTIONS_FOR_UNIT_TEST
    // verity we can compute the inertia tensor of a box in two different ways:
    // (a) as one box
    // (b) as a combination of two partial boxes.
#ifdef VERBOSE_UNIT_TESTS
    std::cout << "\n" << __FUNCTION__ << std::endl;
#endif // VERBOSE_UNIT_TESTS

    btScalar bigBoxX = 7.0;
    btScalar bigBoxY = 9.0;
    btScalar bigBoxZ = 11.0;
    btScalar bigBoxMass = bigBoxX * bigBoxY * bigBoxZ;
    btMatrix3x3 bitBoxInertia;
    computeBoxInertia(bigBoxMass, btVector3(bigBoxX, bigBoxY, bigBoxZ), bitBoxInertia);

    btScalar smallBoxX = bigBoxX / 2.0;
    btScalar smallBoxY = bigBoxY;
    btScalar smallBoxZ = bigBoxZ;
    btScalar smallBoxMass = smallBoxX * smallBoxY * smallBoxZ;
    btMatrix3x3 smallBoxI;
    computeBoxInertia(smallBoxMass, btVector3(smallBoxX, smallBoxY, smallBoxZ), smallBoxI);

    btVector3 smallBoxOffset(smallBoxX / 2.0, 0.0, 0.0);

    btMatrix3x3 smallBoxShiftedRight = smallBoxI;
    applyParallelAxisTheorem(smallBoxShiftedRight, smallBoxOffset, smallBoxMass);

    btMatrix3x3 smallBoxShiftedLeft = smallBoxI;
    applyParallelAxisTheorem(smallBoxShiftedLeft, -smallBoxOffset, smallBoxMass);

    btMatrix3x3 twoSmallBoxesInertia = smallBoxShiftedRight + smallBoxShiftedLeft;

    // verify bigBox same as twoSmallBoxes
    btScalar error;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            error = bitBoxInertia[i][j] - twoSmallBoxesInertia[i][j];
            if (fabsf(error) > acceptableAbsoluteError) {
                std::cout << __FILE__ << ":" << __LINE__ << " ERROR : box inertia[" << i << "][" << j << "] off by = " << error << std::endl;
            }
        }
    }

#ifdef VERBOSE_UNIT_TESTS
    printMatrix("expected inertia", bitBoxInertia);
    printMatrix("computed inertia", twoSmallBoxesInertia);
#endif // VERBOSE_UNIT_TESTS
#endif // EXPOSE_HELPER_FUNCTIONS_FOR_UNIT_TEST
}

void MeshInfoTests::testTetrahedron(){
    // given the four vertices of a tetrahedron verify the analytic formula for inertia
    // agrees with expected results
#ifdef VERBOSE_UNIT_TESTS
    std::cout << "\n" << __FUNCTION__ << std::endl;
#endif // VERBOSE_UNIT_TESTS

    // these numbers from the Tonon paper:
    btVector3 points[4];
    points[0] = btVector3(8.33220, -11.86875, 0.93355);
    points[1] = btVector3(0.75523, 5.00000, 16.37072);
    points[2] = btVector3(52.61236, 5.00000, -5.38580);
    points[3] = btVector3(2.00000, 5.00000, 3.00000);

    btScalar expectedVolume = 1873.233236;

    btMatrix3x3 expectedInertia;
    expectedInertia[0][0] = 43520.33257;
    expectedInertia[1][1] = 194711.28938;
    expectedInertia[2][2] = 191168.76173;
    expectedInertia[1][2] = -4417.66150;
    expectedInertia[2][1] = -4417.66150;
    expectedInertia[0][2] = 46343.16662;
    expectedInertia[2][0] = 46343.16662;
    expectedInertia[0][1] = -11996.20119;
    expectedInertia[1][0] = -11996.20119;

    // compute volume
    btScalar volume = computeTetrahedronVolume(points);
    btScalar error = (volume - expectedVolume) / expectedVolume;
    if (fabsf(error) > acceptableRelativeError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : volume of tetrahedron off by = "
            << error << std::endl;
    }

    btVector3 centerOfMass = 0.25f * (points[0] + points[1] + points[2] + points[3]);

    // compute inertia tensor
    // (shift the points so that tetrahedron's local centerOfMass is at origin)
    for (int i = 0; i < 4; ++i) {
        points[i] -= centerOfMass;
    }
    btMatrix3x3 inertia;
    computeTetrahedronInertia(volume, points, inertia);

    // verify
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            error = (inertia[i][j] - expectedInertia[i][j]) / expectedInertia[i][j];
            if (fabsf(error) > acceptableRelativeError) {
                std::cout << __FILE__ << ":" << __LINE__ << " ERROR : inertia[" << i << "][" << j << "] off by "
                    << error << std::endl;
            }
        }
    }

#ifdef VERBOSE_UNIT_TESTS
    std::cout << "expected volume = " << expectedVolume << std::endl;
    std::cout << "measured volume = " << volume << std::endl;
    printMatrix("expected inertia", expectedInertia);
    printMatrix("computed inertia", inertia);

    // when building VERBOSE you might be instrested in the results from the brute force method:
    btMatrix3x3 bruteInertia;
    computeTetrahedronInertiaByBruteForce(points, bruteInertia);
    printMatrix("brute inertia", bruteInertia);
#endif // VERBOSE_UNIT_TESTS
}

void MeshInfoTests::testOpenTetrahedonMesh() {
    // given the simplest possible mesh (open, with one triangle)
    // verify MeshMassProperties computes the right nubers
#ifdef VERBOSE_UNIT_TESTS
    std::cout << "\n" << __FUNCTION__ << std::endl;
#endif // VERBOSE_UNIT_TESTS

    // these numbers from the Tonon paper:
    VectorOfPoints points;
    points.push_back(btVector3(8.33220, -11.86875, 0.93355));
    points.push_back(btVector3(0.75523, 5.00000, 16.37072));
    points.push_back(btVector3(52.61236, 5.00000, -5.38580));
    points.push_back(btVector3(2.00000, 5.00000, 3.00000));

    btScalar expectedVolume = 1873.233236;

    btMatrix3x3 expectedInertia;
    expectedInertia[0][0] = 43520.33257;
    expectedInertia[1][1] = 194711.28938;
    expectedInertia[2][2] = 191168.76173;
    expectedInertia[1][2] = -4417.66150;
    expectedInertia[2][1] = -4417.66150;
    expectedInertia[0][2] = 46343.16662;
    expectedInertia[2][0] = 46343.16662;
    expectedInertia[0][1] = -11996.20119;
    expectedInertia[1][0] = -11996.20119;

    // test as an open mesh with one triangle
    VectorOfPoints shiftedPoints;
    shiftedPoints.push_back(points[0] - points[0]);
    shiftedPoints.push_back(points[1] - points[0]);
    shiftedPoints.push_back(points[2] - points[0]);
    shiftedPoints.push_back(points[3] - points[0]);
    VectorOfIndices triangles = { 1, 2, 3 };
    btVector3 expectedCenterOfMass = 0.25 * (shiftedPoints[0] + shiftedPoints[1] + shiftedPoints[2] + shiftedPoints[3]);

    // compute mass properties
    MeshMassProperties mesh(shiftedPoints, triangles);

    // verify
    btScalar error = (mesh.m_volume - expectedVolume) / expectedVolume;
    if (fabsf(error) > acceptableRelativeError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : volume of tetrahedron off by = " << error << std::endl;
    }

    error = (mesh.m_centerOfMass - expectedCenterOfMass).length();
    if (fabsf(error) > acceptableAbsoluteError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : centerOfMass of tetrahedron off by = " << error << std::endl;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            error = (mesh.m_inertia[i][j] - expectedInertia[i][j]) / expectedInertia[i][j];
            if (fabsf(error) > acceptableRelativeError) {
                std::cout << __FILE__ << ":" << __LINE__ << " ERROR : inertia[" << i << "][" << j << "] off by " << error << std::endl;
            }
        }
    }

#ifdef VERBOSE_UNIT_TESTS
    std::cout << "expected volume = " << expectedVolume << std::endl;
    std::cout << "measured volume = " << mesh.m_volume << std::endl;
    printMatrix("expected inertia", expectedInertia);
    printMatrix("computed inertia", mesh.m_inertia);
#endif // VERBOSE_UNIT_TESTS
}

void MeshInfoTests::testClosedTetrahedronMesh() {
    // given a tetrahedron as a closed mesh of four tiangles
    // verify MeshMassProperties computes the right nubers
#ifdef VERBOSE_UNIT_TESTS
    std::cout << "\n" << __FUNCTION__ << std::endl;
#endif // VERBOSE_UNIT_TESTS

    // these numbers from the Tonon paper:
    VectorOfPoints points;
    points.reserve(4);
    points.push_back(btVector3(8.33220, -11.86875, 0.93355));
    points.push_back(btVector3(0.75523, 5.00000, 16.37072));
    points.push_back(btVector3(52.61236, 5.00000, -5.38580));
    points.push_back(btVector3(2.00000, 5.00000, 3.00000));

    btScalar expectedVolume = 1873.233236;

    btMatrix3x3 expectedInertia;
    expectedInertia[0][0] = 43520.33257;
    expectedInertia[1][1] = 194711.28938;
    expectedInertia[2][2] = 191168.76173;

    expectedInertia[1][2] = -4417.66150;
    expectedInertia[2][1] = -4417.66150;

    expectedInertia[0][2] = 46343.16662;
    expectedInertia[2][0] = 46343.16662;

    expectedInertia[0][1] = -11996.20119;
    expectedInertia[1][0] = -11996.20119;

    btVector3 expectedCenterOfMass = 0.25 * (points[0] + points[1] + points[2] + points[3]);

    VectorOfIndices triangles = {
        0, 2, 1,
        0, 3, 2,
        0, 1, 3,
        1, 2, 3 };

    // compute mass properties
    MeshMassProperties mesh(points, triangles);

    // verify
    btScalar error;
    error = (mesh.m_volume - expectedVolume) / expectedVolume;
    if (fabsf(error) > acceptableRelativeError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : volume of tetrahedron off by = " << error << std::endl;
    }

    error = (mesh.m_centerOfMass - expectedCenterOfMass).length();
    if (fabsf(error) > acceptableAbsoluteError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : centerOfMass of tetrahedron off by = " << error << std::endl;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            error = (mesh.m_inertia[i][j] - expectedInertia[i][j]) / expectedInertia[i][j];
            if (fabsf(error) > acceptableRelativeError) {
                std::cout << __FILE__ << ":" << __LINE__ << " ERROR : inertia[" << i << "][" << j << "] off by " << error << std::endl;
            }
        }
    }

#ifdef VERBOSE_UNIT_TESTS
    std::cout << "(a) tetrahedron as mesh" << std::endl;
    std::cout << "expected volume = " << expectedVolume << std::endl;
    std::cout << "measured volume = " << mesh.m_volume << std::endl;
    printMatrix("expected inertia", expectedInertia);
    printMatrix("computed inertia", mesh.m_inertia);
#endif // VERBOSE_UNIT_TESTS

    // test again, but this time shift the points so that the origin is definitely OUTSIDE the mesh
    btVector3 shift = points[0] + expectedCenterOfMass;
    for (int i = 0; i < (int)points.size(); ++i) {
        points[i] += shift;
    }
    expectedCenterOfMass = 0.25 * (points[0] + points[1] + points[2] + points[3]);

    // compute mass properties
    mesh.computeMassProperties(points, triangles);

    // verify
    error = (mesh.m_volume - expectedVolume) / expectedVolume;
    if (fabsf(error) > acceptableRelativeError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : volume of tetrahedron off by = " << error << std::endl;
    }

    error = (mesh.m_centerOfMass - expectedCenterOfMass).length();
    if (fabsf(error) > acceptableAbsoluteError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : centerOfMass of tetrahedron off by = " << error << std::endl;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            error = (mesh.m_inertia[i][j] - expectedInertia[i][j]) / expectedInertia[i][j];
            if (fabsf(error) > acceptableRelativeError) {
                std::cout << __FILE__ << ":" << __LINE__ << " ERROR : inertia[" << i << "][" << j << "] off by " << error << std::endl;
            }
        }
    }

#ifdef VERBOSE_UNIT_TESTS
    std::cout << "(b) shifted tetrahedron as mesh" << std::endl;
    std::cout << "expected volume = " << expectedVolume << std::endl;
    std::cout << "measured volume = " << mesh.m_volume << std::endl;
    printMatrix("expected inertia", expectedInertia);
    printMatrix("computed inertia", mesh.m_inertia);
#endif // VERBOSE_UNIT_TESTS
}

void MeshInfoTests::testBoxAsMesh() {
    // verify that a mesh box produces the same mass properties as the analytic box.
#ifdef VERBOSE_UNIT_TESTS
    std::cout << "\n" << __FUNCTION__ << std::endl;
#endif // VERBOSE_UNIT_TESTS


    // build a box:
    //                            /
    //                           y
    //                          /
    //            6-------------------------7
    //           /|                        /|
    //          / |                       / |
    //         /  2----------------------/--3
    //        /  /                      /  /
    //   |   4-------------------------5  /  --x--
    //   z   | /                       | /
    //   |   |/                        |/
    //       0 ------------------------1

    btScalar x(5.0);
    btScalar y(3.0);
    btScalar z(2.0);

    VectorOfPoints points;
    points.reserve(8);

    points.push_back(btVector3(0.0, 0.0, 0.0));
    points.push_back(btVector3(x, 0.0, 0.0));
    points.push_back(btVector3(0.0, y, 0.0));
    points.push_back(btVector3(x, y, 0.0));
    points.push_back(btVector3(0.0, 0.0, z));
    points.push_back(btVector3(x, 0.0, z));
    points.push_back(btVector3(0.0, y, z));
    points.push_back(btVector3(x, y, z));

    VectorOfIndices triangles = {
        0, 1, 4,
        1, 5, 4,
        1, 3, 5,
        3, 7, 5,
        2, 0, 6,
        0, 4, 6,
        3, 2, 7,
        2, 6, 7,
        4, 5, 6,
        5, 7, 6,
        0, 2, 1,
        2, 3, 1
    };

    // compute expected mass properties analytically
    btVector3 expectedCenterOfMass = 0.5 * btVector3(x, y, z);
    btScalar expectedVolume = x * y * z;
    btMatrix3x3 expectedInertia;
    computeBoxInertia(expectedVolume, btVector3(x, y, z), expectedInertia);

    // compute the mass properties using the mesh
    MeshMassProperties mesh(points, triangles);

    // verify
    btScalar error;
    error = (mesh.m_volume - expectedVolume) / expectedVolume;
    if (fabsf(error) > acceptableRelativeError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : volume of tetrahedron off by = " << error << std::endl;
    }

    error = (mesh.m_centerOfMass - expectedCenterOfMass).length();
    if (fabsf(error) > acceptableAbsoluteError) {
        std::cout << __FILE__ << ":" << __LINE__ << " ERROR : centerOfMass of tetrahedron off by = " << error << std::endl;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (expectedInertia [i][j] == btScalar(0.0)) {
                error = mesh.m_inertia[i][j] - expectedInertia[i][j];
                if (fabsf(error) > acceptableAbsoluteError) {
                    std::cout << __FILE__ << ":" << __LINE__ << " ERROR : inertia[" << i << "][" << j << "] off by " << error
                        << " absolute"<< std::endl;
                }
            } else {
                error = (mesh.m_inertia[i][j] - expectedInertia[i][j]) / expectedInertia[i][j];
                if (fabsf(error) > acceptableRelativeError) {
                    std::cout << __FILE__ << ":" << __LINE__ << " ERROR : inertia[" << i << "][" << j << "] off by " << error << std::endl;
                }
            }
        }
    }

#ifdef VERBOSE_UNIT_TESTS
    std::cout << "expected volume = " << expectedVolume << std::endl;
    std::cout << "measured volume = " << mesh.m_volume << std::endl;
    std::cout << "expected center of mass = < "
        << expectedCenterOfMass[0] << ", "
        << expectedCenterOfMass[1] << ", "
        << expectedCenterOfMass[2] << "> " << std::endl;
    std::cout << "computed center of mass = < "
        << mesh.m_centerOfMass[0] << ", "
        << mesh.m_centerOfMass[1] << ", "
        << mesh.m_centerOfMass[2] << "> " << std::endl;
    printMatrix("expected inertia", expectedInertia);
    printMatrix("computed inertia", mesh.m_inertia);
#endif // VERBOSE_UNIT_TESTS
}

void MeshInfoTests::runAllTests() {
    testParallelAxisTheorem();
    testTetrahedron();
    testOpenTetrahedonMesh();
	testClosedTetrahedronMesh();
    testBoxAsMesh();
    //testWithCube();
}
