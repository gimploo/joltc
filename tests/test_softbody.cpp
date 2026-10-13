// Copyright (c) Amer Koleci and Contributors.
// Licensed under the MIT License (MIT). See LICENSE in the repository root for more information.

// Tests for the soft body API surface of joltc:
//   - JPH_SoftBodySharedSettings (Create/CreateCube/Clone, vertices, faces,
//     CreateConstraints/CreateConstraints2, Calculate* helpers, all constraint
//     Add/Get/Count accessors including edges, dihedral bends, volumes, LRAs,
//     inv bind matrices, skinned constraints and rods)
//   - JPH_SoftBodyCreationSettings
//   - JPH_BodyInterface_CreateSoftBody and JPH_Body soft body accessors
//   - JPH_SoftBodyMotionProperties runtime state (vertex accessors, bulk
//     GetVertices/SetVertices, face accessors, solver settings, skinning,
//     rods, CustomUpdate)
//
// Where possible the expectations are derived from the Jolt v5.6.0 sources,
// e.g. SoftBodySharedSettings::sCreateCube produces a grid of gridSize^3
// vertices with 3*gridSize^3 - 3*gridSize^2 edges, 6*(gridSize-1)^3 volume
// constraints and 12*(gridSize-1)^2 faces.

#include <gtest/gtest.h>
#include "joltc.h"

#include <cmath>

namespace SoftBodyLayers {
    static constexpr JPH_BroadPhaseLayer NON_MOVING = 0;
    static constexpr JPH_BroadPhaseLayer MOVING = 1;
    static constexpr uint32_t NUM_BROAD_PHASE_LAYERS = 2;
}

namespace SoftBodyObjectLayers {
    static constexpr JPH_ObjectLayer NON_MOVING = 0;
    static constexpr JPH_ObjectLayer MOVING = 1;
    static constexpr uint32_t NUM_OBJECT_LAYERS = 2;
}

class SoftBodyTest : public ::testing::Test {
protected:
    JPH_JobSystem* jobSystem = nullptr;
    JPH_BroadPhaseLayerInterface* broadPhaseLayerInterface = nullptr;
    JPH_ObjectVsBroadPhaseLayerFilter* objectVsBroadPhaseLayerFilter = nullptr;
    JPH_ObjectLayerPairFilter* objectLayerPairFilter = nullptr;
    JPH_PhysicsSystem* physicsSystem = nullptr;

    void SetUp() override {
        ASSERT_TRUE(JPH_Init());

        jobSystem = JPH_JobSystemThreadPool_Create(nullptr);
        ASSERT_NE(jobSystem, nullptr);

        broadPhaseLayerInterface = JPH_BroadPhaseLayerInterfaceTable_Create(SoftBodyObjectLayers::NUM_OBJECT_LAYERS, SoftBodyLayers::NUM_BROAD_PHASE_LAYERS);
        ASSERT_NE(broadPhaseLayerInterface, nullptr);
        JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(broadPhaseLayerInterface, SoftBodyObjectLayers::NON_MOVING, SoftBodyLayers::NON_MOVING);
        JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(broadPhaseLayerInterface, SoftBodyObjectLayers::MOVING, SoftBodyLayers::MOVING);

        objectLayerPairFilter = JPH_ObjectLayerPairFilterTable_Create(SoftBodyObjectLayers::NUM_OBJECT_LAYERS);
        ASSERT_NE(objectLayerPairFilter, nullptr);
        JPH_ObjectLayerPairFilterTable_EnableCollision(objectLayerPairFilter, SoftBodyObjectLayers::NON_MOVING, SoftBodyObjectLayers::MOVING);
        JPH_ObjectLayerPairFilterTable_EnableCollision(objectLayerPairFilter, SoftBodyObjectLayers::MOVING, SoftBodyObjectLayers::MOVING);

        objectVsBroadPhaseLayerFilter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(
            broadPhaseLayerInterface, SoftBodyLayers::NUM_BROAD_PHASE_LAYERS,
            objectLayerPairFilter, SoftBodyObjectLayers::NUM_OBJECT_LAYERS);
        ASSERT_NE(objectVsBroadPhaseLayerFilter, nullptr);

        JPH_PhysicsSystemSettings settings = {};
        settings.maxBodies = 1024;
        settings.numBodyMutexes = 0;
        settings.maxBodyPairs = 1024;
        settings.maxContactConstraints = 1024;
        settings.broadPhaseLayerInterface = broadPhaseLayerInterface;
        settings.objectVsBroadPhaseLayerFilter = objectVsBroadPhaseLayerFilter;
        settings.objectLayerPairFilter = objectLayerPairFilter;

        physicsSystem = JPH_PhysicsSystem_Create(&settings);
        ASSERT_NE(physicsSystem, nullptr);
    }

    void TearDown() override {
        if (physicsSystem) JPH_PhysicsSystem_Destroy(physicsSystem);
        if (jobSystem) JPH_JobSystem_Destroy(jobSystem);
        JPH_Shutdown();
    }

    /// gridSize=2, spacing=1 produces a centered cube of side 1: vertices at +/-0.5,
    /// 8 vertices, 12 edges, 6 volume constraints and 12 faces.
    JPH_SoftBodySharedSettings* MakeCubeSettings(uint32_t gridSize = 2, float spacing = 1.0f) const {
        JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_CreateCube(gridSize, spacing);
        EXPECT_NE(settings, nullptr);
        return settings;
    }

    /// Create a soft body from the given settings, add it to the physics system and return it.
    JPH_Body* CreateSoftBody(const JPH_SoftBodySharedSettings* settings, float y, float x = 0.0f, float z = 0.0f) const {
        JPH_BodyInterface* bodyInterface = JPH_PhysicsSystem_GetBodyInterface(physicsSystem);
        EXPECT_NE(bodyInterface, nullptr);
        if (!bodyInterface)
            return nullptr;

        JPH_RVec3 position = { x, y, z };
        JPH_Quat rotation = { 0.0f, 0.0f, 0.0f, 1.0f };
        JPH_SoftBodyCreationSettings* creationSettings =
            JPH_SoftBodyCreationSettings_Create2(settings, &position, &rotation, SoftBodyObjectLayers::MOVING);
        EXPECT_NE(creationSettings, nullptr);
        if (!creationSettings)
            return nullptr;

        JPH_Body* body = JPH_BodyInterface_CreateSoftBody(bodyInterface, creationSettings);
        EXPECT_NE(body, nullptr);
        if (!body) {
            JPH_SoftBodyCreationSettings_Destroy(creationSettings);
            return nullptr;
        }

        JPH_BodyID bodyID = JPH_Body_GetID(body);
        JPH_BodyInterface_AddBody(bodyInterface, bodyID, JPH_Activation_Activate);

        JPH_SoftBodyCreationSettings_Destroy(creationSettings);
        return body;
    }

    void DestroySoftBody(JPH_Body* body) const {
        if (!body)
            return;
        JPH_BodyInterface* bodyInterface = JPH_PhysicsSystem_GetBodyInterface(physicsSystem);
        JPH_BodyID bodyID = JPH_Body_GetID(body);
        JPH_BodyInterface_RemoveBody(bodyInterface, bodyID);
        JPH_BodyInterface_DestroyBody(bodyInterface, bodyID);
    }

    void StepFrames(int numFrames, float deltaTime = 1.0f / 60.0f) const {
        for (int i = 0; i < numFrames; ++i)
            JPH_PhysicsSystem_Update(physicsSystem, deltaTime, 1, jobSystem);
    }

    static JPH_Mat4 MakeIdentityMat4() {
        JPH_Mat4 m = {};
        m.column[0] = { 1, 0, 0, 0 };
        m.column[1] = { 0, 1, 0, 0 };
        m.column[2] = { 0, 0, 1, 0 };
        m.column[3] = { 0, 0, 0, 1 };
        return m;
    }

    static JPH_Mat4 MakeTranslationMat4(float x, float y, float z) {
        JPH_Mat4 m = MakeIdentityMat4();
        m.column[3] = { x, y, z, 1 };
        return m;
    }
};

// ============================================================================
// JPH_SoftBodySharedSettings
// ============================================================================

TEST_F(SoftBodyTest, Create_Destroy) {
    JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
    ASSERT_NE(settings, nullptr);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 0u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetFaceCount(settings), 0u);
    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CreateCube_CountsAndLayout) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);

    // gridSize=2 -> 2^3 vertices, 3*2^3 - 3*2^2 edges, 12*(2-1)^2 faces, 6*(2-1)^3 volumes
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 8u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 12u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetFaceCount(settings), 12u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVolumeConstraintCount(settings), 6u);

    // sCreateCube calls CalculateEdgeLengths and CalculateVolumeConstraintVolumes itself
    JPH_SoftEdge edge;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetEdgeConstraint(settings, 0, &edge));
    EXPECT_NEAR(edge.restLength, 1.0f, 1.0e-4f);

    JPH_SoftVolume volume;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVolumeConstraint(settings, 0, &volume));
    // A unit cube corner tetrahedron has volume 1/6 so 6 * volume = 1 -> rest volume 1
    EXPECT_NEAR(volume.sixRestVolume, 1.0f, 1.0e-3f);

    // Vertices are centered around the origin
    JPH_SoftVertex vertex;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVertex(settings, 0, &vertex));
    EXPECT_NEAR(vertex.position.x, -0.5f, 1.0e-5f);
    EXPECT_NEAR(vertex.position.y, -0.5f, 1.0e-5f);
    EXPECT_NEAR(vertex.position.z, -0.5f, 1.0e-5f);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, Clone_MatchesSource) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);
    JPH_SoftBodySharedSettings* clone = JPH_SoftBodySharedSettings_Clone(settings);
    ASSERT_NE(clone, nullptr);

    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(clone), JPH_SoftBodySharedSettings_GetVertexCount(settings));
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetFaceCount(clone), JPH_SoftBodySharedSettings_GetFaceCount(settings));
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(clone), JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings));

    JPH_SoftVertex a, b;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVertex(settings, 0, &a));
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVertex(clone, 0, &b));
    EXPECT_EQ(a.position.x, b.position.x);
    EXPECT_EQ(a.position.y, b.position.y);
    EXPECT_EQ(a.position.z, b.position.z);

    // The clone is independent
    JPH_SoftBodySharedSettings_RemoveVertex(clone, 0);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(clone), 7u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 8u);

    JPH_SoftBodySharedSettings_Destroy(clone);
    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, AddVertex_GetVertex_Roundtrip) {
    JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
    ASSERT_NE(settings, nullptr);

    JPH_SoftVertex vertex = {};
    vertex.position = { 1.0f, 2.0f, 3.0f };
    vertex.velocity = { 0.5f, -0.5f, 0.25f };
    vertex.invMass = 0.75f;

    JPH_SoftBodySharedSettings_AddVertex(settings, &vertex);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 1u);

    JPH_SoftVertex out = {};
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVertex(settings, 0, &out));
    EXPECT_FLOAT_EQ(out.position.x, 1.0f);
    EXPECT_FLOAT_EQ(out.position.y, 2.0f);
    EXPECT_FLOAT_EQ(out.position.z, 3.0f);
    EXPECT_FLOAT_EQ(out.velocity.x, 0.5f);
    EXPECT_FLOAT_EQ(out.velocity.y, -0.5f);
    EXPECT_FLOAT_EQ(out.velocity.z, 0.25f);
    EXPECT_FLOAT_EQ(out.invMass, 0.75f);

    // Out of bounds index
    EXPECT_FALSE(JPH_SoftBodySharedSettings_GetVertex(settings, 1, &out));

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, AddVertices_Batch) {
    JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
    ASSERT_NE(settings, nullptr);

    JPH_SoftVertex vertices[3] = {};
    vertices[0].position = { 0, 0, 0 };
    vertices[0].invMass = 1.0f;
    vertices[1].position = { 1, 0, 0 };
    vertices[1].invMass = 1.0f;
    vertices[2].position = { 0, 1, 0 };
    vertices[2].invMass = 1.0f;

    JPH_SoftBodySharedSettings_AddVertices(settings, vertices, 3);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 3u);

    JPH_SoftVertex out;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVertex(settings, 2, &out));
    EXPECT_FLOAT_EQ(out.position.y, 1.0f);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, RemoveVertex_ShiftsIndices) {
    JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
    ASSERT_NE(settings, nullptr);

    JPH_SoftVertex vertex = {};
    vertex.invMass = 1.0f;
    for (int i = 0; i < 3; ++i) {
        vertex.position = { (float)i, 0, 0 };
        JPH_SoftBodySharedSettings_AddVertex(settings, &vertex);
    }

    EXPECT_TRUE(JPH_SoftBodySharedSettings_RemoveVertex(settings, 1));
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 2u);

    JPH_SoftVertex out;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVertex(settings, 1, &out));
    EXPECT_FLOAT_EQ(out.position.x, 2.0f); // former index 2 shifted down
    EXPECT_FALSE(JPH_SoftBodySharedSettings_RemoveVertex(settings, 5));

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, AddFace_GetFace_Roundtrip) {
    JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
    ASSERT_NE(settings, nullptr);

    JPH_SoftFace face = { 0, 1, 2, 3 };
    JPH_SoftBodySharedSettings_AddFace(settings, &face);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetFaceCount(settings), 1u);

    JPH_SoftFace out = {};
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetFace(settings, 0, &out));
    EXPECT_EQ(out.vertex1, 0u);
    EXPECT_EQ(out.vertex2, 1u);
    EXPECT_EQ(out.vertex3, 2u);
    EXPECT_EQ(out.materialIndex, 3u);
    EXPECT_FALSE(JPH_SoftBodySharedSettings_GetFace(settings, 1, &out));

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, AddFaces_RemoveFace) {
    JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
    ASSERT_NE(settings, nullptr);

    JPH_SoftFace faces[2] = { { 0, 1, 2, 0 }, { 2, 1, 3, 0 } };
    JPH_SoftBodySharedSettings_AddFaces(settings, faces, 2);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetFaceCount(settings), 2u);

    EXPECT_TRUE(JPH_SoftBodySharedSettings_RemoveFace(settings, 0));
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetFaceCount(settings), 1u);
    EXPECT_FALSE(JPH_SoftBodySharedSettings_RemoveFace(settings, 5));

    JPH_SoftBodySharedSettings_Destroy(settings);
}

namespace {
    /// Build a flat quad of 4 vertices (unit square in the Z=0 plane) with 2 faces.
    /// Optionally makes vertex 0 kinematic (invMass = 0).
    JPH_SoftBodySharedSettings* MakeQuadSettings(bool makeVertex0Kinematic) {
        JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_Create();
        EXPECT_NE(settings, nullptr);
        if (!settings)
            return nullptr;

        JPH_SoftVertex vertices[4] = {};
        vertices[0].position = { 0, 0, 0 };
        vertices[0].invMass = makeVertex0Kinematic ? 0.0f : 1.0f;
        vertices[1].position = { 1, 0, 0 };
        vertices[1].invMass = 1.0f;
        vertices[2].position = { 1, 1, 0 };
        vertices[2].invMass = 1.0f;
        vertices[3].position = { 0, 1, 0 };
        vertices[3].invMass = 1.0f;
        JPH_SoftBodySharedSettings_AddVertices(settings, vertices, 4);

        JPH_SoftFace faces[2] = { { 0, 1, 2, 0 }, { 0, 2, 3, 0 } };
        JPH_SoftBodySharedSettings_AddFaces(settings, faces, 2);
        return settings;
    }
}

TEST_F(SoftBodyTest, CreateConstraints_GeneratesEdges) {
    // A quad generates 5 boundary edges + 1 shear edge (the diagonal of the quad) = 6
    JPH_SoftBodySharedSettings* settings = MakeQuadSettings(false);
    JPH_SoftBodySharedSettings_CreateConstraints(settings, 0.0f, JPH_SoftBodyBendType_None);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 6u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetDihedralBendConstraintCount(settings), 0u);

    JPH_SoftEdge edge;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetEdgeConstraint(settings, 0, &edge));
    // Rest length was calculated by CreateConstraints
    EXPECT_NEAR(edge.restLength, 1.0f, 1.0e-4f);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CreateConstraints_BendTypeDoesNotCreateBends) {
    // The CreateConstraints wrapper only sets mCompliance/mShearCompliance on the
    // attributes it builds; mBendCompliance stays at the Jolt default of FLT_MAX,
    // so no bend constraints are generated regardless of the requested bend type.
    JPH_SoftBodySharedSettings* settings = MakeQuadSettings(false);

    JPH_SoftBodySharedSettings_CreateConstraints(settings, 0.0f, JPH_SoftBodyBendType_Dihedral);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 6u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetDihedralBendConstraintCount(settings), 0u);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CreateConstraints2_WithAttributes_Dihedral) {
    JPH_SoftBodySharedSettings* settings = MakeQuadSettings(false);

    JPH_SoftBodyVertexAttributes attributes[4];
    for (JPH_SoftBodyVertexAttributes& attr : attributes) {
        attr.compliance = 0.0f;
        attr.shearCompliance = 0.0f;
        attr.bendCompliance = 0.5f;
        attr.lraType = JPH_SoftBodyLRAType_None;
        attr.lraMaxDistanceMultiplier = 1.0f;
    }

    // 8 degrees in radians, matches Jolt's default tolerance; the square is perfectly
    // perpendicular so the shear edge is created.
    JPH_SoftBodySharedSettings_CreateConstraints2(settings, attributes, 4, JPH_SoftBodyBendType_Dihedral, 0.13962634f);

    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 6u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetDihedralBendConstraintCount(settings), 1u);

    // The two faces share edge (0, 2), the opposing vertices are 1 and 3
    JPH_SoftDihedralBend bend;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetDihedralBendConstraint(settings, 0, &bend));
    EXPECT_EQ(bend.vertex1, 0u);
    EXPECT_EQ(bend.vertex2, 2u);
    EXPECT_EQ(bend.vertex3, 1u);
    EXPECT_EQ(bend.vertex4, 3u);
    EXPECT_NEAR(bend.compliance, 0.5f, 1.0e-6f);
    EXPECT_TRUE(std::isfinite(bend.initialAngle)); // calculated by CalculateBendConstraintConstants

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CreateConstraints2_LRAConstraints) {
    // Vertex 0 is kinematic, the other three are dynamic. Each dynamic vertex should
    // get a Euclidean distance LRA constraint back to vertex 0.
    JPH_SoftBodySharedSettings* settings = MakeQuadSettings(true);

    JPH_SoftBodyVertexAttributes attributes[4];
    for (JPH_SoftBodyVertexAttributes& attr : attributes) {
        attr.compliance = 0.0f;
        attr.shearCompliance = 0.0f;
        attr.bendCompliance = 0.5f;
        attr.lraType = JPH_SoftBodyLRAType_None;
        attr.lraMaxDistanceMultiplier = 1.0f;
    }
    attributes[1].lraType = JPH_SoftBodyLRAType_EuclideanDistance;
    attributes[2].lraType = JPH_SoftBodyLRAType_EuclideanDistance;
    attributes[3].lraType = JPH_SoftBodyLRAType_EuclideanDistance;

    JPH_SoftBodySharedSettings_CreateConstraints2(settings, attributes, 4, JPH_SoftBodyBendType_None, 0.13962634f);

    EXPECT_EQ(JPH_SoftBodySharedSettings_GetLRAConstraintCount(settings), 3u);

    JPH_SoftLRA lra;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetLRAConstraint(settings, 0, &lra));
    EXPECT_EQ(lra.vertex1, 0u);
    EXPECT_EQ(lra.vertex2, 1u);
    EXPECT_NEAR(lra.maxDistance, 1.0f, 1.0e-4f);

    // CalculateLRALengths re-scales the max distance with the given multiplier
    JPH_SoftBodySharedSettings_CalculateLRALengths(settings, 1.5f);
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetLRAConstraint(settings, 0, &lra));
    EXPECT_NEAR(lra.maxDistance, 1.5f, 1.0e-4f);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CalculateEdgeLengths_FillsRestLength) {
    JPH_SoftBodySharedSettings* settings = MakeQuadSettings(false);

    // Add an edge manually with a bogus rest length
    JPH_SoftEdge edge = { 0, 1, 100.0f, 0.0f };
    JPH_SoftBodySharedSettings_AddEdgeConstraint(settings, &edge);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 1u);

    JPH_SoftBodySharedSettings_CalculateEdgeLengths(settings);
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetEdgeConstraint(settings, 0, &edge));
    EXPECT_NEAR(edge.restLength, 1.0f, 1.0e-4f); // distance between vertex 0 and 1

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, ConstraintAccessors_Roundtrip_AllTypes) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);

    // Edge
    JPH_SoftEdge edge = { 0, 1, 2.0f, 0.3f };
    JPH_SoftBodySharedSettings_AddEdgeConstraint(settings, &edge);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 13u); // 12 from the cube + 1
    JPH_SoftEdge edgeOut;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetEdgeConstraint(settings, 12, &edgeOut));
    EXPECT_EQ(edgeOut.vertex1, 0u);
    EXPECT_EQ(edgeOut.vertex2, 1u);
    EXPECT_FLOAT_EQ(edgeOut.restLength, 2.0f);
    EXPECT_FLOAT_EQ(edgeOut.compliance, 0.3f);
    EXPECT_FALSE(JPH_SoftBodySharedSettings_GetEdgeConstraint(settings, 13, &edgeOut));

    // Dihedral bend
    JPH_SoftDihedralBend bend = { 0, 1, 2, 3, 0.1f, 0.5f };
    JPH_SoftBodySharedSettings_AddDihedralBendConstraint(settings, &bend);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetDihedralBendConstraintCount(settings), 1u);
    JPH_SoftDihedralBend bendOut;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetDihedralBendConstraint(settings, 0, &bendOut));
    EXPECT_EQ(bendOut.vertex1, 0u);
    EXPECT_EQ(bendOut.vertex4, 3u);
    EXPECT_FLOAT_EQ(bendOut.compliance, 0.1f);
    EXPECT_FLOAT_EQ(bendOut.initialAngle, 0.5f);

    // Volume
    JPH_SoftVolume volume = { 0, 1, 2, 3, 2.0f, 0.4f };
    JPH_SoftBodySharedSettings_AddVolumeConstraint(settings, &volume);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVolumeConstraintCount(settings), 7u); // 6 from the cube + 1
    JPH_SoftVolume volumeOut;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVolumeConstraint(settings, 6, &volumeOut));
    EXPECT_FLOAT_EQ(volumeOut.sixRestVolume, 2.0f);
    EXPECT_FLOAT_EQ(volumeOut.compliance, 0.4f);

    // LRA
    JPH_SoftLRA lra = { 0, 1, 1.5f };
    JPH_SoftBodySharedSettings_AddLRAConstraint(settings, &lra);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetLRAConstraintCount(settings), 1u);
    JPH_SoftLRA lraOut;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetLRAConstraint(settings, 0, &lraOut));
    EXPECT_EQ(lraOut.vertex1, 0u);
    EXPECT_EQ(lraOut.vertex2, 1u);
    EXPECT_FLOAT_EQ(lraOut.maxDistance, 1.5f);

    // Inv bind matrix
    JPH_SoftInvBind invBind = {};
    invBind.jointIndex = 3;
    invBind.invBind = MakeTranslationMat4(1.0f, 2.0f, 3.0f);
    JPH_SoftBodySharedSettings_AddInvBind(settings, &invBind);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetInvBindCount(settings), 1u);
    JPH_SoftInvBind invBindOut = {};
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetInvBind(settings, 0, &invBindOut));
    EXPECT_EQ(invBindOut.jointIndex, 3u);
    EXPECT_FLOAT_EQ(invBindOut.invBind.column[3].x, 1.0f);
    EXPECT_FLOAT_EQ(invBindOut.invBind.column[3].y, 2.0f);
    EXPECT_FLOAT_EQ(invBindOut.invBind.column[3].z, 3.0f);

    // Skinned constraint
    JPH_SoftSkinned skinned = {};
    skinned.vertex = 0;
    skinned.weights[0] = { 2, 0.5f };
    skinned.weights[1] = { 3, 0.5f };
    skinned.maxDistance = 0.4f;
    skinned.backStopDistance = 0.1f;
    skinned.backStopRadius = 0.9f;
    skinned.normalInfo = 0x01000000;
    JPH_SoftBodySharedSettings_AddSkinnedConstraint(settings, &skinned);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetSkinnedConstraintCount(settings), 1u);
    JPH_SoftSkinned skinnedOut = {};
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetSkinnedConstraint(settings, 0, &skinnedOut));
    EXPECT_EQ(skinnedOut.vertex, 0u);
    EXPECT_EQ(skinnedOut.weights[0].invBindIndex, 2u);
    EXPECT_FLOAT_EQ(skinnedOut.weights[0].weight, 0.5f);
    EXPECT_FLOAT_EQ(skinnedOut.maxDistance, 0.4f);
    EXPECT_EQ(skinnedOut.normalInfo, 0x01000000u);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, SetEdgeRestLength) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);
    EXPECT_TRUE(JPH_SoftBodySharedSettings_SetEdgeRestLength(settings, 0, 3.0f));

    JPH_SoftEdge edge;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetEdgeConstraint(settings, 0, &edge));
    EXPECT_FLOAT_EQ(edge.restLength, 3.0f);

    EXPECT_FALSE(JPH_SoftBodySharedSettings_SetEdgeRestLength(settings, 100, 1.0f));

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CalculateRodProperties) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);

    // Two rods + a bend twist constraint between them. The bishop frame must be a
    // zero quaternion so CalculateRodProperties knows it hasn't been calculated yet.
    JPH_SoftRodStretchShear rod0 = { 0, 1, 0.0f, 0.0f, 0.0f, { 0, 0, 0, 0 } };
    JPH_SoftRodStretchShear rod1 = { 2, 3, 0.0f, 0.0f, 0.0f, { 0, 0, 0, 0 } };
    JPH_SoftBodySharedSettings_AddRodStretchShearConstraint(settings, &rod0);
    JPH_SoftBodySharedSettings_AddRodStretchShearConstraint(settings, &rod1);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetRodStretchShearConstraintCount(settings), 2u);

    JPH_SoftRodBendTwist bendTwist = { 0, 1, 0.0f, { 0, 0, 0, 0 } };
    JPH_SoftBodySharedSettings_AddRodBendTwistConstraint(settings, &bendTwist);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetRodBendTwistConstraintCount(settings), 1u);

    JPH_SoftBodySharedSettings_CalculateRodProperties(settings);

    JPH_SoftRodStretchShear rodOut;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetRodStretchShearConstraint(settings, 0, &rodOut));
    EXPECT_NEAR(rodOut.length, 1.0f, 1.0e-4f);          // distance between vertex 0 and 1
    EXPECT_NEAR(rodOut.invMass, 1.0f, 1.0e-4f);         // min inv mass of both vertices
    // Bishop frame is a valid rotation
    float bishopLen = std::sqrt(rodOut.bishop.x * rodOut.bishop.x + rodOut.bishop.y * rodOut.bishop.y +
                                rodOut.bishop.z * rodOut.bishop.z + rodOut.bishop.w * rodOut.bishop.w);
    EXPECT_NEAR(bishopLen, 1.0f, 1.0e-3f);

    // Both rods are parallel (aligned with X), so omega0 should be identity
    JPH_SoftRodBendTwist bendOut;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetRodBendTwistConstraint(settings, 0, &bendOut));
    EXPECT_NEAR(bendOut.omega0.x, 0.0f, 1.0e-3f);
    EXPECT_NEAR(bendOut.omega0.y, 0.0f, 1.0e-3f);
    EXPECT_NEAR(bendOut.omega0.z, 0.0f, 1.0e-3f);
    EXPECT_NEAR(bendOut.omega0.w, 1.0f, 1.0e-3f);
    EXPECT_FALSE(JPH_SoftBodySharedSettings_GetRodBendTwistConstraint(settings, 1, &bendOut));

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CalculateVolumeConstraintVolumes) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);

    JPH_SoftVolume volume;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVolumeConstraint(settings, 0, &volume));
    EXPECT_NEAR(volume.sixRestVolume, 1.0f, 1.0e-3f);

    // Corrupt the rest volumes and recalculate
    volume.sixRestVolume = 0.0f;
    volume.compliance = 0.25f;
    JPH_SoftBodySharedSettings_AddVolumeConstraint(settings, &volume);
    JPH_SoftBodySharedSettings_CalculateVolumeConstraintVolumes(settings);
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetVolumeConstraint(settings, 6, &volume));
    EXPECT_GT(volume.sixRestVolume, 0.5f);
    EXPECT_FLOAT_EQ(volume.compliance, 0.25f);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, CalculateSkinnedConstraintNormals) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);

    // Normal info is only generated for skinned vertices whose adjacent faces connect
    // exclusively to skinned vertices, so skin every vertex of the cube.
    JPH_SoftInvBind invBind = {};
    invBind.jointIndex = 0;
    invBind.invBind = MakeIdentityMat4();
    JPH_SoftBodySharedSettings_AddInvBind(settings, &invBind);

    for (uint32_t v = 0; v < 8; ++v) {
        JPH_SoftSkinned skinned = {};
        skinned.vertex = v;
        skinned.weights[0] = { 0, 1.0f };
        skinned.maxDistance = 0.0f;
        skinned.backStopDistance = 0.0f;
        skinned.backStopRadius = 1.0f;
        JPH_SoftBodySharedSettings_AddSkinnedConstraint(settings, &skinned);
    }

    JPH_SoftBodySharedSettings_CalculateSkinnedConstraintNormals(settings);

    // normalInfo packs (#faces << 24) | face-list start index. Verify the packed face
    // count matches the number of faces actually adjacent to each vertex, computed
    // straight from the settings' own face list.
    for (uint32_t i = 0; i < 8; ++i) {
        JPH_SoftSkinned out = {};
        ASSERT_TRUE(JPH_SoftBodySharedSettings_GetSkinnedConstraint(settings, i, &out));

        uint32_t expectedFaceCount = 0;
        for (uint32_t f = 0; f < JPH_SoftBodySharedSettings_GetFaceCount(settings); ++f) {
            JPH_SoftFace face;
            if (JPH_SoftBodySharedSettings_GetFace(settings, f, &face) &&
                (face.vertex1 == i || face.vertex2 == i || face.vertex3 == i))
                ++expectedFaceCount;
        }

        EXPECT_GT(expectedFaceCount, 0u);
        EXPECT_NE(out.normalInfo, 0u);
        EXPECT_EQ(out.normalInfo >> 24, expectedFaceCount) << "skinned vertex " << i;
    }

    JPH_SoftBodySharedSettings_Destroy(settings);
}

TEST_F(SoftBodyTest, Optimize_PreservesConstraints) {
    JPH_SoftBodySharedSettings* settings = MakeCubeSettings(2, 1.0f);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 12u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVolumeConstraintCount(settings), 6u);

    JPH_SoftBodySharedSettings_Optimize(settings);

    EXPECT_EQ(JPH_SoftBodySharedSettings_GetEdgeConstraintCount(settings), 12u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(settings), 8u);

    JPH_SoftBodySharedSettings_Destroy(settings);
}

// ============================================================================
// JPH_SoftBodyCreationSettings
// ============================================================================

TEST_F(SoftBodyTest, CreationSettings_Create_GettersAndSetters) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);

    JPH_SoftBodyCreationSettings* settings = JPH_SoftBodyCreationSettings_Create();
    ASSERT_NE(settings, nullptr);
    // The default-constructed creation settings has no shared settings assigned yet
    // (this matches Jolt's SoftBodyCreationSettings default constructor).
    EXPECT_EQ(JPH_SoftBodyCreationSettings_GetSettings(settings), nullptr);

    // Position
    JPH_RVec3 position = { 1, 2, 3 };
    JPH_SoftBodyCreationSettings_SetPosition(settings, &position);
    JPH_RVec3 positionOut;
    JPH_SoftBodyCreationSettings_GetPosition(settings, &positionOut);
    EXPECT_EQ(positionOut.x, 1.0);
    EXPECT_EQ(positionOut.y, 2.0);
    EXPECT_EQ(positionOut.z, 3.0);

    // Rotation
    JPH_Quat rotation = { 0, 0, 0.70710678f, 0.70710678f };
    JPH_SoftBodyCreationSettings_SetRotation(settings, &rotation);
    JPH_Quat rotationOut;
    JPH_SoftBodyCreationSettings_GetRotation(settings, &rotationOut);
    EXPECT_FLOAT_EQ(rotationOut.w, 0.70710678f);

    // Shared settings
    JPH_SoftBodyCreationSettings_SetSettings(settings, shared);
    EXPECT_EQ(JPH_SoftBodyCreationSettings_GetSettings(settings), shared);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(JPH_SoftBodyCreationSettings_GetSettings(settings)), 8u);

    // Scalar / boolean state
    JPH_SoftBodyCreationSettings_SetUserData(settings, 0xDEADBEEF);
    EXPECT_EQ(JPH_SoftBodyCreationSettings_GetUserData(settings), 0xDEADBEEFu);
    JPH_SoftBodyCreationSettings_SetObjectLayer(settings, 1);
    EXPECT_EQ(JPH_SoftBodyCreationSettings_GetObjectLayer(settings), 1u);
    JPH_SoftBodyCreationSettings_SetNumIterations(settings, 7);
    EXPECT_EQ(JPH_SoftBodyCreationSettings_GetNumIterations(settings), 7u);
    JPH_SoftBodyCreationSettings_SetLinearDamping(settings, 0.25f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetLinearDamping(settings), 0.25f);
    JPH_SoftBodyCreationSettings_SetMaxLinearVelocity(settings, 120.0f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetMaxLinearVelocity(settings), 120.0f);
    JPH_SoftBodyCreationSettings_SetRestitution(settings, 0.1f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetRestitution(settings), 0.1f);
    JPH_SoftBodyCreationSettings_SetFriction(settings, 0.9f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetFriction(settings), 0.9f);
    JPH_SoftBodyCreationSettings_SetPressure(settings, 100.0f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetPressure(settings), 100.0f);
    JPH_SoftBodyCreationSettings_SetGravityFactor(settings, 0.5f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetGravityFactor(settings), 0.5f);
    JPH_SoftBodyCreationSettings_SetVertexRadius(settings, 0.05f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyCreationSettings_GetVertexRadius(settings), 0.05f);
    JPH_SoftBodyCreationSettings_SetUpdatePosition(settings, false);
    EXPECT_FALSE(JPH_SoftBodyCreationSettings_GetUpdatePosition(settings));
    JPH_SoftBodyCreationSettings_SetMakeRotationIdentity(settings, true);
    EXPECT_TRUE(JPH_SoftBodyCreationSettings_GetMakeRotationIdentity(settings));
    JPH_SoftBodyCreationSettings_SetAllowSleeping(settings, false);
    EXPECT_FALSE(JPH_SoftBodyCreationSettings_GetAllowSleeping(settings));
    JPH_SoftBodyCreationSettings_SetFacesDoubleSided(settings, true);
    EXPECT_TRUE(JPH_SoftBodyCreationSettings_GetFacesDoubleSided(settings));

    JPH_SoftBodyCreationSettings_Destroy(settings);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, CreationSettings_Create2_MatchesInputs) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);

    JPH_RVec3 position = { 4, 5, 6 };
    JPH_Quat rotation = { 0, 0, 0, 1 };
    JPH_SoftBodyCreationSettings* settings = JPH_SoftBodyCreationSettings_Create2(shared, &position, &rotation, 1);
    ASSERT_NE(settings, nullptr);

    EXPECT_EQ(JPH_SoftBodyCreationSettings_GetObjectLayer(settings), 1u);
    EXPECT_EQ(JPH_SoftBodySharedSettings_GetVertexCount(JPH_SoftBodyCreationSettings_GetSettings(settings)), 8u);

    JPH_RVec3 positionOut;
    JPH_SoftBodyCreationSettings_GetPosition(settings, &positionOut);
    EXPECT_EQ(positionOut.x, 4.0);
    EXPECT_EQ(positionOut.y, 5.0);
    EXPECT_EQ(positionOut.z, 6.0);

    JPH_SoftBodyCreationSettings_Destroy(settings);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

// ============================================================================
// Body creation & runtime access
// ============================================================================

TEST_F(SoftBodyTest, CreateSoftBody_BasicAccessors) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    ASSERT_NE(body, nullptr);

    EXPECT_TRUE(JPH_Body_IsSoftBody(body));
    EXPECT_FALSE(JPH_Body_IsRigidBody(body));
    EXPECT_EQ(JPH_Body_GetBodyType(body), JPH_BodyType_Soft);
    EXPECT_NE(JPH_Body_GetID(body), 0xFFFFFFFFu);

    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetVertexCount(properties), 8u);
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetFaceCount(properties), 12u);
    EXPECT_NE(JPH_SoftBodyMotionProperties_GetSettings(properties), nullptr);

    // Motion properties share the same shared settings instance as the creation settings used
    JPH_RVec3 position = { 0, 0.0f, 0 };
    JPH_Quat rotation = { 0, 0, 0, 1 };
    JPH_SoftBodyCreationSettings* creationSettings = JPH_SoftBodyCreationSettings_Create2(shared, &position, &rotation, 1);
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetSettings(properties), JPH_SoftBodyCreationSettings_GetSettings(creationSettings));
    JPH_SoftBodyCreationSettings_Destroy(creationSettings);

    // World space accessors
    EXPECT_EQ(JPH_Body_GetSoftBodyVertexCount(body), 8u);
    JPH_Vec3 worldVertex;
    JPH_Body_GetSoftBodyVertexPosition(body, 0, &worldVertex);
    EXPECT_TRUE(std::isfinite(worldVertex.x));
    EXPECT_TRUE(std::isfinite(worldVertex.y));
    EXPECT_TRUE(std::isfinite(worldVertex.z));
    EXPECT_NEAR(worldVertex.y, -0.5f, 1.0e-5f);

    // Face accessors match the shared settings
    JPH_SoftFace settingsFace;
    JPH_SoftFace propertiesFace;
    ASSERT_TRUE(JPH_SoftBodySharedSettings_GetFace(shared, 0, &settingsFace));
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetFace(properties, 0, &propertiesFace));
    EXPECT_EQ(propertiesFace.vertex1, settingsFace.vertex1);
    EXPECT_EQ(propertiesFace.vertex2, settingsFace.vertex2);
    EXPECT_EQ(propertiesFace.vertex3, settingsFace.vertex3);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, MotionProperties_VertexAccessors) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    // Initial COM-relative position of a cube corner
    JPH_Vec3 position;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 0, &position));
    EXPECT_NEAR(position.x, -0.5f, 1.0e-5f);
    EXPECT_NEAR(position.y, -0.5f, 1.0e-5f);
    EXPECT_NEAR(position.z, -0.5f, 1.0e-5f);

    // Full per-vertex data
    JPH_SoftBodyVertex vertex;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertex(properties, 0, &vertex));
    EXPECT_NEAR(vertex.position.x, -0.5f, 1.0e-5f);
    EXPECT_FLOAT_EQ(vertex.invMass, 1.0f);
    EXPECT_FALSE(JPH_SoftBodyMotionProperties_GetVertex(properties, 100, &vertex));

    // Read-write velocity
    JPH_Vec3 velocity = { 1.0f, -2.0f, 3.0f };
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_SetVertexVelocity(properties, 1, &velocity));
    JPH_Vec3 velocityOut;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexVelocity(properties, 1, &velocityOut));
    EXPECT_FLOAT_EQ(velocityOut.x, 1.0f);
    EXPECT_FLOAT_EQ(velocityOut.y, -2.0f);
    EXPECT_FLOAT_EQ(velocityOut.z, 3.0f);

    // Read-write inverse mass
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_SetVertexInvMass(properties, 1, 0.25f));
    float invMass = 0.0f;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexInvMass(properties, 1, &invMass));
    EXPECT_FLOAT_EQ(invMass, 0.25f);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, GetVertices_SetVertices_Roundtrip) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    const uint32_t numVertices = JPH_SoftBodyMotionProperties_GetVertexCount(properties);
    ASSERT_EQ(numVertices, 8u);

    // Null argument handling
    EXPECT_FALSE(JPH_SoftBodyMotionProperties_GetVertices(properties, nullptr, numVertices));
    EXPECT_FALSE(JPH_SoftBodyMotionProperties_SetVertices(properties, nullptr, numVertices));

    // Read all vertices
    JPH_SoftBodyVertex vertices[8] = {};
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertices(properties, vertices, numVertices));

    JPH_Vec3 single;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 1, &single));
    EXPECT_FLOAT_EQ(vertices[1].position.x, single.x);
    EXPECT_FLOAT_EQ(vertices[1].position.y, single.y);
    EXPECT_FLOAT_EQ(vertices[1].position.z, single.z);
    EXPECT_FLOAT_EQ(vertices[0].invMass, 1.0f);

    // Modify and write back
    const float originalY = vertices[1].position.y;
    vertices[1].position.y += 0.5f;
    vertices[1].velocity = { 2.0f, 0.0f, -1.0f };
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_SetVertices(properties, vertices, numVertices));

    JPH_SoftBodyVertex readBack;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertex(properties, 1, &readBack));
    EXPECT_NEAR(readBack.position.y, originalY + 0.5f, 1.0e-4f);
    EXPECT_FLOAT_EQ(readBack.velocity.x, 2.0f);
    EXPECT_FLOAT_EQ(readBack.velocity.z, -1.0f);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, GetVertexPositions_Bulk) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    JPH_Vec3 positions[8] = {};
    uint32_t count = 0;
    JPH_SoftBodyMotionProperties_GetVertexPositions(properties, positions, 8, &count);
    EXPECT_EQ(count, 8u);
    EXPECT_FLOAT_EQ(positions[0].x, -0.5f);
    EXPECT_FLOAT_EQ(positions[0].y, -0.5f);

    // World space bulk accessor
    JPH_Vec3 worldPositions[8] = {};
    JPH_Body_GetSoftBodyVertexPositions(body, worldPositions, 8, &count);
    EXPECT_EQ(count, 8u);
    EXPECT_TRUE(std::isfinite(worldPositions[0].x));
    EXPECT_TRUE(std::isfinite(worldPositions[0].y));
    EXPECT_TRUE(std::isfinite(worldPositions[0].z));

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, MotionProperties_RuntimeState) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    // Solver / pressure
    JPH_SoftBodyMotionProperties_SetNumIterations(properties, 3);
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetNumIterations(properties), 3u);
    JPH_SoftBodyMotionProperties_SetPressure(properties, 50.0f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyMotionProperties_GetPressure(properties), 50.0f);

    // Flags
    JPH_SoftBodyMotionProperties_SetUpdatePosition(properties, false);
    EXPECT_FALSE(JPH_SoftBodyMotionProperties_GetUpdatePosition(properties));
    JPH_SoftBodyMotionProperties_SetFacesDoubleSided(properties, true);
    EXPECT_TRUE(JPH_SoftBodyMotionProperties_GetFacesDoubleSided(properties));
    JPH_SoftBodyMotionProperties_SetEnableSkinConstraints(properties, true);
    EXPECT_TRUE(JPH_SoftBodyMotionProperties_GetEnableSkinConstraints(properties));
    JPH_SoftBodyMotionProperties_SetSkinnedMaxDistanceMultiplier(properties, 1.25f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyMotionProperties_GetSkinnedMaxDistanceMultiplier(properties), 1.25f);
    JPH_SoftBodyMotionProperties_SetVertexRadius(properties, 0.05f);
    EXPECT_FLOAT_EQ(JPH_SoftBodyMotionProperties_GetVertexRadius(properties), 0.05f);

    // Bounds and volume
    JPH_AABox bounds = {};
    JPH_SoftBodyMotionProperties_GetLocalBounds(properties, &bounds);
    EXPECT_LT(bounds.min.x, bounds.max.x);
    EXPECT_LT(bounds.min.y, bounds.max.y);
    EXPECT_LT(bounds.min.z, bounds.max.z);
    EXPECT_NEAR(bounds.min.x, -0.5f, 1.0e-4f);
    EXPECT_NEAR(bounds.max.x, 0.5f, 1.0e-4f);

    float volume = JPH_SoftBodyMotionProperties_GetVolume(properties);
    EXPECT_GT(volume, 0.5f);
    EXPECT_LT(volume, 1.5f);

    // Smoke: mass + inertia calculation must not disturb the vertex state
    JPH_SoftBodyMotionProperties_CalculateMassAndInertia(properties);
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetVertexCount(properties), 8u);

    // Null property handling
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetFaceCount(nullptr), 0u);
    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetRodCount(nullptr), 0u);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, SoftBody_Falls_OverTime) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    // Build update groups so edge/volume constraints are actually solved during simulation
    JPH_SoftBodySharedSettings_Optimize(shared);
    JPH_Body* body = CreateSoftBody(shared, 10.0f);

    JPH_RVec3 startPosition;
    JPH_Body_GetPosition(body, &startPosition);
    EXPECT_NEAR(startPosition.y, 10.0, 1.0e-4);

    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);
    JPH_Vec3 startVertex;
    JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 0, &startVertex);

    StepFrames(30);

    JPH_RVec3 endPosition;
    JPH_Body_GetPosition(body, &endPosition);
    EXPECT_LT(endPosition.y, 9.0); // the body fell

    JPH_Vec3 endVertex;
    JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 0, &endVertex);
    EXPECT_TRUE(std::isfinite(endVertex.y));
    EXPECT_NEAR(endVertex.y, startVertex.y, 5.0f); // COM-relative position stays bounded

    JPH_Vec3 linearVelocity;
    JPH_Body_GetLinearVelocity(body, &linearVelocity);
    EXPECT_LT(linearVelocity.y, 0.0f); // falling downwards

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, SoftBody_CustomUpdate) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);
    JPH_SoftBodySharedSettings_Optimize(shared);
    JPH_Body* body = CreateSoftBody(shared, 20.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    JPH_RVec3 startPosition;
    JPH_Body_GetPosition(body, &startPosition);
    EXPECT_NEAR(startPosition.y, 20.0, 1.0e-4);

    // Drive the soft body manually, bypassing JPH_PhysicsSystem_Update. 90 frames at
    // 1/60 s gives ~1.5 s of free fall (0.5 * g * t^2 ~= 11 m for the center of mass).
    const float deltaTime = 1.0f / 60.0f;
    for (int i = 0; i < 90; ++i)
        JPH_SoftBodyMotionProperties_CustomUpdate(properties, deltaTime, body, physicsSystem);

    JPH_RVec3 endPosition;
    JPH_Body_GetPosition(body, &endPosition);
    EXPECT_LT(endPosition.y, 18.0); // gravity moved the body down
    EXPECT_LT(endPosition.y, startPosition.y);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

// ============================================================================
// Skinning
// ============================================================================

namespace {
    JPH_SoftBodySharedSettings* MakeSkinnedCubeSettings() {
        JPH_SoftBodySharedSettings* settings = JPH_SoftBodySharedSettings_CreateCube(2, 1.0f);
        EXPECT_NE(settings, nullptr);
        if (!settings)
            return nullptr;

        // One joint, identity inverse bind matrix; skin vertex 1 (the corner at +0.5,-0.5,-0.5)
        JPH_SoftInvBind invBind = {};
        invBind.jointIndex = 0;
        JPH_Mat4 identity = {};
        identity.column[0] = { 1, 0, 0, 0 };
        identity.column[1] = { 0, 1, 0, 0 };
        identity.column[2] = { 0, 0, 1, 0 };
        identity.column[3] = { 0, 0, 0, 1 };
        invBind.invBind = identity;
        JPH_SoftBodySharedSettings_AddInvBind(settings, &invBind);

        JPH_SoftSkinned skinned = {};
        skinned.vertex = 1;
        skinned.weights[0] = { 0, 1.0f }; // full weight on joint 0
        skinned.maxDistance = 0.0f;        // 0 = hard skin the vertex to the skinned position
        skinned.backStopDistance = 0.0f;   // back stop disabled (backStopDistance >= maxDistance)
        skinned.backStopRadius = 1.0f;
        JPH_SoftBodySharedSettings_AddSkinnedConstraint(settings, &skinned);

        JPH_SoftBodySharedSettings_CalculateSkinnedConstraintNormals(settings);

        // Build internal update groups so the skin constraints are solved during simulation
        JPH_SoftBodySharedSettings_Optimize(settings);
        return settings;
    }
}

TEST_F(SoftBodyTest, SkinVertices_HardSkin) {
    JPH_SoftBodySharedSettings* shared = MakeSkinnedCubeSettings();
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    // Bind pose of vertex 1 (COM-relative)
    JPH_Vec3 bindPosition;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 1, &bindPosition));
    EXPECT_NEAR(bindPosition.y, -0.5f, 1.0e-4f);

    // Joint matrices are expressed relative to the center of mass transform
    JPH_RMat4 comTransform;
    JPH_Body_GetCenterOfMassTransform(body, &comTransform);

    // Skin with an identity joint matrix, then a joint matrix that translates +1 in Y.
    // invBind is identity, so the vertex is placed exactly at (bindPos + offset).
    JPH_Mat4 identityJoint = MakeIdentityMat4();
    JPH_SoftBodyMotionProperties_SkinVertices(properties, &comTransform, &identityJoint, 1, true, nullptr);
    JPH_Vec3 skinnedIdentity;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 1, &skinnedIdentity));
    EXPECT_NEAR(skinnedIdentity.y, -0.5f, 1.0e-3f);

    JPH_Mat4 translatedJoint = MakeTranslationMat4(0.0f, 1.0f, 0.0f);
    JPH_SoftBodyMotionProperties_SkinVertices(properties, &comTransform, &translatedJoint, 1, true, nullptr);
    JPH_Vec3 skinnedTranslated;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 1, &skinnedTranslated));
    EXPECT_NEAR(skinnedTranslated.y, 0.5f, 1.0e-3f); // bind y (-0.5) + 1
    EXPECT_NEAR(skinnedTranslated.x, bindPosition.x, 1.0e-3f);
    EXPECT_NEAR(skinnedTranslated.z, bindPosition.z, 1.0e-3f);

    // Hard skinning also resets the velocity of the vertex
    JPH_Vec3 velocity;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexVelocity(properties, 1, &velocity));
    EXPECT_NEAR(velocity.x, 0.0f, 1.0e-4f);
    EXPECT_NEAR(velocity.y, 0.0f, 1.0e-4f);
    EXPECT_NEAR(velocity.z, 0.0f, 1.0e-4f);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

TEST_F(SoftBodyTest, SkinVertices_HeldDuringSimulation) {
    JPH_SoftBodySharedSettings* shared = MakeSkinnedCubeSettings();
    JPH_Body* body = CreateSoftBody(shared, 0.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    JPH_RMat4 comTransform;
    JPH_Body_GetCenterOfMassTransform(body, &comTransform);
    JPH_Mat4 translatedJoint = MakeTranslationMat4(0.0f, 1.0f, 0.0f);

    // First, skin all vertices to the translated target (hard skin resets the soft body)
    JPH_SoftBodyMotionProperties_SkinVertices(properties, &comTransform, &translatedJoint, 1, true, nullptr);

    // Keep the body position fixed: with mUpdatePosition enabled Jolt re-centers the
    // body on the local bounds center every frame, which shifts both the vertices and
    // the skin state and makes the pinned vertex drift in COM space.
    JPH_SoftBodyMotionProperties_SetUpdatePosition(properties, false);

    // While simulating with skin constraints enabled the vertex should stay attached
    // to the skinned position (maxDistance = 0 => hard skinned every iteration).
    JPH_SoftBodyMotionProperties_SetEnableSkinConstraints(properties, true);
    StepFrames(5);

    JPH_Vec3 position;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetVertexPosition(properties, 1, &position));
    EXPECT_NEAR(position.y, 0.5f, 0.02f);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}

// ============================================================================
// Rods
// ============================================================================

TEST_F(SoftBodyTest, RodRuntime_State) {
    JPH_SoftBodySharedSettings* shared = MakeCubeSettings(2, 1.0f);

    JPH_SoftRodStretchShear rod0 = { 0, 1, 0.0f, 0.0f, 0.0f, { 0, 0, 0, 0 } };
    JPH_SoftRodStretchShear rod1 = { 2, 3, 0.0f, 0.0f, 0.0f, { 0, 0, 0, 0 } };
    JPH_SoftBodySharedSettings_AddRodStretchShearConstraint(shared, &rod0);
    JPH_SoftBodySharedSettings_AddRodStretchShearConstraint(shared, &rod1);
    JPH_SoftRodBendTwist bendTwist = { 0, 1, 0.0f, { 0, 0, 0, 0 } };
    JPH_SoftBodySharedSettings_AddRodBendTwistConstraint(shared, &bendTwist);
    JPH_SoftBodySharedSettings_CalculateRodProperties(shared);
    // Build update groups so the rod constraints are actually solved during simulation
    JPH_SoftBodySharedSettings_Optimize(shared);

    JPH_Body* body = CreateSoftBody(shared, 5.0f);
    JPH_SoftBodyMotionProperties* properties = JPH_Body_GetSoftBodyMotionProperties(body);
    ASSERT_NE(properties, nullptr);

    EXPECT_EQ(JPH_SoftBodyMotionProperties_GetRodCount(properties), 2u);

    // Initial state: rotation is identity * bishop, angular velocity is zero
    JPH_Quat rotation;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetRodRotation(properties, 0, &rotation));
    float rotLen = std::sqrt(rotation.x * rotation.x + rotation.y * rotation.y +
                             rotation.z * rotation.z + rotation.w * rotation.w);
    EXPECT_NEAR(rotLen, 1.0f, 1.0e-3f);

    JPH_Vec3 angularVelocity;
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetRodAngularVelocity(properties, 0, &angularVelocity));
    EXPECT_TRUE(std::isfinite(angularVelocity.x));
    EXPECT_TRUE(std::isfinite(angularVelocity.y));
    EXPECT_TRUE(std::isfinite(angularVelocity.z));

    // Bounds checking
    EXPECT_FALSE(JPH_SoftBodyMotionProperties_GetRodRotation(properties, 2, &rotation));
    EXPECT_FALSE(JPH_SoftBodyMotionProperties_GetRodAngularVelocity(properties, 2, &angularVelocity));

    // Simulate; rod states must remain valid unit quaternions
    StepFrames(10);
    ASSERT_TRUE(JPH_SoftBodyMotionProperties_GetRodRotation(properties, 0, &rotation));
    rotLen = std::sqrt(rotation.x * rotation.x + rotation.y * rotation.y +
                       rotation.z * rotation.z + rotation.w * rotation.w);
    EXPECT_NEAR(rotLen, 1.0f, 1.0e-3f);

    DestroySoftBody(body);
    JPH_SoftBodySharedSettings_Destroy(shared);
}