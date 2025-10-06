
//----------------------------------------------------------------------------------
// rayphysics.c - A simple 2D particle physics engine for Raylib
//
// Implementation Details:
// - Uses a fixed number of sub-steps in rpStep() for simulation stability.
// - Implements O(n^2) particle-to-particle collision detection.
//   (Note: This is slow for >1000 particles; a spatial grid would be the next step).
// - Uses a simple Verlet-style integration for updating particle positions.
// - Handles dynamic resizing of the boundary array.
//----------------------------------------------------------------------------------

#include "rayphysics.h"
#include "raymath.h"
#include <stdlib.h> // Required for: malloc, calloc, realloc, free
#include <stdio.h>  // Required for: printf (for error reporting)

// Define the API implementation symbol
#define RPAPI

// Number of sub-steps to perform in each rpStep call.
// More sub-steps increase accuracy and prevent particles from "tunneling"
// through each other or boundaries at high speeds, at the cost of performance.
#define SUB_STEPS 8
#define INITIAL_BOUNDARY_CAPACITY 16

//----------------------------------------------------------------------------------
// Module Internal Functions
//----------------------------------------------------------------------------------

// Finds the closest point on a line segment to a given point.
static Vector2 GetClosestPointOnSegment(Vector2 point, Vector2 start, Vector2 end) {
    Vector2 segment = Vector2Subtract(end, start);
    float segmentLengthSq = Vector2LengthSqr(segment);

    if (segmentLengthSq == 0.0f) return start; // The segment is a point

    // Project point onto the line defined by the segment
    float t = Vector2DotProduct(Vector2Subtract(point, start), segment) / segmentLengthSq;
    t = Clamp(t, 0.0f, 1.0f); // Clamp t to be on the segment

    return Vector2Add(start, Vector2Scale(segment, t));
}

//----------------------------------------------------------------------------------
// World (Model) Management Functions
//----------------------------------------------------------------------------------

// Creates and initializes the physics world (the "model")
RPAPI rpPhysicsWorld* rpCreatePhysicsWorld(int maxParticles, Vector2 gravity) {
    rpPhysicsWorld* world = (rpPhysicsWorld*)malloc(sizeof(rpPhysicsWorld));
    if (!world) {
        printf("ERROR: Failed to allocate memory for physics world.\n");
        return NULL;
    }

    world->gravity = gravity;
    world->maxParticles = maxParticles;
    
    // Set some sane defaults
    world->particleRadius = 5.0f;
    world->particleMass = 1.0f;
    world->restitution = 0.4f;
    world->friction = 0.05f;

    world->boundaryCount = 0;
    world->boundaries = (rpBoundary*)malloc(sizeof(rpBoundary) * INITIAL_BOUNDARY_CAPACITY);
    if (!world->boundaries) {
        printf("ERROR: Failed to allocate memory for world boundaries.\n");
        free(world);
        return NULL;
    }
    // Store capacity for dynamic resizing
    // We use a temporary variable to store capacity as it's not part of the public API
    world->boundaries[INITIAL_BOUNDARY_CAPACITY-1].a.x = (float)INITIAL_BOUNDARY_CAPACITY; // Hacky way to store capacity

    return world;
}

// Frees all memory associated with the physics world
RPAPI void rpDestroyPhysicsWorld(rpPhysicsWorld* world) {
    if (!world) return;
    
    if (world->boundaries) free(world->boundaries);
    free(world);
}

// Adds a static boundary to the world
RPAPI void rpAddBoundary(rpPhysicsWorld* world, rpBoundary boundary) {
    if (!world || !world->boundaries) return;
    
    // Retrieve capacity stored in the last element's x component
    int capacity = (int)world->boundaries[INITIAL_BOUNDARY_CAPACITY-1].a.x;

    // Check if we need to resize the boundaries array
    if (world->boundaryCount >= capacity) {
        int newCapacity = capacity * 2;
        rpBoundary* newBoundaries = (rpBoundary*)realloc(world->boundaries, sizeof(rpBoundary) * newCapacity);

        if (!newBoundaries) {
            printf("ERROR: Failed to reallocate memory for world boundaries.\n");
            return;
        }

        world->boundaries = newBoundaries;
        world->boundaries[INITIAL_BOUNDARY_CAPACITY-1].a.x = (float)newCapacity; // Update stored capacity
    }
    
    world->boundaries[world->boundaryCount] = boundary;
    world->boundaryCount++;
}

// Sets the global gravity for the world
RPAPI void rpSetGravity(rpPhysicsWorld* world, Vector2 gravity) {
    if (world) world->gravity = gravity;
}


//----------------------------------------------------------------------------------
// State (Data) Management Functions
//----------------------------------------------------------------------------------

// Creates and initializes a physics state corresponding to a world
RPAPI rpPhysicsState* rpCreatePhysicsState(const rpPhysicsWorld* world) {
    if (!world) return NULL;
    
    rpPhysicsState* state = (rpPhysicsState*)malloc(sizeof(rpPhysicsState));
    if (!state) {
        printf("ERROR: Failed to allocate memory for physics state.\n");
        return NULL;
    }

    state->particleCount = 0;
    
    // Use calloc to allocate and zero-initialize the arrays
    state->positions = (Vector2*)calloc(world->maxParticles, sizeof(Vector2));
    state->velocities = (Vector2*)calloc(world->maxParticles, sizeof(Vector2));
    state->accelerations = (Vector2*)calloc(world->maxParticles, sizeof(Vector2));

    if (!state->positions || !state->velocities || !state->accelerations) {
        printf("ERROR: Failed to allocate memory for particle data arrays.\n");
        if(state->positions) free(state->positions);
        if(state->velocities) free(state->velocities);
        if(state->accelerations) free(state->accelerations);
        free(state);
        return NULL;
    }

    return state;
}

// Frees all memory associated with the physics state
RPAPI void rpDestroyPhysicsState(rpPhysicsState* state) {
    if (!state) return;
    
    if (state->positions) free(state->positions);
    if (state->velocities) free(state->velocities);
    if (state->accelerations) free(state->accelerations);
    free(state);
}

// Adds a particle to the simulation state at a given position
RPAPI void rpAddParticle(rpPhysicsState* state, Vector2 position) {
    // NOTE: This function assumes the caller will not add more particles than
    // the maxParticles defined in the world that created this state.
    if (!state) return;
    
    // Find the next available slot
    int index = state->particleCount;
    
    state->positions[index] = position;
    state->velocities[index] = (Vector2){0, 0};
    state->accelerations[index] = (Vector2){0, 0};
    
    state->particleCount++;
}

//----------------------------------------------------------------------------------
// Simulation Step Function
//----------------------------------------------------------------------------------

// Advances the physics simulation by a given time step (dt)
RPAPI void rpStep(const rpPhysicsWorld* world, rpPhysicsState* state, float dt) {
    if (!world || !state || dt <= 0.0f) return;
    
    const float sub_dt = dt / (float)SUB_STEPS;
    
    // Sub-stepping loop for stability
    for (int i = 0; i < SUB_STEPS; i++) {
        // 1. Apply forces (gravity)
        for (int p = 0; p < state->particleCount; p++) {
            state->accelerations[p] = world->gravity;
        }

        // 2. Solve collisions
        const float radius = world->particleRadius;
        const float radiusSq = radius * radius;
        
        // 2a. Particle-to-Particle collisions (O(n^2) - slow but simple)
        for (int p1 = 0; p1 < state->particleCount; p1++) {
            for (int p2 = p1 + 1; p2 < state->particleCount; p2++) {
                Vector2 collisionAxis = Vector2Subtract(state->positions[p1], state->positions[p2]);
                float distSq = Vector2LengthSqr(collisionAxis);
                float min_dist = radius * 2;

                if (distSq < min_dist*min_dist && distSq > 0.0f) {
                    float dist = sqrtf(distSq);
                    Vector2 normal = Vector2Scale(collisionAxis, 1.0f / dist);
                    
                    // --- Resolve Overlap ---
                    float overlap = min_dist - dist;
                    Vector2 resolution = Vector2Scale(normal, overlap * 0.5f);
                    state->positions[p1] = Vector2Add(state->positions[p1], resolution);
                    state->positions[p2] = Vector2Subtract(state->positions[p2], resolution);
                    
                    // --- Resolve Velocity (Collision Response) ---
                    Vector2 relVel = Vector2Subtract(state->velocities[p1], state->velocities[p2]);
                    float velAlongNormal = Vector2DotProduct(relVel, normal);
                    
                    if (velAlongNormal > 0) continue; // Particles are separating

                    // Calculate impulse
                    float e = world->restitution;
                    float j = -(1.0f + e) * velAlongNormal;
                    j /= 2.0f; // Assuming equal mass, invMass1 + invMass2 = 2

                    Vector2 impulse = Vector2Scale(normal, j);
                    state->velocities[p1] = Vector2Add(state->velocities[p1], impulse);
                    state->velocities[p2] = Vector2Subtract(state->velocities[p2], impulse);
                }
            }
        }
        
        // 2b. Particle-to-Boundary collisions
        for (int p = 0; p < state->particleCount; p++) {
            for (int b = 0; b < world->boundaryCount; b++) {
                rpBoundary boundary = world->boundaries[b];
                Vector2 closestPoint = GetClosestPointOnSegment(state->positions[p], boundary.a, boundary.b);
                Vector2 collisionAxis = Vector2Subtract(state->positions[p], closestPoint);
                float distSq = Vector2LengthSqr(collisionAxis);

                if (distSq < radiusSq) {
                    float dist = sqrtf(distSq);
                    Vector2 normal = (dist > 0.0f) ? Vector2Scale(collisionAxis, 1.0f / dist) : (Vector2){0, 1}; // Default normal if perfectly on the line

                    // --- Resolve Overlap ---
                    float overlap = radius - dist;
                    state->positions[p] = Vector2Add(state->positions[p], Vector2Scale(normal, overlap));
                    
                    // --- Resolve Velocity (Collision Response) ---
                    float vn = Vector2DotProduct(state->velocities[p], normal);
                    Vector2 v_normal = Vector2Scale(normal, vn);
                    Vector2 v_tangent = Vector2Subtract(state->velocities[p], v_normal);
                    
                    // Apply restitution to normal velocity
                    if (vn < 0) v_normal = Vector2Scale(v_normal, -world->restitution);
                    else v_normal = (Vector2){0,0}; // No bounce if moving away from wall
                    
                    // Apply friction to tangent velocity
                    v_tangent = Vector2Scale(v_tangent, 1.0f - world->friction);
                    
                    state->velocities[p] = Vector2Add(v_normal, v_tangent);
                }
            }
        }

        // 3. Integrate (Update velocity and position)
        for (int p = 0; p < state->particleCount; p++) {
            // Update velocity based on acceleration
            state->velocities[p] = Vector2Add(state->velocities[p], Vector2Scale(state->accelerations[p], sub_dt));
            
            // Update position based on new velocity
            state->positions[p] = Vector2Add(state->positions[p], Vector2Scale(state->velocities[p], sub_dt));
        }
    }
}