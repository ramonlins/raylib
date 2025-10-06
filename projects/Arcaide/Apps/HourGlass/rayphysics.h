
//----------------------------------------------------------------------------------
// rayphysics.h - A simple 2D particle physics engine for Raylib
//
// Inspiration:
// - rpPhysicsWorld is like mjModel: It holds the static definition of our simulation.
// - rpPhysicsState is like mjData: It holds the dynamic, per-frame state of all particles.
// - rpStep() is like mj_step(): It advances the simulation by one time step.
//----------------------------------------------------------------------------------

#ifndef RAYPHYSICS_H
#define RAYPHYSICS_H

#include "raylib.h"


#if defined(__cplusplus)
    #define RPAPI extern "C" // Support C++ calling C functions
#else
    #define RPAPI // Defined as empty for C compilers
#endif

//----------------------------------
// Data Structures
//----------------------------------

// Represents a static, immovable line segment in the world (the hourglass walls)
typedef struct rpBoundary {
    Vector2 a;
    Vector2 b;
} rpBoundary;

// rpPhysicsWorld: The "Model" - Defines the constant properties of our simulation.
// Inspired by mjModel, this struct describes the "rules" of the world.
typedef struct rpPhysicsWorld {
    // Physics properties
    Vector2 gravity;            // Global gravitational acceleration
    float particleRadius;       // Radius of each sand particle
    float particleMass;         // Mass of each sand particle
    float restitution;          // Bounciness of particles (0.0 to 1.0)
    float friction;             // Friction against boundaries

    // World geometry
    rpBoundary* boundaries;     // Array of static line boundaries
    int boundaryCount;          // Number of boundaries

    // Simulation capacity
    int maxParticles;           // The maximum number of particles the world can hold
} rpPhysicsWorld;


// rpPhysicsState: The "Data" - Holds the dynamic state of the simulation.
// Inspired by mjData, this struct contains the data that changes every frame.
typedef struct rpPhysicsState {
    int particleCount;          // Current number of active particles

    // Per-particle data buffers
    Vector2* positions;         // Current position of each particle (size: maxParticles)
    Vector2* velocities;        // Current velocity of each particle (size: maxParticles)
    Vector2* accelerations;     // Current acceleration of each particle (size: maxParticles)

} rpPhysicsState;


//----------------------------------
// API Functions
//----------------------------------

// --- World (Model) Management ---
// Creates and initializes the physics world (the "model")
RPAPI rpPhysicsWorld* rpCreatePhysicsWorld(int maxParticles, Vector2 gravity);
// Frees all memory associated with the physics world
RPAPI void rpDestroyPhysicsWorld(rpPhysicsWorld* world);
// Adds a static boundary to the world
RPAPI void rpAddBoundary(rpPhysicsWorld* world, rpBoundary boundary);
// Sets the global gravity for the world
RPAPI void rpSetGravity(rpPhysicsWorld* world, Vector2 gravity);


// --- State (Data) Management ---
// Creates and initializes a physics state corresponding to a world
RPAPI rpPhysicsState* rpCreatePhysicsState(const rpPhysicsWorld* world);
// Frees all memory associated with the physics state
RPAPI void rpDestroyPhysicsState(rpPhysicsState* state);
// Adds a particle to the simulation state at a given position
RPAPI void rpAddParticle(rpPhysicsState* state, Vector2 position);


// --- Simulation ---
// Advances the physics simulation by a given time step (dt)
// This is our equivalent of mj_step()
RPAPI void rpStep(const rpPhysicsWorld* world, rpPhysicsState* state, float dt);


#endif // RAYPHYSICS_H