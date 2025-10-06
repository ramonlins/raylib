
#include "raylib.h"
#include "raymath.h"
#include "rayphysics.h" // Our new physics engine header!
#include <stdlib.h>     // For rand()

// For mobile, you might need a header from a library like raymob
// #include "raymob.h"

// --- Simulation Constants ---
#define MAX_PARTICLES 800
#define GRAVITY_FORCE 980.0f

// --- Helper Functions ---
// A mock function for mobile tilt. On a real device, this would get
// data from the accelerometer.
Vector2 GetTiltGravity() {
    // On Desktop: Use arrow keys to simulate tilt for testing
    Vector2 tilt = {0};
    if (IsKeyDown(KEY_RIGHT)) tilt.x += 1.0f;
    if (IsKeyDown(KEY_LEFT)) tilt.x -= 1.0f;
    if (IsKeyDown(KEY_DOWN)) tilt.y += 1.0f;
    if (IsKeyDown(KEY_UP)) tilt.y -= 1.0f;

    if (Vector2LengthSqr(tilt) > 0.0f) {
        return Vector2Scale(Vector2Normalize(tilt), GRAVITY_FORCE);
    }

    // Default gravity pointing down
    return (Vector2){ 0.0f, GRAVITY_FORCE };

    // On Mobile (Conceptual with RayMob or similar):
    // Vector3 accel = GetAccelerometer();
    // return (Vector2){ accel.x * GRAVITY_FORCE, -accel.y * GRAVITY_FORCE };
}

int main(void) {
    // --- Initialization ---
    const int screenWidth = 450;
    const int screenHeight = 800;
    InitWindow(screenWidth, screenHeight, "Raylib Hourglass");
    SetTargetFPS(60);

    // 1. Create the Physics World ("Model")
    rpPhysicsWorld* world = rpCreatePhysicsWorld(MAX_PARTICLES, (Vector2){ 0, GRAVITY_FORCE });
    world->particleRadius = 3.0f;
    world->restitution = 0.3f; // Slightly bouncy sand

    // 2. Define Hourglass Shape (the Boundaries)
    float w = screenWidth;
    float h = screenHeight;
    float neck = 20.0f;
    rpAddBoundary(world, (rpBoundary){{0, 0}, {w, 0}});       // Top
    rpAddBoundary(world, (rpBoundary){{w, 0}, {w, h}});       // Right
    rpAddBoundary(world, (rpBoundary){{w, h}, {0, h}});       // Bottom
    rpAddBoundary(world, (rpBoundary){{0, h}, {0, 0}});       // Left
    // Funnels
    rpAddBoundary(world, (rpBoundary){{0, h * 0.4f}, {w / 2 - neck, h / 2}});
    rpAddBoundary(world, (rpBoundary){{w, h * 0.4f}, {w / 2 + neck, h / 2}});

    // 3. Create the Physics State ("Data")
    rpPhysicsState* state = rpCreatePhysicsState(world);

    // 4. Populate with initial particles (sand in the top half)
    for (int i = 0; i < MAX_PARTICLES; i++) {
        float x = (float)GetRandomValue(world->particleRadius * 2, w - world->particleRadius * 2);
        float y = (float)GetRandomValue(world->particleRadius * 2, h * 0.4f - world->particleRadius * 2);
        rpAddParticle(state, (Vector2){x, y});
    }

    // --- Main Game Loop ---
    while (!WindowShouldClose()) {
        // --- Update ---
        float dt = GetFrameTime();

        // Update gravity based on device tilt (or keyboard for testing)
        Vector2 currentGravity = GetTiltGravity();
        rpSetGravity(world, currentGravity);

        // Step the physics simulation!
        rpStep(world, state, dt);

        // --- Draw ---
        BeginDrawing();
        ClearBackground(DARKGRAY);

        // Draw the hourglass boundaries
        for (int i = 0; i < world->boundaryCount; i++) {
            DrawLineV(world->boundaries[i].a, world->boundaries[i].b, LIGHTGRAY);
        }

        // Draw the sand particles
        for (int i = 0; i < state->particleCount; i++) {
            DrawCircleV(state->positions[i], world->particleRadius, BEIGE);
        }

        DrawFPS(10, 10);
        DrawText("Use Arrow Keys to Tilt", 10, 40, 20, LIGHTGRAY);

        EndDrawing();
    }

    // --- De-Initialization ---
    rpDestroyPhysicsState(state);
    rpDestroyPhysicsWorld(world);
    CloseWindow();

    return 0;
}