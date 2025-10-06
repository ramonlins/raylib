#ifndef RAYPHYSICS_H
#define RAYPHYSICS_H

#include "raylib.h"
#include "raymath.h"
#include <stdlib.h> // For malloc, free

//----------------------------------
// Configuration
//----------------------------------
#define RP_MAX_BODIES 32      // Max number of rigid bodies
#define RP_MAX_GEOMS 64       // Max number of collision geometries
#define RP_MAX_CONTACTS 256   // Max number of contacts to solve per step

// A typedef for our floating point numbers, like MuJoCo's mjtNum
// This makes it easy to switch between float and double later.
typedef float rpFloat;

//----------------------------------
// Enums (Inspired by MuJoCo's mjt enums)
//----------------------------------

// Type of geometric shape for collision
typedef enum {
    RP_GEOM_PLANE = 0,
    RP_GEOM_SPHERE,
    // Future geoms: RP_GEOM_BOX, RP_GEOM_CAPSULE, etc.
} rpGeomType;

// Integrator type
typedef enum {
    RP_INT_EULER = 0, // Semi-implicit Euler
    // Future integrators: RP_INT_RK4, etc.
} rpIntegratorType;


//----------------------------------
// Core Structures (Inspired by mjModel and mjData)
//----------------------------------

// --- `rpGeom` ---
// Analogous to a subset of mjModel's geom properties.
// Defines a single collision shape.
typedef struct {
    rpGeomType type;        // Geom type (RP_GEOM_SPHERE, etc.)
    int bodyId;             // ID of the body this geom is attached to
    Vector3 size;           // Size parameters (e.g., for a sphere, size.x is radius)
    Vector3 offset;         // Positional offset from the body's center of mass
    Quaternion orientation; // Rotational offset from the body's orientation

    // Material & Contact properties
    rpFloat friction;       // Coefficient of friction (0-1)
    rpFloat restitution;    // Coefficient of restitution (bounciness, 0-1)
} rpGeom;

// --- `rpBody` ---
// Analogous to a subset of mjModel's body properties.
// Defines the physical properties of a rigid body.
typedef struct {
    rpFloat mass;
    rpFloat invMass; // Inverse mass (1/mass), pre-calculated for efficiency
    // Future properties: inertia, etc.
} rpBody;

// --- `rpOption` ---
// Analogous to mjOption. Contains simulation parameters.
typedef struct {
    Vector3 gravity;
    rpFloat timestep;
    rpIntegratorType integrator;
} rpOption;


// --- `rpModel` (The "Blueprint") ---
// Analogous to mjModel. Contains the static definition of the physics world.
typedef struct {
    rpOption option;

    int num_bodies;
    int num_geoms;

    // Body properties
    rpBody bodies[RP_MAX_BODIES];
    
    // Initial state (like qpos0 in MuJoCo)
    Vector3 qpos0[RP_MAX_BODIES]; // Initial position for each body
    Vector3 qvel0[RP_MAX_BODIES]; // Initial velocity for each body

    // Geometry properties
    rpGeom geoms[RP_MAX_GEOMS];
} rpModel;


// --- `rpContact` ---
// Analogous to mjContact. Holds data for a single contact point.
typedef struct {
    int geom1;              // ID of the first geom in contact
    int geom2;              // ID of the second geom in contact
    Vector3 pos;            // Contact position in world space
    Vector3 normal;         // Contact normal, pointing from geom1 to geom2
    rpFloat depth;          // Penetration depth
} rpContact;


// --- `rpData` (The "Live Instance") ---
// Analogous to mjData. Contains the dynamic state of the simulation.
typedef struct {
    rpFloat time;           // Current simulation time

    // State vectors (like qpos, qvel in MuJoCo)
    Vector3 qpos[RP_MAX_BODIES];    // Position of each body
    Vector3 qvel[RP_MAX_BODIES];    // Linear velocity of each body
    // Future state: Quaternion qrot, Vector3 qangvel, etc.

    // Force accumulators (like qfrc_applied in MuJoCo)
    Vector3 qforce[RP_MAX_BODIES];  // Forces to be applied at next step

    // Contact data
    int num_contacts;
    rpContact contacts[RP_MAX_CONTACTS];
} rpData;


//----------------------------------
// Main Simulation API (Inspired by MuJoCo's mj_ functions)
//----------------------------------

// Creates and initializes a new rpModel. Returns NULL on failure.
// Analogous to creating a model from an XML/MJCF.
rpModel* rp_createModel(void);

// Frees all memory associated with an rpModel.
// Analogous to mj_deleteModel.
void rp_deleteModel(rpModel* m);

// Creates and initializes a new rpData for a given model. Returns NULL on failure.
// Analogous to mj_makeData.
rpData* rp_createData(const rpModel* m);

// Frees all memory associated with an rpData object.
// Analogous to mj_deleteData.
void rp_deleteData(rpData* d);

// Resets an rpData instance to the initial state defined in its model.
// Analogous to mj_resetData.
void rp_resetData(const rpModel* m, rpData* d);

// The core function: advances the simulation by one timestep.
// Analogous to mj_step.
void rp_step(const rpModel* m, rpData* d);

#endif // RAYPHYSICS_H

// --- Implementation ---
#ifdef RAYPHYSICS_IMPLEMENTATION

// --- Private Helper Functions ---

// Simple sphere-plane collision detection
static void DetectCollisionSpherePlane(const rpModel* m, rpData* d, int sphereGeomId, int planeGeomId) {
    const rpGeom* sphereGeom = &m->geoms[sphereGeomId];
    const rpGeom* planeGeom = &m->geoms[planeGeomId];
    
    // Sphere world position and radius
    Vector3 spherePos = d->qpos[sphereGeom->bodyId];
    rpFloat sphereRadius = sphereGeom->size.x;
    
    // Plane normal is assumed to be (0, 1, 0) for this simple floor
    Vector3 planeNormal = { 0.0f, 1.0f, 0.0f };
    rpFloat planeOffset = d->qpos[planeGeom->bodyId].y; // Plane's y-position

    // Signed distance from sphere center to plane surface
    rpFloat distance = Vector3DotProduct(spherePos, planeNormal) - planeOffset - sphereRadius;

    if (distance < 0.0f && d->num_contacts < RP_MAX_CONTACTS) {
        rpContact* c = &d->contacts[d->num_contacts++];
        c->geom1 = sphereGeomId;
        c->geom2 = planeGeomId;
        c->depth = -distance;
        c->normal = planeNormal;
        c->pos = Vector3Subtract(spherePos, Vector3Scale(planeNormal, sphereRadius + distance));
    }
}

// --- API Implementation ---

rpModel* rp_createModel(void) {
    rpModel* m = (rpModel*)malloc(sizeof(rpModel));
    if (!m) return NULL;
    
    // Set default options (like mj_defaultOption)
    m->option.gravity = (Vector3){ 0.0f, -9.81f, 0.0f };
    m->option.timestep = 1.0f / 60.0f;
    m->option.integrator = RP_INT_EULER;

    m->num_bodies = 0;
    m->num_geoms = 0;

    return m;
}

void rp_deleteModel(rpModel* m) {
    if (m) free(m);
}

rpData* rp_createData(const rpModel* m) {
    if (!m) return NULL;
    rpData* d = (rpData*)malloc(sizeof(rpData));
    if (!d) return NULL;
    
    rp_resetData(m, d);
    
    return d;
}

void rp_deleteData(rpData* d) {
    if (d) free(d);
}

void rp_resetData(const rpModel* m, rpData* d) {
    d->time = 0.0f;
    d->num_contacts = 0;
    for (int i = 0; i < m->num_bodies; ++i) {
        d->qpos[i] = m->qpos0[i];
        d->qvel[i] = m->qvel0[i];
        d->qforce[i] = (Vector3){ 0 };
    }
}

void rp_step(const rpModel* m, rpData* d) {
    rpFloat dt = m->option.timestep;

    // 1. Apply forces (like mj_fwdVelocity)
    for (int i = 0; i < m->num_bodies; ++i) {
        // Clear previous forces
        d->qforce[i] = (Vector3){ 0 };
        
        // Apply gravity if body has mass
        if (m->bodies[i].invMass > 0.0f) {
            d->qforce[i] = Vector3Add(d->qforce[i], Vector3Scale(m->option.gravity, m->bodies[i].mass));
        }
    }

    // 2. Integrate velocities and positions (like mj_Euler)
    for (int i = 0; i < m->num_bodies; ++i) {
        if (m->bodies[i].invMass > 0.0f) {
            // v_new = v_old + (F / m) * dt
            Vector3 acceleration = Vector3Scale(d->qforce[i], m->bodies[i].invMass);
            d->qvel[i] = Vector3Add(d->qvel[i], Vector3Scale(acceleration, dt));

            // p_new = p_old + v_new * dt
            d->qpos[i] = Vector3Add(d->qpos[i], Vector3Scale(d->qvel[i], dt));
        }
    }

    // 3. Collision detection (like mj_collision)
    d->num_contacts = 0;
    // Simple N^2 check. For a large number of objects, a broadphase (like BVH) would be needed.
    for (int i = 0; i < m->num_geoms; ++i) {
        for (int j = i + 1; j < m->num_geoms; ++j) {
            const rpGeom* g1 = &m->geoms[i];
            const rpGeom* g2 = &m->geoms[j];
            
            // Sphere-Plane check
            if (g1->type == RP_GEOM_SPHERE && g2->type == RP_GEOM_PLANE) {
                DetectCollisionSpherePlane(m, d, i, j);
            }
            else if (g1->type == RP_GEOM_PLANE && g2->type == RP_GEOM_SPHERE) {
                DetectCollisionSpherePlane(m, d, j, i);
            }
        }
    }

    // 4. Collision resolution (Constraint Solver, like mj_fwdConstraint)
    // This is a very simple impulse-based solver.
    for (int i = 0; i < d->num_contacts; ++i) {
        rpContact* c = &d->contacts[i];
        const rpGeom* g1 = &m->geoms[c->geom1];
        const rpGeom* g2 = &m->geoms[c->geom2];
        const rpBody* b1 = &m->bodies[g1->bodyId];
        const rpBody* b2 = &m->bodies[g2->bodyId];

        Vector3 v1 = d->qvel[g1->bodyId];
        Vector3 v2 = d->qvel[g2->bodyId];
        Vector3 relVel = Vector3Subtract(v1, v2);

        rpFloat velAlongNormal = Vector3DotProduct(relVel, c->normal);

        // Do not resolve if velocities are separating
        if (velAlongNormal > 0) continue;

        // Use the minimum restitution of the two bodies
        rpFloat e = fminf(g1->restitution, g2->restitution);
        
        // Calculate impulse magnitude
        rpFloat j = -(1.0f + e) * velAlongNormal;
        j /= (b1->invMass + b2->invMass);
        
        // Apply impulse
        Vector3 impulse = Vector3Scale(c->normal, j);
        if (b1->invMass > 0) d->qvel[g1->bodyId] = Vector3Add(d->qvel[g1->bodyId], Vector3Scale(impulse, b1->invMass));
        if (b2->invMass > 0) d->qvel[g2->bodyId] = Vector3Subtract(d->qvel[g2->bodyId], Vector3Scale(impulse, b2->invMass));
        
        // Positional correction (to fix penetration)
        const rpFloat percent = 0.4f; // Penetration recovery percentage
        const rpFloat slop = 0.01f;   // Penetration allowance
        Vector3 correction = Vector3Scale(c->normal, percent * fmaxf(c->depth - slop, 0.0f) / (b1->invMass + b2->invMass));
        if (b1->invMass > 0) d->qpos[g1->bodyId] = Vector3Add(d->qpos[g1->bodyId], Vector3Scale(correction, b1->invMass));
        if (b2->invMass > 0) d->qpos[g2->bodyId] = Vector3Subtract(d->qpos[g2->bodyId], Vector3Scale(correction, b2->invMass));
    }
    
    // Update time
    d->time += dt;
}

#endif // RAYPHYSICS_IMPLEMENTATION