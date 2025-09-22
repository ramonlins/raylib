// =========================
// FILE: main.cpp
// Minimal DRL (REINFORCE) agent integrated with raylib using Eigen
// Now supports manual control option to collect trajectories and train.
// Keys:
//   [M] toggle manual mode
//   [T] toggle training on/off
//   [R] reset episode
//   [S] save model
//   [L] load model
//   [ESC] quit
// Manual controls: [Left]/[Right] arrows to move
// =========================

#include <raylib.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <random>
#include "../../../../src/external/eigen/Eigen/Dense"

using Eigen::MatrixXf; using Eigen::VectorXf;
using namespace std;

auto seed = random_device{}();
static mt19937 rng{seed};

#define LIGHTBLACK CLITERAL(Color) {20, 20, 20, 255}
#define LIGHTBLUE GetColor(0X3A5FE5FF)

inline float clampf(float value, float min, float max) {return value < min ? min: (value > max ? max : value);}

VectorXf softmax(const VectorXf& z){
    float m = z.maxCoeff();
    VectorXf e = (z.array() - m).exp();
    return e / e.sum();
}

// Static scene configuration
struct EnvConfig {
    // window
    int screenW = 1280;
    int screenH = 720;
    // target
    int targetW = 20;
    // player
    int playerW = 20;
    float speed = 200.0f;
    // episodes
    int maxSteps = 10; // 10 seconds
};

struct Transition {
    VectorXf s;      // State at time t
    VectorXf logits; // Pre-softmax outputs (needed for gradient calculation)
    VectorXf h;      // Hidden layer activations (needed for backprop to W1)
    int a;           // Action taken
    float r;         // Immediate reward
    // Note: REINFORCE doesn't need next state s' since it uses full returns
};

struct Env {
    EnvConfig cfg;

    float targetX{0};
    float playerX{0};
    float player2X{0};
    float steps{0};
    bool done{false};
    int scorePlayer{0};
    int scoreAgent{0};
    
    void init(){
        uniform_real_distribution<float> distX(40.0f, cfg.screenW - 40.0f);
        targetX = distX(rng);
        playerX = distX(rng);
        player2X = distX(rng);
    }


    void reset(){
        uniform_real_distribution<float> distX(40.0f, cfg.screenW - 40.0f);
        targetX = distX(rng);
        
        done = false;
        steps = 0;
    }

    void scoreReset(){
        scorePlayer = 0;
        scoreAgent = 0;
    }

    VectorXf observe() const{
        float dist = playerX - targetX;

        VectorXf s(3);
        float npx = (playerX / cfg.screenW) * 2.f - 1.f;
        float ntx = (targetX / cfg.screenW) * 2.f - 1.f;
        float dx =  dist / (cfg.screenW * 0.5f);

        s << npx, ntx, dx;
        return s;
    }

    tuple<float, bool> step(int action, float dt){
        
        float reward{0.f};
        if (action == 0) playerX -= cfg.speed * dt;
        else if (action == 2) playerX += cfg.speed * dt;
        playerX = clampf(playerX, 0.f, (float)cfg.screenW - cfg.targetW);
        float dist = fabs(playerX - targetX);
        bool done = false;
        if (dist <= cfg.targetW) {
            reward = 1.0f;
            done = true;
        } else if (steps >= cfg.maxSteps) {
            reward = -0.1f; // Small penalty for timeout
            done = true;
        } else {
            reward = -(dist / cfg.screenW) * 0.99f;
        }

        return {reward, done};
    }

    tuple<float, bool> step2(int action, float dt){
        float reward{0.f};
        if (action == 0) player2X -= cfg.speed * dt;
        else if (action == 2) player2X += cfg.speed * dt;
        player2X = clampf(player2X, 0.f, (float)cfg.screenW - cfg.targetW);
        float dist = fabs(player2X - targetX);
        bool done = false;
        if (dist <= cfg.targetW) {
            reward = 1.0f;
            done = true;
        } else if (steps >= cfg.maxSteps) {
            reward = -0.1f; // Small penalty for timeout
            done = true;
        } else {
            reward = -(dist / cfg.screenW) * 0.99f;
        }

        return {reward, done};
    }
};

struct PolicyMLP {
    int in{3}, hid{32}, out{3};
    MatrixXf W1; VectorXf b1;
    MatrixXf W2; VectorXf b2;
    float lr{1e-4f};

    // Constructor: initialize policy network parameters
    PolicyMLP() {
        // Normal distribution for weight initialization (mean = 0, std = 0.1)
        std::normal_distribution<float> nd(0.f, 0.1f);

        // First layer weights: (hid × in), initialized with random values
        W1 = MatrixXf(hid, in).unaryExpr([&](float){ return nd(rng); });

        // First layer bias: initialized to zeros (hid × 1)
        b1 = VectorXf::Zero(hid);

        // Second layer weights: (out × hid), initialized with random values
        W2 = MatrixXf(out, hid).unaryExpr([&](float){ return nd(rng); });

        // Second layer bias: initialized to zeros (out × 1)
        b2 = VectorXf::Zero(out);
    }

    // Forward pass of a simple 2-layer neural network
    // Input:  x (input vector)
    // Output: h (hidden activations), logits (unnormalized scores), probs (softmax probabilities)
    void forward(const VectorXf x, VectorXf& h, VectorXf& logits, VectorXf& probs) const {
        // Computation flow:
        //   x → [W1, b1, ReLU] → h → [W2, b2] → logits → [softmax] → probs

        // Hidden layer: linear transformation + ReLU activation
        // Dimensions: (32×3) * (3×1) + (32×1) → (32×1)
        h = (W1 * x + b1).array().max(0.f);

        // Output layer (logits before softmax)
        // Dimensions: (3×32) * (32×1) + (3×1) → (3×1)
        logits = W2 * h + b2;

        // Softmax activation: converts logits into probability distribution
        probs = softmax(logits);
    }


    // Sample action stochastically from probability distribution
    // This implements exploration: even low-probability actions can be chosen
    int sampleAction(const VectorXf& probs){
        // Create discrete distribution where each action's probability
        // determines how likely it is to be sampled
        // e.g., probs=[0.1, 0.3, 0.6] → action 2 chosen 60% of the time
        std::discrete_distribution<int> dd(probs.data(), probs.data()+probs.size());
        return dd(rng);  // Sample: returns 0, 1, or 2 based on probabilities
    }

    bool load(const string& path){
        ifstream f(path, ios::binary);  // binary mode
        if (!f) return false;
        auto loadM = [&](auto& M) {f.read((char*)M.data(), sizeof(float) * M.size());};
        loadM(W1); loadM(b1); loadM(W2); loadM(b2); 
        return true;
    }

};

bool isCollision(int playerX, int targetX, const EnvConfig& cfg){
    int playerRightEdge = playerX + cfg.playerW;
    int targetRightEdge = targetX + cfg.targetW;

    if ((playerRightEdge > targetX) && (playerX < targetRightEdge )) return true;
    return false;
}

int manualControl(Env& env){
    int action;
    if (IsKeyDown(KEY_A)) action = 0;
    else if (IsKeyDown(KEY_D)) action = 2;
    else action = 1;
    return action;
}

void drawGlowSprite(const Texture2D& glowSprite, const Rectangle& rect, float& destW, const float& destH, Color glowColor){
    DrawTexturePro(
        glowSprite,
        (Rectangle){ 0.0f, 0.0f, (float)glowSprite.width, (float)glowSprite.height }, // source rect
        (Rectangle){ rect.x, rect.height+100, destW, destH}, // dest rect
        (Vector2){ destW/2.0f, destH/2.0f }, // origin (center)
        0.0f, // rotation
        glowColor
    );
}


int main(){
    // === Initialization ===
    EnvConfig cfg;
    Env env{cfg};
    env.init();

    PolicyMLP pol;          // Agent policy network

    InitWindow(cfg.screenW, cfg.screenH, "Move to Goal");

    int a1{0}, a2{0};                              // Action
    std::vector<Transition> traj;          // Episode trajectory (for REINFORCE)

    Texture2D glowSpriteRed = LoadTexture("./assets/glow_light_red.png");
    Texture2D glowSpriteWhite = LoadTexture("./assets/glow_white.png");

    // === Main Loop ===
    while (!WindowShouldClose()){
        int fps = GetFPS();
        float dt = GetFrameTime();

        // --- Load the Model ---
        pol.load("policy.bin");

        // --- Keyboard Controls ---
        if(IsKeyPressed(KEY_R)) { env.reset(); traj.clear(); } // Reset env+traj
        
        // --- Agent Perception (forward pass) ---
        VectorXf s = env.observe();  // Current state
        VectorXf h, logits, probs;
        pol.forward(s, h, logits, probs);

        // --- Action Selection ---
        a1 = manualControl(env);  // Human control
        a2 = pol.sampleAction(probs);  // Agent picks action
        
        // --- Environment Step ---
        auto [rP, doneP] = env.step2(a1, dt);
        auto [r, done] = env.step(a2, dt);

        if (env.steps >= cfg.maxSteps) { env.reset(); traj.clear(); }
        if (doneP) { env.reset(); traj.clear(); env.scorePlayer++; }
        if (done) { env.reset(); traj.clear(); env.scoreAgent++; }

        if ((env.scorePlayer > 100) || (env.scoreAgent > 100)) { env.scoreReset(), traj.clear(); }

        // --- Rendering ---
        BeginDrawing();
        ClearBackground(BLACK);

        // UI overlays
        DrawText(TextFormat("fps: %d", fps), cfg.screenW - 50, 5, 9, LIGHTGRAY);
        DrawText(TextFormat("PLAYER: %d", env.scorePlayer), 10, 20, 18, RED);
        DrawText(TextFormat("SMITH  : %d", env.scoreAgent), 10, 40, 18, RED);
        DrawText("\xC2\xA9 2025 ARCAIDE STUDIO", cfg.screenW / 2 - 110, cfg.screenH - 50, 18, RED);
        //DrawLine(cfg.screenW/2, 0.f, cfg.screenW/2, cfg.screenH, WHITE);
        
        // Display agent policy
        // int bx = cfg.screenW - 200; int by = 20; int bw = 24; int gap = 6;
        // DrawText("pi(a/s):", bx, by, 20, DARKGRAY);
        // const char* labels[3] = {"LEFT", "IDLE", "RIGHT"};
        // for(int i=0; i < 3; ++i){
        //     int hbar = (int)(probs[i] * 100); //Calculates the horizontal width of each bar
        //     DrawRectangle(bx, by+30+i*(bw+gap), hbar, bw, i==a2? GRAY: DARKGRAY); //Draws a rectangle at position
        //     DrawText(TextFormat("%s %.2f", labels[i], probs[i]), bx + hbar + 8, by+30+i*(bw+gap)+4, 18, DARKGRAY); // Draws text showing the label and probability value
        // }
        
        // Scene rendering
        //DrawLine(0, cfg.screenH/2, cfg.screenW, cfg.screenH/2, LIGHTGRAY);
        BeginBlendMode(BLEND_ADDITIVE);
        float scale = 12.0f;
        float destW = (float)cfg.targetW * scale;
        float destH = (float)cfg.targetW * scale;
        Rectangle rectAgent = {(float)env.playerX, (float)cfg.screenH/2, destW, destH};
        Rectangle rectPlayer = {(float)env.player2X, (float)cfg.screenH/2, destW, destH};
        Rectangle rectTarget = {(float)env.targetX, (float)cfg.screenH/2, destW, destH};
        drawGlowSprite(glowSpriteRed, rectTarget, destW, destH, RED);
        drawGlowSprite(glowSpriteWhite, rectAgent, destW, destH, WHITE);
        drawGlowSprite(glowSpriteWhite, rectPlayer, destW, destH, WHITE);       
        EndBlendMode();

        EndDrawing();
        env.steps += dt;
        
    }

    // === Cleanup ===
    UnloadTexture(glowSpriteRed);
    UnloadTexture(glowSpriteWhite);
    CloseWindow();
    return 0;
}
