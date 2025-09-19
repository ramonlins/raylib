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
    int screenW = 800;
    int screenH = 450;
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
    float steps{0};
    bool done{false};

    void reset(){
        uniform_real_distribution<float> distX(40.0f, cfg.screenW - 40.0f);
        targetX = distX(rng);
        //playerX = (float)getRandomInt(cfg.playerW, cfg.screenW);
        done = false;
        steps = 0;
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

    tuple<float, bool> step(int action){
        float dt = GetFrameTime();
        float reward{0.f};
        if (action == 0) playerX -= cfg.speed * dt;
        else if (action == 2) playerX += cfg.speed * dt;
        playerX = clampf(playerX, 0.f, (float)cfg.screenW - cfg.targetW);
        steps += dt;
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

    int update(const vector<Transition>& traj, float gamma=0.99f) {
        // ============================================================
        // Step 1: Compute discounted returns (Monte Carlo estimate)
        // ------------------------------------------------------------
        // Returns are calculated backwards:
        //   G[t] = r[t] + γ * G[t+1]
        // This gives the total future reward from timestep t onward.
        // ============================================================
        vector<float> G(traj.size());
        float g = 0.f;
        for (int t = (int)traj.size()-1; t >= 0; --t) {
            g = traj[t].r + gamma * g;
            G[t] = g;
        }

        // ============================================================
        // Step 2: Normalize returns (variance reduction)
        // ------------------------------------------------------------
        // Normalization creates a baseline effect:
        //   - actions better than average → positive advantage
        //   - actions worse than average → negative advantage
        // This reduces variance in policy gradient updates.
        // ============================================================
        float mean = 0.f, sq = 0.f;
        for (float v : G) {
            mean += v;
            sq   += v*v;
        }
        mean /= G.size();
        float var = sq/G.size() - mean*mean;       // Var[X] = E[X²] - (E[X])²
        var = max(1e-8f, var);                     // avoid divide-by-zero
        float std = sqrt(var);

        for (float& v : G) {
            v = (v - mean) / (std + 1e-8f);
        }

        /* ============================================================
        Step 3: Compute gradients (REINFORCE rule)
        ------------------------------------------------------------
        Gradient estimate:
            ∇J(θ) ≈ Σ_t [ ∇ log π(a_t | s_t; θ) * G[t] ]

        Backpropagation through the 2-layer network:
        ∂loss/∂probs → ∂loss/∂logits → ∂loss/∂W2,b2 → ∂loss/∂W1,b1 → ∂loss/∂v1 → ∂loss/∂h → (apply ReLU derivative)

        More precisely:
            ∂loss/∂logits = (onehot - probs) * G[t] (softmax+cross-entropy derivative)
            ∂loss/∂W2 = ∂loss/∂logits * h^T (chain rule)
            ∂loss/∂b2 = ∂loss/∂logits (bias gradient)
            ∂loss/∂h = W2^T * ∂loss/∂logits (backprop through W2)
            ∂loss/∂v1 = ∂loss/∂h ⊙ ReLU'(v1) (element-wise, ReLU derivative)
            ∂loss/∂W1 = ∂loss/∂v1 * input^T (chain rule)
            ∂loss/∂b1 = ∂loss/∂v1 (bias gradient)
        // ============================================================ */

        MatrixXf dW2 = MatrixXf::Zero(W2.rows(), W2.cols());
        VectorXf db2 = VectorXf::Zero(b2.size());
        MatrixXf dW1 = MatrixXf::Zero(W1.rows(), W1.cols());
        VectorXf db1 = VectorXf::Zero(b1.size());

        // Softmax derivative simplifies nicely:
        //   ∂L/∂logits = (onehot - probs) * G[t]
        // This comes from combining softmax + cross-entropy.
        for (size_t t = 0; t < traj.size(); t++) {
            const auto& tr = traj[t];

            // --- Output layer ---
            VectorXf probs = softmax(tr.logits);
            VectorXf onehot = VectorXf::Zero(probs.size());
            onehot[tr.a] = 1.f;

            // δ2 = ∂Loss/∂logits (scaled by return G[t])
            VectorXf grad_logits = (onehot - probs) * G[t];

            // Accumulate gradients for W2, b2
            dW2 += grad_logits * tr.h.transpose();
            db2 += grad_logits;

            // --- Hidden layer ---
            // Backprop: δh = (W2^T * δ2) ∘ ReLU′(h_pre)
            VectorXf dh = W2.transpose() * grad_logits;
            VectorXf relu_mask = tr.h.unaryExpr([](float v){ return v > 0.f ? 1.f : 0.f; });
            dh = dh.cwiseProduct(relu_mask);

            // Accumulate gradients for W1, b1
            dW1 += dh * tr.s.transpose();
            db1 += dh;
        }

        // ============================================================
        // Step 4: Gradient ascent update
        // ------------------------------------------------------------
        // θ ← θ + α * ∇J(θ)
        // Note: using += because we *maximize* reward
        // ============================================================
        W2 += lr * dW2;  b2 += lr * db2;
        W1 += lr * dW1;  b1 += lr * db1;

        return 0;

    }

    bool save(const string& path){
        ofstream f(path, ios::binary);
        if(!f) return false;
        auto dumpM = [&](const auto& M){f.write((const char*)M.data(), sizeof(float) * M.size());};
        dumpM(W1); dumpM(b1); dumpM(W2); dumpM(b2);
        return true; 
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

int main(){
    // === Initialization ===
    EnvConfig cfg;
    Env env{cfg};
    env.reset();

    bool isManual{true};    // Toggle manual vs. agent control
    bool isTraining{false};  // Toggle training vs. evaluation

    PolicyMLP pol;          // Agent policy network

    InitWindow(cfg.screenW, cfg.screenH, "Move to Goal");

    int a{0};                              // Action
    std::vector<Transition> traj;          // Episode trajectory (for REINFORCE)

    // === Main Loop ===
    while (!WindowShouldClose()){
        // --- Keyboard Controls ---
        if(IsKeyPressed(KEY_M)) isManual = !isManual;    // Toggle manual/auto
        if(IsKeyPressed(KEY_T)) isTraining = !isTraining;// Toggle training/eval
        if(IsKeyPressed(KEY_R)) { env.reset(); traj.clear(); } // Reset env+traj
        if(IsKeyPressed(KEY_S)) { pol.save("policy.bin");}
        if(IsKeyPressed(KEY_L)) { pol.load("policy.bin");}

        // --- Agent Perception (forward pass) ---
        VectorXf s = env.observe();  // Current state
        VectorXf h, logits, probs;
        pol.forward(s, h, logits, probs);

        // --- Action Selection ---
        if (isManual){
            a = manualControl(env);  // Human control
        }else{
            a = pol.sampleAction(probs);  // Agent picks action
        }

        // --- Environment Step ---
        auto [r, done] = env.step(a);

        // --- Training Logic ---
        if (isTraining){
            // Save transition for Monte Carlo update
            traj.push_back({s, logits, h, a, r});

            // If episode ends → update policy
            if (env.steps >= cfg.maxSteps || done){
                pol.update(traj);   // REINFORCE update
                env.reset();        // Start new episode
                traj.clear();       // Clear trajectory
            }
        }else{
            // Evaluation mode → just reset on episode end
            if (env.steps >= cfg.maxSteps || done) { env.reset(); traj.clear(); }
        }

        // --- Rendering ---
        BeginDrawing();
        ClearBackground(LIGHTBLACK);

        // UI overlays
        DrawText(isManual ? "MANUAL" : "AUTO", 140, 10, 20, isManual ? ORANGE : DARKGRAY);
        DrawText(isTraining ? "TRAIN" : "EVAL", 240, 10, 20, isTraining ? ORANGE : DARKGRAY);
        DrawText("M: toggle manual | T: toggle training | R: reset | S: save | R: load | ESC: quit",10, 390, 18, DARKGRAY);
        float dist = std::fabs(env.playerX - env.targetX);
        DrawText(TextFormat("dist: %.1f", dist), 10, 86, 18, RED);
        DrawText(TextFormat("steps: %d/%d", (int)env.steps, cfg.maxSteps), 10, 64, 18, RED);
        int bx = cfg.screenW - 200; int by = 20; int bw = 24; int gap = 6;
        DrawText("pi(a/s):", bx, by, 20, DARKGRAY);
        const char* labels[3] = {"LEFT", "IDLE", "RIGHT"};
        for(int i=0; i < 3; ++i){
            int hbar = (int)(probs[i] * 100); //Calculates the horizontal width of each bar
            DrawRectangle(bx, by+30+i*(bw+gap), hbar, bw, i==a? ORANGE: DARKBLUE); //Draws a rectangle at position
            DrawText(TextFormat("%s %.2f", labels[i], probs[i]), bx + hbar + 8, by+30+i*(bw+gap)+4, 18, DARKGRAY); // Draws text showing the label and probability value
        }

        // Scene rendering
        DrawLine(0, cfg.screenH/2, cfg.screenW, cfg.screenH/2, LIGHTGRAY);
        DrawRectangle((int)env.playerX, cfg.screenH/2 - cfg.targetW/2, cfg.playerW, cfg.targetW, LIGHTBLUE);
        DrawRectangle((int)env.targetX, cfg.screenH/2 - cfg.targetW/2, cfg.targetW, cfg.targetW, RED);
        
        EndDrawing();
    }

    // === Cleanup ===
    CloseWindow();
    return 0;
}
