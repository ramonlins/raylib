#include <raylib.h>
#include <cmath>
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <random>
#include <filesystem>
#include "../../../../src/external/eigen/Eigen/Dense"

using Eigen::MatrixXf;using Eigen::VectorXf;
using namespace std;

#define UI_COLOR RED
#define WIDTH 720               //1920 FHD
#define HEIGHT 480              // 1080 FHD
#define OFFSET 120
#define EDGE_0FFSET 10
#define SCREEN_OFFSET_TOP 100
#define SCREEN_OFFSET_BOT 100
#define VISUAL_SCALE 10.0f
#define TEXT_SPACE 20           // vertical space between lines
#define SPIKE_MIN_SPEED 100.f
#define SPIKE_MAX_SPEED 255.f
#define SPIKE_W 15
#define SPIKES_NEAREST_MAX 5
#define SPIKES_MAX 17
#define SPIKES_MIN 3
#define SPAWN_TIME 20.0f
#define QUASAR_W 20             // agent width
#define POSITRON_W 20           // target width
#define XDIM 4 + SPIKES_MIN * 5 // 4 (agent features) + 5 (spikes) * 5 (features by spike)

constexpr const char* POLICY_PATH = "policy.bin";

enum Action {
    HOLD,
    LEFT,
    RIGHT,
    UP,
    DOWN,
    LEFT_UP,
    RIGHT_UP,
    LEFT_DOWN,
    RIGHT_DOWN,
    ACTION_COUNT
};

auto seed = random_device{}();
static mt19937 rng(seed);

int LoadMaxScore(const string& filename) {
    int maxScore{0};
    
    ifstream inFile(filename);
    if (inFile.is_open()) {
        inFile >> maxScore; // Read the score
        inFile.close();
    } else {
        std::ofstream outFile(filename);
        if (outFile.is_open()) {
            outFile << 0;
            outFile.close();
        }
    }
    
    return maxScore;
}

void SaveMaxScore(const string& filename, int score) {
    ofstream file(filename);

    if (file.is_open()) {
        file << score;
        file.close();
        cout << "New high score saved: " << score << endl;
    } else {
        cout << "Error: Could not save high score to file." << endl;
    }
}

bool hasFileBeenModified(const string& path) {
    static bool is_first_run = true;
    static filesystem::file_time_type last_write_time;

    if (!filesystem::exists(path)) {
        return false;
    }

    auto current_write_time = filesystem::last_write_time(path);

    if (is_first_run) {
        last_write_time = current_write_time;
        is_first_run = false;
        return false;
    }

    if (current_write_time > last_write_time) {
        last_write_time = current_write_time;
        return true;
    }

    return false;
}

struct Env {
    struct Game {
        bool isShowFPS{false};
        bool isDebug{false};
        bool isPaused{false};
        //NOTE: Will add more buttons
    };

    struct Target {
        float x{0.f};
        float y{0.f};
        float speed{10.f};

        float w{POSITRON_W};
        float h{POSITRON_W};

        Texture2D texture{};
        Color textureColor{GRAY};
    };

    struct Agent {
        float x{0.f};
        float y{0.f};
        float speed{200.f};

        float w{QUASAR_W};
        float h{QUASAR_W};

        int score {0};

        Texture2D texture{};
    };

    struct Spike {
        float x{0.f};
        float y{0.f};
        float speed{0.f};
        float w{SPIKE_W};
        float h{SPIKE_W};
        float distAgent{0.f};
        float distTarget{0.f};
        float angleAgent{0.f};
        float angleTarget{0.f};
        
        bool isActive{false};
    };

    struct DebugInfo {
        float relPosAgentTargetX{0.f};
        float relPosAgentTargetY{0.f};
        float distAgentTarget{0.f};
        float angleAgentTarget{0.f};

        array<Spike, SPIKES_NEAREST_MAX> nearestSpikes;

    };
    
    struct Transition {
        VectorXf s{};
        VectorXf zh1{};
        VectorXf logits{};
        
        int a{0};
        float r{0.f};
    };

    Game game;
    Target target;
    Agent agent;
    vector<Spike> spikes;
    DebugInfo debugInfo;
    
    Texture2D spikeTexture{};
    
    bool isManual{false};
    bool isTraining{true};
    bool isSpikeStable{false};
    
    float elapsedTime{0.f};
    float lastSpawn{0.f};
    float lastTarget{0.f};
    float spriteScale{120.0f};
    float level{40.f};
    
    int scoreOffset{1};
    int maxScore{0};
    
    tuple<float, float, float, float> computeAgentMetrics(){
        float nax = (agent.x / WIDTH) * 2.f - 1.f;
        float nay = (agent.y / HEIGHT) * 2.f - 1.f;
        float ntx = (target.x / WIDTH) * 2.f - 1.f;
        float nty = (target.y / HEIGHT) * 2.f - 1.f;

        float relX = ntx - nax;     // [-2, 2]
        float relY = nty - nay;
        float dist = sqrt(relX*relX + relY*relY);   // sqrt(8) ~2.828
        
        float nrelX = relX / 2.f;
        float nrelY = relY / 2.f; 
        float nDist = dist / sqrt(8.f);
        float nangle = atan2(nrelY, nrelX) / PI;
        
        return {nrelX, nrelY, nDist, nangle};
    }

    VectorXf observe() {
        vector<float> data;
        data.reserve(4 + SPIKES_MIN * 5); // Assuming STATE_DIM is correctly defined

        auto [relX_t, relY_t, dist_t, angle_t] = computeAgentMetrics();
        
        debugInfo.relPosAgentTargetX = relX_t;
        debugInfo.relPosAgentTargetY = relY_t;
        debugInfo.distAgentTarget = dist_t;
        debugInfo.angleAgentTarget = angle_t;

        data.push_back(relX_t);
        data.push_back(relY_t);
        data.push_back(dist_t);
        data.push_back(angle_t);

        struct SpikeInfo {
            float norm_rx, norm_ry, norm_speed, dist, angle;
        };
        vector<SpikeInfo> nearest_spike_features;

        float nax = (agent.x / WIDTH) * 2.f - 1.f;
        float nay = (agent.y / HEIGHT) * 2.f - 1.f;

        vector<pair<float, int>> spike_distances;
        for(int i = 0; i < spikes.size(); i++) {
            float nsx = (spikes[i].x / WIDTH) * 2.f - 1.f;
            float nsy = (spikes[i].y / HEIGHT) * 2.f - 1.f;
            float relX_s = nsx - nax;
            float relY_s = nsy - nay;
            float dist = sqrt(relX_s*relX_s + relY_s*relY_s) / sqrt(8.f);
            spike_distances.push_back({dist, i});
        }

        sort(spike_distances.begin(), spike_distances.end());

        for (int i = 0; i < SPIKES_NEAREST_MAX; i++) {
            if (i < spike_distances.size()) {
                const auto& spike = spikes[spike_distances[i].second];
                float nsx = (spike.x / WIDTH) * 2.f - 1.f;
                float nsy = (spike.y / HEIGHT) * 2.f - 1.f;
                float nSpeed = (spike.speed / SPIKE_MAX_SPEED);
                float nDist = spike_distances[i].first;
                float nrelX = (nsx - nax)/ 2.f;
                float nrelY = (nsy - nay)/2.f; 
                float nAngle = atan2(nrelY, nrelX) / PI;

                debugInfo.nearestSpikes[i] = Spike{
                    spike.x, // keep it to visual tracking
                    spike.y, // keep it to visual tracking
                    nSpeed,
                    spike.w,
                    spike.h,
                    nDist,
                    spike.distTarget,
                    nAngle,
                    spike.angleTarget,
                    true // isActive (default value)
                };

                nearest_spike_features.push_back({
                    nrelX,
                    nrelY,
                    nSpeed,
                    nDist,
                    nAngle
                });
            } else {
                // Add empty features if not enough spikes
                nearest_spike_features.push_back({0.f, 0.f, 0.f, 0.f, 0.f});
            }
        }

        for (const auto& features : nearest_spike_features) {
            data.push_back(features.norm_rx);
            data.push_back(features.norm_ry);
            data.push_back(features.norm_speed);
            data.push_back(features.dist);
            data.push_back(features.angle);
        }

        // --- 3. Difficulty Feature ---
        //data.push_back((float)spikes.size() / (float)SPIKES_MAX);
        
        return Eigen::Map<VectorXf>(data.data(), data.size());
    }
    
    void initSpikes(){
        for(int i = 0; i < SPIKES_MIN; i++){
            spikes.push_back(Spike{});
            spikes[i].x = (float)GetRandomValue(0, WIDTH);
            spikes[i].y = (float)GetRandomValue(0, HEIGHT);
            spikes[i].angleAgent = (float)GetRandomValue(0, 360) * DEG2RAD;
            spikes[i].speed = level;
        }
    }

    void stabilizeSpikes(){
        level = 40.f;
        for(auto& spike: spikes){
            spike.speed = level;
        }

        target.textureColor = GRAY;
    }
    
    void initGame(void){
        maxScore = LoadMaxScore("max_score.txt");
        agent.x = (float)GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
        agent.y = (float)GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);

        target.x = (float)GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
        target.y = (float)GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);

        initSpikes();
    }

    void loadTextures() {
        agent.texture = LoadTexture("./assets/glow_white.png");
        target.texture = LoadTexture("./assets/glow_red.png");
        spikeTexture = LoadTexture("./assets/glow_red.png");
    }

    void unloadTextures() {
        UnloadTexture(agent.texture);
        UnloadTexture(target.texture);
        UnloadTexture(spikeTexture);
    }
    
    //TODO: Need to add diagonal actions
    void step(int a, float dt){
        switch (a)
        {
        case LEFT:
            agent.x -= agent.speed * dt;
            break;
        case RIGHT:
            agent.x += agent.speed * dt;
            break;
        case UP:
            agent.y -= agent.speed * dt;
            break;
        case DOWN:
            agent.y += agent.speed * dt;
            break;
        case LEFT_UP:
            agent.x -= agent.speed * dt;
            agent.y -= agent.speed * dt;
            break;
        case RIGHT_UP:
            agent.x += agent.speed * dt;
            agent.y -= agent.speed * dt;
            break;
        case LEFT_DOWN:
            agent.x -= agent.speed * dt;
            agent.y += agent.speed * dt;
            break;
        case RIGHT_DOWN:
            agent.x += agent.speed * dt;
            agent.y += agent.speed * dt;
            break;
        case HOLD:    
        default:
            break;
        }
    }

    void reset(){
        // First, save the score if it's a new high
        if (agent.score > maxScore) {
            SaveMaxScore("max_score.txt", agent.score);
        }
        
        // NOTE: Maybe I need to avoid clearing the spikes to keep the order for
        // neural net input features 
        spikes.clear();
        agent.score = 0;
        elapsedTime = 0.f;
        lastSpawn = 0.f;
        isSpikeStable = true;
        target.textureColor = GRAY;
        level = 40.f;
        initGame();
    }

};

struct Policy {
    int xdim{29}, hid{128}, ydim{ACTION_COUNT}; 
    MatrixXf W1; VectorXf b1;
    MatrixXf W2; VectorXf b2;
    float lr{1e-4f};
    
    VectorXf softmax(const VectorXf& logits){
        float m = logits.maxCoeff(); // stabilized softmax
        VectorXf e = (logits.array() - m).exp();
        return e/e.sum();
    }

    int randomAction(){
        return GetRandomValue(0, ACTION_COUNT);
    }

    int imitationAction(){
        bool left = IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT);
        bool right = IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT);
        bool up = IsKeyDown(KEY_W) || IsKeyDown(KEY_UP);
        bool down = IsKeyDown(KEY_S) || IsKeyDown(KEY_DOWN);
        
        if (left && up) return LEFT_UP;
        if (right && up) return RIGHT_UP;
        if (left && down) return LEFT_DOWN;
        if (right && down) return RIGHT_DOWN;
        
        if (left) return LEFT;
        if (right) return RIGHT;
        if (up) return UP;
        if (down) return DOWN;
        return HOLD;
    }

    Policy() {
        normal_distribution<float> nd(0.f, 1.f);
        W1 = MatrixXf(hid, xdim).unaryExpr([&](float) {return nd(rng);});
        b1 = VectorXf::Zero(hid);

        W2 = MatrixXf(ydim, hid).unaryExpr([&](float) {return nd(rng);});
        b2 = VectorXf::Zero(ydim);
    }
    
    void forward(const VectorXf& x, VectorXf& z1, VectorXf& logits, VectorXf& probs){
        VectorXf v1 = W1 * x + b1;
        z1 = v1.array().max(0.f);

        logits = W2 * z1 + b2;
        probs = softmax(logits);
    }
    
    int sampleAction(const VectorXf& probs){
        discrete_distribution<int> dd(probs.begin(), probs.end());
        return dd(rng);
    }

    void update(const vector<Env::Transition>& traj,float gamma = 0.99f){

        // bellman
        vector<float> G(traj.size());
        float g = 0.f;
        for(int t=traj.size() - 1; t >= 0; t--){
            g = traj[t].r + gamma * g;
            G[t] = g;
        }
        
        float mean = 0.f, sq = 0.f;
        for(float g: G){
            mean += g;
            sq += g*g;
        }

        mean /= G.size();
        float var = sq/G.size() - mean*mean;
        var = max(1e-8f, var);
        float std = sqrt(var);

        for(float&g: G){
            g = (g - mean) / std;
        }
        
        /*
        1. initialize grad_W1, grad_W2, grad_b1, grad_b2;
        2. loop over transitions: 
        2.2 d_logits = (onehot - probs) * G[t];
        2.3	grad_W2 += d_logits * z1.T
        2.4	grad_b2 += d_logits
        2.5	d_hidden = W2.T * d_logits * relu`(h1)
        2.6	grad_W1 += d_hidden * x.T
        2.7	grad_b1 += d_hidden
        3. update rule:
        3.1 W2 += alpha * grad_W2; b2 += alpha * grad_b2
        3.2 W1 += alpha * grad_W1; b1 += alpha * grad_b1

        Remember:
        error_for_z1_i = (d_logit_0 * W2_0i) + (d_logit_1 * W2_1i) + (d_logit_2 * W2_2i) + ...
        */
        MatrixXf grad_W2 = MatrixXf::Zero(W2.rows(), W2.cols());
        VectorXf grad_db2 = VectorXf::Zero(b2.size());
        MatrixXf grad_W1 = MatrixXf::Zero(W1.rows(), W1.cols());
        VectorXf grad_db1 = VectorXf::Zero(b1.size());

        for (size_t t = 0; t < traj.size(); t++) {
            const auto& tr = traj[t];

            VectorXf probs = softmax(tr.logits);
            VectorXf onehot = VectorXf::Zero(probs.size());
            onehot[tr.a] = 1.f;

            VectorXf delta_logits = (onehot - probs) * G[t];

            grad_W2 += delta_logits * tr.zh1.transpose();
            grad_db2 += delta_logits;

            VectorXf delta_hidden = W2.transpose() * delta_logits;
            VectorXf relu_mask = tr.zh1.unaryExpr([](float v){ return v > 0.f ? 1.f : 0.f; });
            delta_hidden = delta_hidden.cwiseProduct(relu_mask);

            grad_W1 += delta_hidden * tr.s.transpose();
            grad_db1 += delta_hidden;
        }

        // --- START: GRADIENT CLIPPING BY NORM ---
        // float total_norm = sqrt(grad_W1.squaredNorm() + grad_db1.squaredNorm() + 
        //                     grad_W2.squaredNorm() + grad_db2.squaredNorm());

        // if (total_norm > 0.5f) {
        //     float scale_factor = 0.5f / total_norm;
        //     grad_W1 *= scale_factor;
        //     grad_db1 *= scale_factor;
        //     grad_W2 *= scale_factor;
        //     grad_db2 *= scale_factor;
        // }

        // --- END: GRADIENT CLIPPING BY NORM ---

        // 3. Apply the (potentially clipped) gradients
        W2 += lr * grad_W2;  b2 += lr * grad_db2;
        W1 += lr * grad_W1;  b1 += lr * grad_db1;
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

void DrawSprite(const Texture2D& textureSprite, Rectangle& sourceRect, Rectangle& destRect, Vector2 origin, Color color){
    DrawTexturePro(
        textureSprite,
        sourceRect,
        destRect,
        origin,
        0.f,
        color 
    );
}

static Env env;
static Policy pol;

void DrawFrame(){
    
    BeginDrawing();
        
        ClearBackground(BLACK);
        // UI TOP
        //DrawText("QUANTUM FIELD", WIDTH/2 - 120, 10, 30, RED);
        DrawText(env.isManual ? "MANUAL" : "AUTO", 140, 10, 15, env.isManual ? LIGHTGRAY : DARKGRAY);
        DrawText(env.isTraining ? "TRAIN" : "EVAL", 240, 10, 15, env.isTraining ? LIGHTGRAY : DARKGRAY);
        DrawText(TextFormat("TIME: %.2f", env.elapsedTime), EDGE_0FFSET, EDGE_0FFSET, 15, UI_COLOR);
        DrawText(TextFormat("SCORE: %d", env.agent.score), EDGE_0FFSET, EDGE_0FFSET+20, 15, UI_COLOR);
        DrawText(TextFormat("MAX SCORE: %d", env.maxScore), WIDTH/1.2 - OFFSET, EDGE_0FFSET, 15, UI_COLOR);
        DrawText(TextFormat("NUM OF SPIKES: %d", env.spikes.size()), WIDTH/1.2 - OFFSET, EDGE_0FFSET+20, 15, UI_COLOR);
        if(env.game.isShowFPS) DrawText(TextFormat("FPS: %d", GetFPS()), WIDTH-EDGE_0FFSET-70, EDGE_0FFSET, 15, DARKGRAY);
        // UI DOWN
        DrawText("M: toggle manual | T: toggle training | F: enable/disable fps | R: reset | K: save | L: load | TAB: debug | ESC: quit", EDGE_0FFSET, HEIGHT-40, 10, DARKGRAY);
        // DrawText("Objective: Stabilize the Quantum Field", WIDTH/1.6, HEIGHT-SCREEN_OFFSET_TOP+50, 10, DARKGRAY);
        // DrawText("Controls: Arrow Keys / WASD to move in all directions", WIDTH/1.6, HEIGHT-SCREEN_OFFSET_TOP+60, 10, DARKGRAY);
        // DrawText("Hint: Diagonal movement is faster", WIDTH/1.6, HEIGHT-SCREEN_OFFSET_TOP+70, 10, DARKGRAY);
        // DrawText("Rules:", WIDTH/1.2, HEIGHT-SCREEN_OFFSET_TOP+50, 10, DARKGRAY);
        // DrawText("- Collect RED energy (core for stabilization)",WIDTH/1.2 , HEIGHT-SCREEN_OFFSET_TOP+60, 10, DARKGRAY);
        // DrawText("- Avoid PURPLE SPIKES (they destroy energy)",WIDTH/1.2 , HEIGHT-SCREEN_OFFSET_TOP+70, 10, DARKGRAY);
        // DrawText("- Balance movement to keep control of the field",WIDTH/1.2 , HEIGHT-SCREEN_OFFSET_TOP+80, 10, DARKGRAY);
        // DrawText("\xC2\xA9 ARCAIDE", WIDTH/2 - 60, HEIGHT-SCREEN_OFFSET_TOP+50, 20, UI_COLOR);        
        
        // UI DEBUG
        if(env.game.isDebug) {
            Vector2 agentPosition = {env.agent.x, env.agent.y};
            Vector2 targetPosition = {env.target.x, env.target.y};
            DrawLineV(agentPosition, targetPosition, DARKBROWN);
            DrawText((TextFormat("rx: %.2f", env.debugInfo.relPosAgentTargetX)), agentPosition.x, agentPosition.y + 20, 10, WHITE);
            DrawText((TextFormat("ry: %.2f", env.debugInfo.relPosAgentTargetY)), agentPosition.x, agentPosition.y + 30, 10, WHITE);
            float angleDeg = env.debugInfo.angleAgentTarget;
            //DrawRing(agentPosition, 100.0f, 45.0f, 0, env.state.angleAgentTarget, 36, SKYBLUE);
            DrawLine(agentPosition.x, agentPosition.y, agentPosition.x + 30, agentPosition.y, LIME); // Horizontal reference line
            string angleText = "angle: " + to_string(angleDeg) + " ndeg";
            DrawText(TextFormat("dist: %.2f", env.debugInfo.distAgentTarget), env.agent.x, env.agent.y + 10, 5, WHITE);
            DrawText(angleText.c_str(), agentPosition.x, agentPosition.y + 40, 10, WHITE);
        }

        // Entities
        BeginBlendMode(BLEND_ADDITIVE);
        
            Rectangle agentRectSprite = {0.f, 0.f, (float)env.agent.texture.width, (float)env.agent.texture.height};
            // NOTE: Why centering work like this ?
            Rectangle agentRect = {
                env.agent.x - (env.agent.w * (VISUAL_SCALE - 1) / 2),  // Center the scaled sprite
                env.agent.y - (env.agent.h * (VISUAL_SCALE - 1) / 2),
                env.agent.w * VISUAL_SCALE, 
                env.agent.h * VISUAL_SCALE
            };        
            Rectangle targetRectSprite = {0.f, 0.f, (float)env.target.texture.width, (float)env.target.texture.height};
            Rectangle targetRect = {
                env.target.x - (env.target.w * (VISUAL_SCALE - 1) / 2),  // Center the scaled sprite
                env.target.y - (env.target.h * (VISUAL_SCALE - 1) / 2),
                env.target.w * VISUAL_SCALE, 
                env.target.h * VISUAL_SCALE
            };
            
            Vector2 agentRectCenter = {env.agent.w/2, env.agent.h/2};
            Vector2 targetRectCenter = {env.target.w/2, env.target.h/2};
            
            DrawSprite(env.agent.texture, agentRectSprite, agentRect, agentRectCenter, BLUE);
            DrawSprite(env.agent.texture, agentRectSprite, agentRect, agentRectCenter, BLUE);
            
            DrawSprite(env.target.texture, targetRectSprite, targetRect, targetRectCenter, env.target.textureColor);
            DrawSprite(env.target.texture, targetRectSprite, targetRect, targetRectCenter, env.target.textureColor);

            Rectangle spikeRectSprite = {0.f, 0.f, (float)env.spikeTexture.width, (float)env.spikeTexture.height};
            for(const auto& spike: env.spikes){
                CLITERAL(Color) speedColor = {(unsigned char)max(100,(int)(spike.speed/2.f)), 0, (unsigned char)max(100,(int)spike.speed), 255};
                Rectangle spikeRect = {
                    spike.x - (spike.w * (VISUAL_SCALE - 1) / 2),  // Center the scaled sprite
                    spike.y - (spike.h * (VISUAL_SCALE - 1) / 2),
                    spike.w * VISUAL_SCALE, 
                    spike.h * VISUAL_SCALE
                };
                Vector2 spikeRectCenter = {spike.w/2, spike.h/2};
                DrawSprite(env.spikeTexture, spikeRectSprite, spikeRect, spikeRectCenter, speedColor);
            }

            for(const auto& spike: env.debugInfo.nearestSpikes){
                if(spike.isActive){
                    if(env.game.isDebug) {
                        // mock features
                        //float nax = (env.agent.x / WIDTH) * 2.f - 1.f;
                        //float nay = (env.spike.y / HEIGHT) * 2.f - 1.f;
                        float nsx = (spike.x / WIDTH) * 2.f - 1.f;
                        float nsy = (spike.y / HEIGHT) * 2.f - 1.f;
                        // float nSpeed = (spike.speed / SPIKE_MAX_SPEED);
                        // float nDist = spike.distAgent;
                        // float nAngle = atan2(nsy - nay, nsx - nax) / PI;

                        Vector2 agentPosition = {env.agent.x, env.agent.y};
                        Vector2 spikePosition = {spike.x, spike.y};
                        DrawLineV(agentPosition, spikePosition, PINK);
                        DrawText(TextFormat("rx: %.2f", nsx), spike.x, spike.y-30, 5, WHITE);
                        DrawText(TextFormat("ry: %.2f", nsy), spike.x, spike.y-40, 5, WHITE);
                        DrawLine(spike.x, spike.y, spike.x + 30, spike.y, LIME); // Horizontal reference line
                        string angleText = "angle: " + to_string(spike.angleAgent) + " deg";
                        DrawText(angleText.c_str(), spike.x, spike.y - 20, 5, WHITE);
                        DrawText(TextFormat("dist: %.2f", spike.distAgent), spike.x, spike.y-10, 5, WHITE);
                        DrawText(TextFormat("speed: %.2f", spike.speed), spike.x, spike.y+10, 5, UI_COLOR);
                
                    }
                }
            }

        EndBlendMode();
        // NOTE: Guide center of screen (uncomment to debug)
        if(env.game.isDebug) {
            DrawLine((WIDTH/2), 0, WIDTH/2, HEIGHT, LIGHTGRAY);
            DrawLine(0, HEIGHT/2, WIDTH, HEIGHT/2, LIGHTGRAY);
            DrawLine(0, SCREEN_OFFSET_TOP, WIDTH, SCREEN_OFFSET_TOP, LIGHTGRAY);
            DrawLine(0, HEIGHT-SCREEN_OFFSET_BOT, WIDTH, HEIGHT-SCREEN_OFFSET_BOT, LIGHTGRAY);
        }
        EndBlendMode();

    EndDrawing();  
}

void Update(float dt){
    int a{0};
    float reward{0.f};
    bool isDone{false}, isTerminated{false};
    static vector<Env::Transition> traj{};
    
    env.elapsedTime+=dt;
    if(env.elapsedTime - env.lastSpawn > 60.f){
        env.lastSpawn = env.elapsedTime;
        isTerminated = true;
    }
    
    // spawn 5 spikes after 30s
    // if(env.elapsedTime - env.lastSpawn > 30.f){
    //     if(env.spikes.size() < SPIKES_MAX){
    //         for(int i = 0; i < 5; i++){
    //             env.spikes.push_back(Env::Spike{});

    //             Env::Spike& newSpike = env.spikes.back(); 

    //             newSpike.x = (float)GetRandomValue(0, WIDTH);
    //             newSpike.y = (float)GetRandomValue(0, HEIGHT);
    //             newSpike.angleAgent = (float)GetRandomValue(0, 360) * DEG2RAD;
    //             newSpike.speed = env.level;
    //         }
    //         env.lastSpawn = env.elapsedTime;
    //     }
    // }

    if((env.elapsedTime >= 5.f) && (env.elapsedTime < 8.f)){
        env.target.textureColor = WHITE;
        env.level = 80;
    }else if((env.elapsedTime >= 8.f) && (env.elapsedTime < 10.f)){
        env.target.textureColor = GOLD;
        env.level = 160;
    }else if((env.elapsedTime >= 10.f)){
        env.target.textureColor = RED;
        env.level = 255;
    }

    // Game UI
    if(IsKeyPressed(KEY_F)) env.game.isShowFPS = !env.game.isShowFPS;
    if(IsKeyPressed(KEY_TAB)) env.game.isDebug = !env.game.isDebug;
    if(IsKeyPressed(KEY_T)) env.isTraining = !env.isTraining;
    if(IsKeyPressed(KEY_M)) env.isManual = !env.isManual;
    if(IsKeyPressed(KEY_R)) env.reset();
    if(IsKeyPressed(KEY_K)) pol.save(POLICY_PATH);
    if(IsKeyPressed(KEY_L)) pol.load(POLICY_PATH);
    if(IsKeyPressed(KEY_P)) env.game.isPaused = !env.game.isPaused;

    if(!env.game.isPaused){
        
        Rectangle AgentRect = {env.agent.x, env.agent.y, env.agent.w, env.agent.h};
        Rectangle TargetRect = {env.target.x, env.target.y, env.target.w, env.target.h};
    
        for(auto& spike: env.spikes){
            spike.x += cos(spike.angleAgent) * spike.speed * dt;
            spike.y += sin(spike.angleAgent) * spike.speed * dt;

            if(!env.isSpikeStable){
                if(spike.speed < SPIKE_MAX_SPEED/2.f){
                    spike.speed += dt * (float)GetRandomValue(1, env.level)*0.2f; //increase speed with time
                }
            }

            // Screen wrapping spikes
            if(spike.x > WIDTH) spike.x = 0.f;
            if(spike.x < 0) spike.x = WIDTH;
            if(spike.y > HEIGHT - SCREEN_OFFSET_BOT) spike.y = (float)SCREEN_OFFSET_TOP;
            if(spike.y < SCREEN_OFFSET_TOP) spike.y =(float)(HEIGHT - SCREEN_OFFSET_BOT);
            
            Rectangle SpikesRect = {spike.x, spike.y, spike.w, spike.h};
            
            // Collision agent x spikes
            if (CheckCollisionRecs(AgentRect, SpikesRect)){
                env.elapsedTime = 0.f;
                reward = -0.1f;
                isDone = true;
            }   
            
            // Collsion target x spikes
            if (CheckCollisionRecs(TargetRect, SpikesRect)) {
                env.target.x = (float)GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
                env.target.y = (float)GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);

                // if(env.spikes.size() < SPIKES_MAX){
                //     env.spikes.push_back(Env::Spike{});

                //     Env::Spike& newSpike = env.spikes.back(); 

                //     newSpike.x = env.target.x + (float)GetRandomValue(0, 100);
                //     newSpike.y = env.target.y + (float)GetRandomValue(0, 100);
                //     newSpike.angleAgent = (float)GetRandomValue(0, 360) * DEG2RAD;
                //     newSpike.speed = env.level;
                // }else{
                spike.speed = (float)env.level;
                // }
            }
        }

        // Collsion agent x target
        if (CheckCollisionRecs(AgentRect, TargetRect)){
            env.target.x = (float)GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
            env.target.y = (float)GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);
            env.agent.score += env.scoreOffset;
            env.isSpikeStable = true;   
            reward = 1.f;
            env.elapsedTime = 0.f;
            //env.elapsedTime = 0.f; // reset time to focus on target
            //env.lastSpawn = env.elapsedTime;
            //isDone = true; // this can used to train first with a easier scenarium;
        }else{
            //if(env.elapsedTime - env.lastSpawn > 10.f){
            env.isSpikeStable = false;
            //reward = -(env.debugInfo.distAgentTarget)*0.01f + 0.01f;
            //reward = 0.f;
            //}
        }
    
        // Screen wrapping agent
        if(env.agent.x > WIDTH) env.agent.x = 0.f;
        if(env.agent.x < 0) env.agent.x = WIDTH;
        if(env.agent.y > HEIGHT - SCREEN_OFFSET_BOT) env.agent.y = (float)SCREEN_OFFSET_TOP;
        if(env.agent.y < SCREEN_OFFSET_TOP) env.agent.y =(float)(HEIGHT - SCREEN_OFFSET_BOT);

        // Observe
        VectorXf s = env.observe();

        VectorXf z1, logits, probs;
        pol.forward(s, z1, logits, probs);

        if(env.isManual){
            a = pol.imitationAction();
        }else{
            a = pol.sampleAction(probs);
        }
        
        // Act
        env.step(a, dt);

        if(env.isTraining){
            traj.push_back({s, z1, logits, a, reward});

            if(isTerminated){
                pol.update(traj);
                traj.clear();
            }

            if(env.isSpikeStable){
                env.stabilizeSpikes();
            }

            if(isDone){
                pol.update(traj);
                env.reset();
                traj.clear();
            }
            
            if(env.agent.score > env.maxScore){
                SaveMaxScore("max_score.txt", env.agent.score);
                pol.save(POLICY_PATH);
                env.maxScore = env.agent.score;
            }

            if(hasFileBeenModified("policy.bin")){
                pol.load("policy.bin");
                cout << "Load new policy !" << endl;
            }
        }else{
            if(isDone){
                env.reset();
                traj.clear();
            }

            if(env.isSpikeStable){
                env.stabilizeSpikes();
            }
        }

        DrawFrame();
    }
}

int main(void){

    InitWindow(WIDTH, HEIGHT, "QUANTUM");
    //ToggleBorderlessWindowed();
    //ToggleFullscreen();           // switch to fullscreen mode
    SetTargetFPS(165);
    env.initGame();
    env.loadTextures();
    
    pol.load("policy.bin");
    
    while(!WindowShouldClose()){
        float dt = GetFrameTime();
        Update(dt);
    }

    env.unloadTextures();
    CloseWindow();
    return 0;
}
