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
#define WIDTH 1920                               //1920 FHD
#define HEIGHT 1080                              // 1080 FHD
#define OFFSET (int)(WIDTH*0.0625)
#define EDGE_0FFSET (int)(WIDTH*0.0104)
#define SCREEN_OFFSET_TOP (int)(WIDTH*0.052)
#define SCREEN_OFFSET_BOT (int)(WIDTH*0.052)
#define VISUAL_SCALE (int)(WIDTH*0.0052)
#define TEXT_SPACE (int)(WIDTH*0.0104)           // vertical space between lines
#define SPIKE_MIN_SPEED 100.f
#define SPIKE_MAX_SPEED 255.f
#define SPIKE_W (int)(WIDTH*0.0104)
#define SPIKES_NEAREST_MAX 5
#define SPIKES_MAX 17
#define SPIKES_MIN 100
#define SPAWN_TIME 20.0f
#define QUASAR_W (int)(WIDTH*0.0104)             // agent width
#define POSITRON_W (int)(WIDTH*0.0104)          // target width
#define XDIM 4                                  // relPos, absPos

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
        float fTargetRelXx{0.f};
        float fTargetRelXy{0.f};
        float fTargetRelYx{0.f};
        float fTargetRelYy{0.f};
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
        float flatTorusW = WIDTH/2;
        float flatTorusH = HEIGHT/2;
        float relX;
        float relY;

        float deltaX = target.x - agent.x;

        if(deltaX > flatTorusW){
            relX = deltaX - WIDTH;
        } else if (deltaX < -flatTorusW){
            relX = WIDTH + deltaX;    
        }else{
            relX = deltaX;
        }

        float deltaY = target.y - agent.y;
        
        if(deltaY > flatTorusH){
            relY = deltaY - HEIGHT;
        } else if (deltaY < -flatTorusH){
            relY = HEIGHT + deltaY;    
        }else{
            relY = deltaY;
        }

        // convert xy to cycle space
        float fRelXx = cos(2*PI*relX/WIDTH);
        float fRelXy = sin(2*PI*relX/WIDTH);        
        float fRelYx = cos(2*PI*relY/HEIGHT);
        float fRelYy = sin(2*PI*relY/HEIGHT);

        // float nPosX = (target.x / WIDTH) * 2.f - 1.f;
        // float nPosY = (target.y / HEIGHT) * 2.f - 1.f;

        return {fRelXx, fRelXy, fRelYx, fRelYy};
    }

    VectorXf observe() {
        vector<float> data;
        data.reserve(XDIM); // Corrected reserve size

        auto [fRelXx, fRelXy, fRelYx, fRelYy] = computeAgentMetrics();
        
        debugInfo.fTargetRelXx = fRelXx;
        debugInfo.fTargetRelXy = fRelXy;
        debugInfo.fTargetRelYx = fRelYx;
        debugInfo.fTargetRelYy = fRelYy;
       
        data.push_back(fRelXx);
        data.push_back(fRelXy);
        data.push_back(fRelYx);
        data.push_back(fRelYy);
        
        VectorXf result = Eigen::Map<VectorXf>(data.data(), data.size());

        return result;
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
    int xdim{XDIM}, hid{128}, ydim{ACTION_COUNT}; 
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
        // 1. Define proportional font sizes based on a 1080p height.
        //    (Original Size / 1080.0f)
        int titleFontSize   = HEIGHT * 0.0278f; // Original: 30px
        int mainFontSize    = HEIGHT * 0.0185f; // Original: 20px
        int regularFontSize = HEIGHT * 0.0139f; // Original: 15px
        int smallFontSize   = HEIGHT * 0.0049f; // Original: 10px

        // 2. Define proportional spacing.
        float topMargin       = HEIGHT * 0.0093f; // Original: 10px
        float verticalSpacing = HEIGHT * 0.0185f; // Original: 20px

        // --- TOP UI ---

        // QUANTUM Title (positioned with a static offset from the center)
        DrawText("QUANTUM", WIDTH * 0.463f, topMargin, titleFontSize, UI_COLOR); // Center - 80px

        // Mode Indicators
        DrawText(env.isManual ? "MANUAL" : "AUTO", WIDTH * 0.073f, topMargin, regularFontSize, env.isManual ? LIGHTGRAY : DARKGRAY); // 140px
        DrawText(env.isTraining ? "TRAIN" : "EVAL", WIDTH * 0.125f, topMargin, regularFontSize, env.isTraining ? LIGHTGRAY : DARKGRAY); // 240px

        // Left-side Stats (TIME, SCORE)
        DrawText(TextFormat("TIME: %.2f", env.elapsedTime), EDGE_0FFSET, topMargin, regularFontSize, UI_COLOR);
        DrawText(TextFormat("SCORE: %d", env.agent.score), EDGE_0FFSET, topMargin + verticalSpacing, regularFontSize, UI_COLOR);

        // Right-side Stats (MAX SCORE, NUM OF SPIKES)
        // NOTE: This uses your original formula, which is already proportional.
        DrawText(TextFormat("MAX SCORE: %d", env.maxScore), WIDTH/1.2f - OFFSET, topMargin, regularFontSize, UI_COLOR);
        //DrawText(TextFormat("NUM OF SPIKES: %zu", env.spikes.size()), WIDTH/1.2f - OFFSET, topMargin + verticalSpacing, regularFontSize, UI_COLOR);

        // FPS Counter (positioned from the right edge with a static offset)
        if(env.game.isShowFPS) DrawText(TextFormat("FPS: %d", GetFPS()), WIDTH - EDGE_0FFSET*4, topMargin, regularFontSize, DARKGRAY); // 70px offset

        // --- BOTTOM UI ---

        // Controls Hint Bar
        DrawText("M: manual | T: train | F: fps | R: reset | K: save | L: load | TAB: debug | ESC: quit",
               EDGE_0FFSET, HEIGHT * 0.94f, smallFontSize, DARKGRAY); // 40px from bottom

        // Instructions Section
        float footerStartY    = HEIGHT * 0.94f;   // Base Y position for the footer
        float footerLineHeight = HEIGHT * 0.013f;  // Space between lines in the footer

        // Left Column
        DrawText("Objective: Stabilize the Quantum Field", WIDTH * 0.6f, footerStartY, smallFontSize, DARKGRAY); // 1200px
        DrawText("Controls: Arrow Keys / WASD to move in all directions", WIDTH * 0.6f, footerStartY + footerLineHeight, smallFontSize, DARKGRAY);
        DrawText("Hint: Diagonal movement is faster", WIDTH * 0.6f, footerStartY + footerLineHeight * 2, smallFontSize, DARKGRAY);

        // Right Column
        DrawText("Rules:", WIDTH * 0.833f, footerStartY, smallFontSize, DARKGRAY); // 1600px
        DrawText("- Collect RED energy (core for stabilization)", WIDTH * 0.833f, footerStartY + footerLineHeight, smallFontSize, DARKGRAY);
        DrawText("- Avoid PURPLE SPIKES (they destroy energy)", WIDTH * 0.833f, footerStartY + footerLineHeight * 2, smallFontSize, DARKGRAY);
        DrawText("- Balance movement to keep control of the field", WIDTH * 0.833f, footerStartY + footerLineHeight * 3, smallFontSize, DARKGRAY);

        // Copyright Text
        DrawText("\xC2\xA9 ARCAIDE", WIDTH * 0.468f, HEIGHT * 0.95f, mainFontSize, UI_COLOR); // Center - 60px 
        
        // UI DEBUG
        if(env.game.isDebug) {
            Vector2 agentPosition = {env.agent.x, env.agent.y};
            Vector2 targetPosition = {env.target.x, env.target.y};
            DrawLineV(agentPosition, targetPosition, DARKBROWN);
            DrawText((TextFormat("fxx: %.2f", env.debugInfo.fTargetRelXx)), agentPosition.x, agentPosition.y + 10, 10, WHITE);
            DrawText((TextFormat("fxy: %.2f", env.debugInfo.fTargetRelXy)), agentPosition.x, agentPosition.y + 20, 10, WHITE);
            DrawText((TextFormat("fyx: %.2f", env.debugInfo.fTargetRelYx)), agentPosition.x, agentPosition.y + 30, 10, WHITE);
            DrawText((TextFormat("fyy: %.2f", env.debugInfo.fTargetRelYy)), agentPosition.x, agentPosition.y + 40, 10, WHITE);
            
            //float angleDeg = env.debugInfo.angleAgentTarget;
            //DrawRing(agentPosition, 100.0f, 45.0f, 0, env.state.angleAgentTarget, 36, SKYBLUE);
            //DrawLine(agentPosition.x, agentPosition.y, agentPosition.x + 30, agentPosition.y, LIME); // Horizontal reference line
            //string angleText = "angle: " + to_string(angleDeg) + " ndeg";
            //DrawText(TextFormat("dist: %.2f", env.debugInfo.distAgentTarget), env.agent.x, env.agent.y + 10, 5, WHITE);
            //DrawText(angleText.c_str(), agentPosition.x, agentPosition.y + 40, 10, WHITE);
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
                reward = -1.f;
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

            if(hasFileBeenModified("policy.bin")){
                pol.load("policy.bin");
                cout << "Load new policy !" << endl;
            }
        }

        DrawFrame();
    }
}

float fixed_timestep (1.f / 60.f);

int main(void){
    //SetTargetFPS(60);

    InitWindow(WIDTH, HEIGHT, "QUANTUM STREAM");
    DisableCursor();
    
    if(WIDTH == 1920){
        //ToggleBorderlessWindowed();
        ToggleFullscreen();
    }

    env.initGame();
    env.loadTextures(); 
    
    pol.load("policy.bin");
    bool isRenderMode{true};
    while(!WindowShouldClose()){
        if(IsKeyPressed(KEY_SPACE)) isRenderMode = !isRenderMode;   
        
        if(!isRenderMode){
            Update(fixed_timestep);
        }else{
            float dt = GetFrameTime();
            Update(dt);
        }
    }

    env.unloadTextures();
    CloseWindow();
    return 0;
}
