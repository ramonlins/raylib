#include <raylib.h>
#include <cmath>
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include "../../../../src/external/eigen/Eigen/Dense"

using Eigen::VectorXf;
using namespace std;

#define UI_COLOR RED
#define WIDTH 1920 //1920 FHD
#define HEIGHT 1080 // 1080 FHD
#define OFFSET 120
#define EDGE_0FFSET 10
#define QUASAR_W 20
#define SPIKE_W 15
#define POSITRON_W 20
#define SPIKES_MAX 100
#define SCREEN_OFFSET_TOP 100
#define SCREEN_OFFSET_BOT 100
#define VISUAL_SCALE 10.0f
#define TEXT_SPACE 20   // vertical space between lines
#define SPIKE_MIN_SPEED 100.f
#define SPIKE_MAX_SPEED 255.f
#define SPAWN_TIME 20.0f

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

int LoadMaxScore(const string& filename) {
    int maxScore = 0;
    
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

struct Policy {
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

    // int sampleAction(State& state){

    // }

};

struct Env {
    struct Game {
        bool isShowFPS{false};
        bool isDebug{false};
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

    struct State {
        float relPosAgentTargetX{0.f};
        float relPosAgentTargetY{0.f};
        float distAgentTarget{0.f};
        float angleAgentTarget{0.f};

        array<Spike, 1> nearestSpikes;

    };
    
    Game game;
    Target target;
    Agent agent;
    vector<Spike> spikes;
    State state{};
    
    Texture2D spikeTexture{};
    
    bool isManual{true};
    bool isTraining{true};
    bool isSpikeStable{false};
    
    float elapsedTime{0.f};
    float lastSpawn{0.f};
    float lastTarget{0.f};
    float spriteScale{120.0f};
    
    int scoreOffset{1};
    int maxScore;
    
    tuple<float, float, float, float> computeAgentMetrics(){
        float relX = agent.x - target.x;
        float relY = agent.y - target.y;
        float dist = sqrt(relX*relX + relY*relY);
        float angle = (float)atan2((double)relY, (double)relX);
        
        return {relX, relY, dist, angle};
    }

    State observe(){
        State state;

        auto [relX, relY, dist, angle] = computeAgentMetrics();

        state.relPosAgentTargetX = relX;
        state.relPosAgentTargetY = relY;
        state.distAgentTarget = dist;
        state.angleAgentTarget = angle;
        
        struct SpikeDistancePair{
            int originalIndex;
            float distance;
        };

        vector<SpikeDistancePair> sortedSpikes;
        
        for(int i = 0; i < spikes.size(); ++i){
            const auto& spike = spikes[i];
            
            float relX = agent.x - spike.x;
            float relY = agent.y - spike.y;
            float dist = sqrt(relX*relX + relY*relY);
            float angle = (float)atan2((double)relY, (double)relX);

            sortedSpikes.push_back({i, dist});
        }
        
        // NOTE: Beter understand the logic here
        sort(sortedSpikes.begin(), sortedSpikes.end(), [](const auto& a, const auto& b){
            return a.distance < b.distance;
        });

        for (int i = 0; i < 1; ++i) {
            if (i < sortedSpikes.size()) {
                int spikeIndex = sortedSpikes[i].originalIndex;
                const auto& spike = spikes[spikeIndex];

                float relX = agent.x - spike.x;
                float relY = agent.y - spike.y;
                
                state.nearestSpikes[i] = Spike{
                    spike.x,                        // x (default value)
                    spike.y,
                    spike.speed,
                    spike.w,
                    spike.h,
                    sortedSpikes[i].distance,
                    spike.distTarget,
                    std::atan2(relY, relX),
                    spike.angleTarget,
                    true                        // isActive (default value)
                };

            } else {
                // If there are fewer than 5 spikes, fill the rest of the array with inactive/default data
                state.nearestSpikes[i].isActive = false;
            }
        }

        return state;
    }
    
    void initSpikes(){
        for(int i = 0; i < 10; i++){
            spikes.push_back(Spike{});
            spikes[i].x = (float)GetRandomValue(0, WIDTH);
            spikes[i].y = (float)GetRandomValue(0, HEIGHT);
            spikes[i].angleAgent = (float)GetRandomValue(0, 360) * DEG2RAD;
            spikes[i].speed = (float)GetRandomValue(0, SPIKE_MAX_SPEED);
        }
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
        initGame();
        agent.score = 0;
        isSpikeStable = true;
        
    }
};

static Env env;
static Policy pol;

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

void DrawFrame(){
    
    BeginDrawing();
        
        ClearBackground(BLACK);
        // UI TOP
        DrawText("QUANTUM FIELD", WIDTH/2 - 120, 10, 30, RED);
        DrawText(env.isManual ? "MANUAL" : "AUTO", 140, 10, 15, env.isManual ? LIGHTGRAY : DARKGRAY);
        DrawText(env.isTraining ? "TRAIN" : "EVAL", 240, 10, 15, env.isTraining ? LIGHTGRAY : DARKGRAY);
        DrawText(TextFormat("TIME: %.2f", env.elapsedTime), EDGE_0FFSET, EDGE_0FFSET, 15, UI_COLOR);
        DrawText(TextFormat("SCORE: %d", env.agent.score), EDGE_0FFSET, EDGE_0FFSET+20, 15, UI_COLOR);
        DrawText(TextFormat("MAX SCORE: %d", env.maxScore), WIDTH/1.2 - OFFSET, EDGE_0FFSET, 15, UI_COLOR);
        DrawText(TextFormat("NUM OF SPIKES: %d", env.spikes.size()), WIDTH/1.2 - OFFSET, EDGE_0FFSET+20, 15, UI_COLOR);
        if(env.game.isShowFPS) DrawText(TextFormat("FPS: %d", GetFPS()), WIDTH-EDGE_0FFSET-70, EDGE_0FFSET, 15, DARKGRAY);
        // UI DOWN
        DrawText("M: toggle manual | T: toggle training | F: enable/disable fps | R: reset | S: save | L: load | D: debug | ESC: quit", EDGE_0FFSET, HEIGHT-40, 10, DARKGRAY);
        DrawText("Objective: Stabilize the Quantum Field", WIDTH/1.6, HEIGHT-SCREEN_OFFSET_TOP+50, 10, DARKGRAY);
        DrawText("Controls: Arrow Keys / WASD to move in all directions", WIDTH/1.6, HEIGHT-SCREEN_OFFSET_TOP+60, 10, DARKGRAY);
        DrawText("Hint: Diagonal movement is faster", WIDTH/1.6, HEIGHT-SCREEN_OFFSET_TOP+70, 10, DARKGRAY);
        DrawText("Rules:", WIDTH/1.2, HEIGHT-SCREEN_OFFSET_TOP+50, 10, DARKGRAY);
        DrawText("- Collect RED energy (core for stabilization)",WIDTH/1.2 , HEIGHT-SCREEN_OFFSET_TOP+60, 10, DARKGRAY);
        DrawText("- Avoid PURPLE SPIKES (they destroy energy)",WIDTH/1.2 , HEIGHT-SCREEN_OFFSET_TOP+70, 10, DARKGRAY);
        DrawText("- Balance movement to keep control of the field",WIDTH/1.2 , HEIGHT-SCREEN_OFFSET_TOP+80, 10, DARKGRAY);
        DrawText("\xC2\xA9 ARCAIDE", WIDTH/2 - 60, HEIGHT-SCREEN_OFFSET_TOP+50, 20, UI_COLOR);        
        
        // UI DEBUG
        if(env.game.isDebug) {
            Vector2 agentPosition = {env.agent.x, env.agent.y};
            Vector2 targetPosition = {env.target.x, env.target.y};
            DrawLineV(agentPosition, targetPosition, DARKBROWN);
            float angleDeg = env.state.angleAgentTarget * RAD2DEG;
            //DrawRing(agentPosition, 100.0f, 45.0f, 0, env.state.angleAgentTarget, 36, SKYBLUE);
            DrawLine(agentPosition.x, agentPosition.y, agentPosition.x + 30, agentPosition.y, LIME); // Horizontal reference line
            string angleText = "Angle: " + to_string(angleDeg) + " deg";
            DrawText(angleText.c_str(), agentPosition.x + 50, agentPosition.y + 10, 10, WHITE);
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
                CLITERAL(Color) speedColor = {(unsigned char)max(50,(int)spike.speed/2), 0, (unsigned char)max(50,(int)spike.speed), 255};
                if(env.game.isDebug) DrawText(TextFormat("speed: %.2f", spike.speed), spike.x-100, spike.y, 5, UI_COLOR);
                Rectangle spikeRect = {
                    spike.x - (spike.w * (VISUAL_SCALE - 1) / 2),  // Center the scaled sprite
                    spike.y - (spike.h * (VISUAL_SCALE - 1) / 2),
                    spike.w * VISUAL_SCALE, 
                    spike.h * VISUAL_SCALE
                };
                Vector2 spikeRectCenter = {spike.w/2, spike.h/2};
                DrawSprite(env.spikeTexture, spikeRectSprite, spikeRect, spikeRectCenter, speedColor);
            }

            for(const auto& spike: env.state.nearestSpikes){
                if(spike.isActive){
                    if(env.game.isDebug) {
                        Vector2 agentPosition = {env.agent.x, env.agent.y};
                        Vector2 spikePosition = {spike.x, spike.y};
                        DrawLineV(agentPosition, spikePosition, PINK);
                        float angleDeg = spike.angleAgent * RAD2DEG;
                        DrawLine(spike.x, spike.y, spike.x + 30, spike.y, LIME); // Horizontal reference line
                        string angleText = "Angle: " + to_string(angleDeg) + " deg";
                        DrawText(angleText.c_str(), spike.x + 50, spike.y + 10, 10, WHITE);
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

//NOW
void Update(float dt){
    int a{0};
    
    // Time elapsed
    env.elapsedTime+=dt;
    if(env.elapsedTime - env.lastSpawn > 5){
        if(env.spikes.size() < SPIKES_MAX){
            for(int i = 0; i < 10; i++){
                env.spikes.push_back(Env::Spike{});

                Env::Spike& newSpike = env.spikes.back(); 

                newSpike.x = (float)GetRandomValue(0, WIDTH);
                newSpike.y = (float)GetRandomValue(0, HEIGHT);
                newSpike.angleAgent = (float)GetRandomValue(0, 360) * DEG2RAD;
                newSpike.speed = (float)GetRandomValue(1, SPIKE_MIN_SPEED);
            }
            env.lastSpawn = env.elapsedTime;
        }
    }

    // After 5 mins "BIG BOSS"
    if((env.spikes.size() >= 20) && env.spikes.size() < 60){
        env.target.textureColor = YELLOW;
        env.scoreOffset = 10;
    }else if((env.spikes.size() >= 60) && env.spikes.size() < 80){
        env.target.textureColor = GOLD;
        env.scoreOffset = 20;
    }else if(env.spikes.size() >= 80){
        env.target.textureColor = RED;
        env.scoreOffset = 100;
    }

    int length = env.spikes.size();
    
    // Game UI
    if(IsKeyPressed(KEY_F)) env.game.isShowFPS = !env.game.isShowFPS;
    if(IsKeyPressed(KEY_TAB)) env.game.isDebug = !env.game.isDebug;
    if(IsKeyPressed(KEY_T)) env.isTraining = !env.isTraining;
    if(IsKeyPressed(KEY_M)) env.isManual = !env.isManual;
    if(IsKeyPressed(KEY_R)) env.reset();
    
    Rectangle AgentRect = {env.agent.x, env.agent.y, env.agent.w, env.agent.h};
    Rectangle TargetRect = {env.target.x, env.target.y, env.target.w, env.target.h};
    //Rectangle SpikesRect[length] = {0};

    for(auto& spike: env.spikes){
        spike.x += cos(spike.angleAgent) * spike.speed * dt;
        spike.y += sin(spike.angleAgent) * spike.speed * dt;

        if(!env.isSpikeStable){
            if(spike.speed < SPIKE_MAX_SPEED){
                spike.speed += dt * 2; //increase speed with time
            }
        }else{
            spike.speed = (float)GetRandomValue(1, SPIKE_MAX_SPEED);
        }

        // Screen wrapping spikes
        if(spike.x > WIDTH) spike.x = 0.f;
        if(spike.x < 0) spike.x = WIDTH;
        if(spike.y > HEIGHT - SCREEN_OFFSET_BOT) spike.y = (float)SCREEN_OFFSET_TOP;
        if(spike.y < SCREEN_OFFSET_TOP) spike.y =(float)(HEIGHT - SCREEN_OFFSET_BOT);
        
        Rectangle SpikesRect = {spike.x, spike.y, spike.w, spike.h};
        
        // // Collision agent x spikes
        // if (CheckCollisionRecs(AgentRect, SpikesRect)){
        //     env.elapsedTime = 0.f;
        //     env.reset();
        // }   
        // Collsion target x spikes
        if (CheckCollisionRecs(TargetRect, SpikesRect)) {
            env.target.x = (float)GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
            env.target.y = (float)GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);
        }
    }

    // Collsion agent x target
    if (CheckCollisionRecs(AgentRect, TargetRect)){
        env.target.x = (float)GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
        env.target.y = (float)GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);
        env.agent.score += env.scoreOffset ;
        env.isSpikeStable = true;   
    }else{
        env.isSpikeStable = false;
    }
   
    // Screen wrapping agent
    if(env.agent.x > WIDTH) env.agent.x = 0.f;
    if(env.agent.x < 0) env.agent.x = WIDTH;
    if(env.agent.y > HEIGHT - SCREEN_OFFSET_BOT) env.agent.y = (float)SCREEN_OFFSET_TOP;
    if(env.agent.y < SCREEN_OFFSET_TOP) env.agent.y =(float)(HEIGHT - SCREEN_OFFSET_BOT);

    // Observe
    env.state = env.observe();

    if(env.isManual){
        a = pol.imitationAction();
    }else{
        a = pol.randomAction();
    }
    
    // Act
    env.step(a, dt);

    if(env.agent.score > env.maxScore){
        SaveMaxScore("max_score.txt", env.agent.score);
        env.maxScore = env.agent.score;
    }

    DrawFrame();    
}

int main(void){

    InitWindow(1920, 1080, "QUANTUM");
    ToggleBorderlessWindowed();
    //ToggleFullscreen();           // switch to fullscreen mode

    env.initGame();
    env.loadTextures();
    
    while(!WindowShouldClose()){
        float dt = GetFrameTime();
        Update(dt);
    }

    env.unloadTextures();
    CloseWindow();
    return 0;
}
