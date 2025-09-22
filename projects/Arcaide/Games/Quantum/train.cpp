/* Entities*/
// Target
// Player
// Enemy

/* NNets*/
// Architecture
// Forward
// Compute Error
// Gradient
// Update

/* RLearning*/
// Markov Decision Process
// Collect Sequences
// Compute Return
// Calculate Value or Gradient 
// Update policy

/* Game*/
// InitWindow
// InitGame
// Update
// DrawFrame
// ReleaseGameObjects

#include<raylib.h>
#include<math.h>

using namespace std;

#define UI_COLOR RED
#define WIDTH 1280
#define HEIGHT 720
#define OFFSET 60
#define EDGE_0FFSET 10
#define QUASAR_W 20
#define SPIKE_W 10
#define POSITRON_W 20
#define SPIKES_MAX 50
#define SCREEN_OFFSET_TOP 80
#define SCREEN_OFFSET_BOT 80

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

inline float clamp(float value, float min, float max) { return value < min ? min : (value > max ? max: value);}

struct Policy {
    int randomAction(){
        return GetRandomValue(0, ACTION_COUNT);
    }

    int imitationAction(){
        bool left = IsKeyDown(KEY_A);
        bool right = IsKeyDown(KEY_D);
        bool up = IsKeyDown(KEY_W);
        bool down = IsKeyDown(KEY_S);
        
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
};

struct Env {
    struct GameState {
        bool isShowFPS{true};
        //NOTE: Will add more buttons
    };

    struct Target {
        float x{0.f};
        float y{0.f};
        float speed{10.f};

        float w{POSITRON_W};
        float h{POSITRON_W};

    };

    struct Agent {
        float x{0.f};
        float y{0.f};
        float speed{200.f};

        float w{QUASAR_W};
        float h{QUASAR_W};

        int score {0};
    };

    struct Spike {
        float x{0.f};
        float y{0.f};
        float angle{0.f};
        float speed{50.f};
        float w{SPIKE_W};
        float h{SPIKE_W};
    };

    Agent agent;
    Target target;
    GameState gameState;
    Spike spikes[SPIKES_MAX] = {0};

    bool isManual{false};
    bool isTraining{true};
    bool isSpikeStable{false};
    float elapsedTime{0.f};

    void initGame(void){
        agent.x = GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
        agent.y = GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);

        target.x = GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
        target.y = GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);

        for(int i = 0; i < SPIKES_MAX; i++){
            spikes[i].x = GetRandomValue(0, WIDTH);
            spikes[i].y = GetRandomValue(0, HEIGHT);
            spikes[i].angle = GetRandomValue(0, 360) * DEG2RAD;
        }
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
        initGame();
        agent.score = 0;
        isSpikeStable = true;
    }
};

static Env env;
static Policy pol;

void DrawFrame(){
    
    BeginDrawing();
    
        ClearBackground(BLACK);
        // UI
        DrawText(env.isManual ? "MANUAL" : "AUTO", 140, 10, 20, env.isManual ? ORANGE : DARKGRAY);
        DrawText(env.isTraining ? "TRAIN" : "EVAL", 240, 10, 20, env.isTraining ? ORANGE : DARKGRAY);
        DrawText("M: toggle manual | T: toggle training | F: enable/disable fps | R: reset | S: save | L: load | ESC: quit", EDGE_0FFSET, HEIGHT-40, 10, DARKGRAY);
        DrawText(TextFormat("TIME: %.2f", env.elapsedTime), EDGE_0FFSET, EDGE_0FFSET, 15, UI_COLOR);
        DrawText(TextFormat("SCORE: %d", env.agent.score), EDGE_0FFSET, EDGE_0FFSET+20, 15, UI_COLOR);
        DrawText(TextFormat("MAX SCORE: %d", 1000), WIDTH/2 - OFFSET, EDGE_0FFSET, 15, UI_COLOR);

        if(env.gameState.isShowFPS) DrawText(TextFormat("FPS: %d", GetFPS()), WIDTH-EDGE_0FFSET-70, EDGE_0FFSET, 15, UI_COLOR);

        DrawText("\xC2\xA9 ARCAIDE", WIDTH/2 - OFFSET, HEIGHT-50, 20, UI_COLOR);
        DrawRectangle(env.agent.x, env.agent.y, env.agent.w, env.agent.h, WHITE);
        DrawRectangle(env.target.x, env.target.y, env.target.w, env.target.h, RED);
        // NOTE: How to draw the positrons ("Meteors")
        for(const auto& spike: env.spikes){
            DrawRectangle(spike.x, spike.y, spike.w, spike.h, VIOLET);    
        }

        // NOTE: Guide center of screen (uncomment to debug)
        DrawLine((WIDTH/2), 0, WIDTH/2, HEIGHT, LIGHTGRAY);
        DrawLine(0, HEIGHT/2, WIDTH, HEIGHT/2, LIGHTGRAY);
        DrawLine(0, SCREEN_OFFSET_TOP, WIDTH, SCREEN_OFFSET_TOP, LIGHTGRAY);
        DrawLine(0, HEIGHT-SCREEN_OFFSET_BOT, WIDTH, HEIGHT-SCREEN_OFFSET_BOT, LIGHTGRAY);

    EndDrawing();   
}

void Update(float dt){
    int a{0};
    env.elapsedTime+=dt;

    // Game UI
    if(IsKeyPressed(KEY_F)) env.gameState.isShowFPS = !env.gameState.isShowFPS; // inversion of previous state
    if(IsKeyPressed(KEY_T)) env.isTraining = !env.isTraining;
    if(IsKeyPressed(KEY_M)) env.isManual = !env.isManual;
    if(IsKeyPressed(KEY_R)) env.reset();
    
    Rectangle AgentRect = {env.agent.x, env.agent.y, env.agent.w, env.agent.h};
    Rectangle TargetRect = {env.target.x, env.target.y, env.target.w, env.target.h};
    Rectangle SpikesRect[SPIKES_MAX] = {0};

    for(auto& spike: env.spikes){
        spike.x += cos(spike.angle) * spike.speed * dt;
        spike.y += sin(spike.angle) * spike.speed * dt;

        if(!env.isSpikeStable){
            spike.speed += dt * 3; //increase speed with time
            spike.speed = min(spike.speed, 1000.f);
        }else{
            spike.speed = 50.f;
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
            env.reset();
        }   
        // Collsion target x spikes
        if (CheckCollisionRecs(TargetRect, SpikesRect)) {
            env.target.x = GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
            env.target.y = GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);
        }
    }

    // Collsion agent x target
    if (CheckCollisionRecs(AgentRect, TargetRect)){
        env.target.x = GetRandomValue(SCREEN_OFFSET_TOP, WIDTH);
        env.target.y = GetRandomValue(SCREEN_OFFSET_TOP, HEIGHT - SCREEN_OFFSET_BOT);
        env.agent.score +=1 ;
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
    if(env.isManual){
        a = pol.imitationAction();
    }else{
        a = pol.randomAction();
    }
    
    // Act
    env.step(a, dt);
    
    DrawFrame();       
}


int main(void){

    InitWindow(WIDTH, HEIGHT, "QUANTUM");    
    env.initGame();

    while(!WindowShouldClose()){
        float dt = GetFrameTime();
        Update(dt);
    }

    return 0;
}
