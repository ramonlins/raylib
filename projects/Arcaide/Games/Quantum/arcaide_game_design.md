----------GAME----------
Goal: All game needs an objective;
Rules: How player can interact with environment;
Core loop: Repeated movements that a player can do;
Feedback: The return the player receives after a core loop;
Flow: Feedback mantaining the player engaged; e.g: character evolution;
Balance: The game can`t be to easy, boring, or to hard, not good to play;
Reward: How interaction bring consequences, creating tension on the decisions;
Meaning: The need to create a different experiences like a deep conection;
Immersion: Making player feel trully party of the game;
Fun: With all of this, remember, games need to be fun;

----------NN MLP----------
Forwards: v(l) = W(l)*y(l-1)
Activation: z(l) = phi(v(l))
Loss: J = F(d(L) - (z(L)), e.g: F = MSE
DeltaOutput: dW(L) = J'(L) * z'(L)
DeltaHidden: dW(l) = dW(L)*W(l)T (.) z'(l-1)
Gradient: J'/W' = dW(l) * y(l-1)T
Update: W = W - lRate * dW + beta * W(t-1)

---------- RL PG NN---------
MPD: (s, a, r, T, a', pi, gamma)
Rollout: (s, a, r, a')
Return: G = SIGMA(gamma*r)
Bellman: Gt = SIGMA(r + gamma*G_t+1)
Forward: z(l) = W(l)y(l-1)
Loss: J'(L) = SIGMA(log pol * Gt)'
Backward:
    dW(L) = J'(L) * z'(L)
	 dW(l) = dW(L)*W(L)T (.) y(l)
Gradient: L'/W' = dW(l) * y(l-1)T
Update:
    w(L) = w(L) + lRate.dW(L)
	w(l) = w(l) + lRate.dW(l)

-------------- Features Design ----------------
fixed-based-features:
    standard: position, distance and angle (player and target)
    zero padding: for spikes not created yet (active or not active)
    order: distance to player, target (close first)
    relative position: (obstacle_x - player_x) and (obstacle_y - player_y)
    more features: angle, min d(player, spikes), number of surround obstacles
    more nets: attention, gnn
proximity-based features:
    fixed-number: k obstacles
    nearest: sort by distance
    k relative: position, velocity, angle, (player x obstacles) (target x obstacles)
