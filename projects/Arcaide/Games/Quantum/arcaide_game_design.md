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
Forwards: v(l) = W(l)*y(l-1)
Activation: z(l) = phi(v(l))
Loss: J'(L) = SIGMA(log pol * Gt)'
Backward:
    dW(L) = J'(L) * z'(L)
	dW(l) = dW(L)*W(L)T (.) y(l)
Gradient: L'/W' = dW(l) * y(l-1)T
Update:
    w(L) = w(L) + lRate.dW(L)
	w(l) = w(l) + lRate.dW(l)

algorithm gradient of policy:
    1. initialize grad_W1, grad_W2, grad_b1, grad_b2
    2. loop over transitions
    2.2 d_logits = (onehot - probs) * G[t];
    2.3	grad_W2 += d_logits * z1.T
    2.4	grad_b2 += d_logits
    2.5	d_hidden = W2.T * d_logits * relu`(h1)
    2.6	grad_W1 += d_hidden * x.T
    2.7	grad_b1 += d_hidden
    3. update rule
    3.1 W2 += alpha * grad_W2; b2 += alpha * grad_b2
    3.2 W1 += alpha * grad_W1; b1 += alpha * grad_b1

    Remember:
    error_for_z1_i = (d_logit_0 * W2_0i) + (d_logit_1 * W2_1i) + (d_logit_2 * W2_2i) + ...

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
