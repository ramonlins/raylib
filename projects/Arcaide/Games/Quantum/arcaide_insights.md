Insights:
i: I believe there is a issue in my problem. When calculating the return, I am calculating the average and std, and use to normalize the return. if I have 1000 steps with 1 step -1, reaching the spike, and 10 steps with 1 step -1 (hit spike), this seems to be different, and maybe I am inducing the agent to spend more time possible in the env without being hit. This 
    
a: is not a problem it self, but can indeed be a problem since the negative reward for hitting the target is very small.

i: maybe there is another problem. there is only one distance calculated, but the through is that the distance can also be calculated from the oposite way, because in the environment, if reach the boundarys it will continues the movement from the other side, like pac man game.

a: The toroidal distance need to be calculated, because the euclidian distance alone, will only tell one part of the history, so for example, if the agent is far from 700 pixels in a direct distance, it will assume that is to far, and need to traverse all the environment but it can use the opposite way that is closer. the same for spikes.
