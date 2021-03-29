import numpy as np

from erudit import Environment
from erudit.utils import decay_schedule


def temporal_difference(
    pi,
    env: Environment,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.3,
    n_episodes=500,
):
    nS = len(env.get_action_space())
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    for e in range(n_episodes):
        feedback = env.reset()
        while not feedback.done:
            action = pi(feedback.observation)
            feedback = env.step(action)
            td_target = feedback.reward + gamma * V[feedback.observation] * (not feedback.done)

            td_error = td_target - V[feedback.old_observation]
            V[feedback.old_observation] = V[feedback.old_observation] + alphas[e] * td_error

        V_track[e] = V
    
    return V, V_track

# TODO: N-step TD, page 153
# TODO: TD-lambda, page 158