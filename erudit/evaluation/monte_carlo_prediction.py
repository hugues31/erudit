"""
Monte Carlo Prediction is use to predict a policy value (= quality) by
running several simulations/trajectory and estimate the state-value function.
(averaging value from states)
"""

import numpy as np

from erudit import Environment
from erudit.utils import generate_trajectory, decay_schedule

def monte_carlo_prediction(pi, env: Environment, gamma=1.0, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.3, n_episodes=500, max_steps=100, first_visit=True):
    nS = len(env.get_action_space())
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    for e in range(n_episodes):
        trajectory = generate_trajectory(pi, env, max_steps)
        visited = np.zeros(nS, dtype=np.bool)

        for t, feedback in enumerate(trajectory):
            state = feedback.observation
            if visited[state] and first_visit:
                continue
        visited[state] = True
        n_steps = len(trajectory[t:])
        G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
        V[state] = V[state] + alphas[e] * (G - V[state])
        
    V_track[e] = V

    return V.copy(), V_track
