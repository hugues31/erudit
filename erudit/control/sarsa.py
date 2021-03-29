"""
SARSA agent for evaluating action-value function Q
"""

import numpy as np

from erudit import Environment
from erudit.utils import generate_trajectory_from_action, decay_schedule


def sarsa(
    env: Environment,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    init_epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay_ratio=0.9,
    n_episodes=3_000,
):
    nS = len(env.get_observation_space())
    nA = len(env.get_action_space())
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # epsilon-greedy action selection
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random_sample() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in range(n_episodes):
        feedback = env.reset()
        state = feedback.observation

        action = select_action(state, Q, epsilons[e])

        while not feedback.done:
            feedback = env.step(action)
            next_state = feedback.observation
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = feedback.reward + gamma * Q[next_state][next_action] * (not feedback.done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error
            state = next_state
            action = next_action


        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track

