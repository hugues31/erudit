"""
In this example, we learn how to create different strategies.
"""

import time
import numpy as np

from erudit import FrozenLake


def pure_exploitation(env, n_episodes=5_000):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Pure exploitation"
    for e in range(n_episodes):
        action = np.argmax(Q)
        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions


def pure_exploration(env, n_episodes=5_000):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Pure exploration"
    for e in range(n_episodes):
        action = env.get_action_space().sample()
        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions


def epsilon_greedy(env, epsilon=0.01, n_episodes=5_000):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Epsilon-Greedy strategy"
    for e in range(n_episodes):
        if np.random.random_sample() > epsilon:
            action = np.argmax(Q)

        else:
            action = env.get_action_space().sample()

        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions


def linearly_decaying_epsilon_greedy(
    env, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.05, n_episodes=5_000
):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Linearly decaying epsilon greedy {} {} {}".format(
        init_epsilon, min_epsilon, decay_ratio
    )

    for e in range(n_episodes):
        decay_episodes = n_episodes * decay_ratio

        epsilon = 1 - e / decay_episodes
        epsilon *= init_epsilon - min_epsilon
        epsilon += min_epsilon
        epsilon = np.clip(epsilon, min_epsilon, init_epsilon)

        if np.random.random_sample() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions


def exponentially_decaying_epsilon_greedy_strategy(
    env, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.1, n_episodes=5_000
):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    decay_episodes = int(n_episodes * decay_ratio)
    rem_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes)
    epsilons *= init_epsilon - min_epsilon
    epsilons += min_epsilon
    epsilons = np.pad(epsilons, (0, rem_episodes), 'edge')


    name = "Exponentially decaying epsilon greedy {} {} {}".format(
        init_epsilon, min_epsilon, decay_ratio
    )

    for e in range(n_episodes):
        decay_episodes = n_episodes * decay_ratio

        if np.random.random_sample() > epsilons[e]:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions


def optimistic_initialization(
    env, optimistic_estimate=1.0, initial_count=100, n_episodes=5_000
):
    len_action_space = len(env.get_action_space())
    Q = np.full((len_action_space), optimistic_estimate, dtype=np.float64)
    N = np.full((len_action_space), initial_count, dtype=np.float64)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Optimistic initialization strategy {} {}".format(
        optimistic_estimate, initial_count
    )

    for e in range(n_episodes):
        action = np.argmax(Q)
        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions


def softmax(env, init_temp=1000.0, min_temp=0.01, decay_ratio=0.04, n_episodes=5_000):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Softmax {} {} {}".format(init_temp, min_temp, decay_ratio)

    for e in range(n_episodes):
        decay_episodes = n_episodes * decay_ratio
        temp = 1 - e / decay_episodes
        temp *= init_temp - min_temp
        temp += min_temp
        temp = np.clip(temp, min_temp, init_temp)
        scaled_Q = Q / temp
        norm_Q = scaled_Q - np.max(scaled_Q)
        exp_Q = np.exp(norm_Q)
        probs = exp_Q / np.sum(exp_Q)
        
        assert np.isclose(probs.sum(), 1.0)
        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action


    return name, returns, Qe, actions

def upper_confidence_bound(env, c=2, n_episodes=5_000):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "UCB {} ".format(c)

    for e in range(n_episodes):
        if e < len(Q):
            action = e
        else:
            U = np.sqrt(c * np.log(e)/N)
            action = np.argmax(Q + U)

        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action

    return name, returns, Qe, actions

def thompson_sampling(env, alpha=1, beta=0, n_episodes=5_000):
    len_action_space = len(env.get_action_space())
    Q = np.zeros(len_action_space)
    N = np.zeros(len_action_space)

    Qe = np.empty((n_episodes, len_action_space))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=int)

    name = "Thompson Sampling {} {}".format(alpha, beta)

    for e in range(n_episodes):
        samples = np.random.normal(loc=Q, scale = alpha/(np.sqrt(N) + beta))

        action = np.argmax(samples)

        feedback = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (feedback.reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = feedback.reward
        actions[e] = action
    
    return name, returns, Qe, actions
    
if __name__ == "__main__":
    # np.set_printoptions(threshold=np.inf)

    frozen_lake = FrozenLake(config={"normal_cell_reward": -0.1})

    name, returns, Qe, actions = pure_exploitation(frozen_lake)
    print(f"The policty {name} has returns {returns} Qe {Qe} and actions {actions}")

    name, returns, Qe, actions = pure_exploration(frozen_lake)
    print(f"The policty {name} has returns {returns} Qe {Qe} and actions {actions}")

    name, returns, Qe, actions = epsilon_greedy(frozen_lake)
    print(f"The policty {name} has returns {returns} Qe {Qe} and actions {actions}")