import numpy as np

from erudit import Environment

def generate_trajectory(pi, env: Environment, max_steps=20):
    """
    Returns a list of Feedback based on a greedy policy
    """
    done = False
    trajectory = []

    while not done:
        state = env.reset()
        for t in range(max_steps):
            action = pi(state)
            feedback = env.step(action)
            trajectory.append(feedback)
            if done:
                break
            if t >= max_steps - 1:
                trajectory = []
                break
            state = feedback.observation
    
    return np.array(trajectory, dtype=np.object)


def generate_trajectory_from_action(select_action, Q, epsilon, env: Environment, max_steps=200):
    """
    Returns a list of Feedback based on a action-selecting strategy
    """

    done = False
    trajectory = []

    while not done:
        feedback = env.reset()

        for t in range(max_steps):
            action = select_action(feedback.observation, Q, epsilon)
            feedback = env.step(action)
            trajectory.append(feedback)
            if done:
                break
            if t >= max_steps - 1:
                trajectory = []
                break
        
        return np.array(trajectory, dtype=np.object)