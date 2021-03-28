import numpy as np

from erudit import Environment

def generate_trajectory(pi, env: Environment, max_steps=20):
    """
    Returns a list of Feedback
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