from erudit import Observation

class Feedback:
    def __init__(self, observation: Observation, reward: float, done: bool):
        self.observation = observation
        self.reward = reward
        self.done = done
