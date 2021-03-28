"""
Feedback is sent to the agent after each action taken. 
observation contains a collection of observations
reward is a float
done is use to indicate a terminal state if True. If False, the agent can
continue to explore the environment as the episode is not ended.
"""

from erudit import Observation
from erudit import Action

class Feedback:
    def __init__(
        self,
        old_observation: Observation,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
    ):
        self.old_observation = old_observation
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
