from abc import ABC, abstractmethod

from erudit import Action, ActionSpace, ObservationSpace, Feedback


class Environment(ABC):
    def __init__(self, config: dict = {}):
        self.config = config
        self.steps_limit = self.config.get("steps_limit", 100)

    @abstractmethod
    def step(self, action: Action) -> Feedback:
        pass

    @abstractmethod
    def render(self, mode="terminal"):
        pass

    @abstractmethod
    def reset(self) -> Feedback:
        pass

    @abstractmethod
    def get_action_space(self) -> ActionSpace:
        pass

    @abstractmethod
    def get_observation_space(self) -> ObservationSpace:
        pass