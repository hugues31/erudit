from abc import ABC, abstractmethod

from erudit import Action, ActionSpace, Feedback


class Environment(ABC):
    def __init__(self, config: dict = {}):
        self.config = config

    @abstractmethod
    def step(self, action: Action) -> Feedback:
        pass

    @abstractmethod
    def render(self, mode="terminal"):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action_space(self) -> ActionSpace:
        pass