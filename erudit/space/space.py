from abc import ABC, abstractmethod

import numpy as np


class Space:
    def __init__(self):
        pass

    @abstractmethod
    def sample(self):
        """
        Return a randomly sampled observation.
        """
        pass

    @abstractmethod
    def contains(self, x) -> bool:
        """
        Return True if x is contained/valid inside the space. False otherwise.
        """
        pass

    def __contains__(self, x):
        return self.contains(x)


class DiscreteSpace(Space):
    def __init__(self, n):
        assert n > 0
        self.n = n

    def sample(self):
        return np.random.randint(low=0, high=self.n)
