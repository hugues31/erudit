from erudit import Space, DiscreteSpace

class Action:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other


class ActionSpace(Space):
    pass

DiscreteActionSpace = DiscreteSpace