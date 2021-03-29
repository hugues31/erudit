from erudit import Space, DiscreteSpace

class Observation:
    pass

class ObservationSpace(Space):
    pass

class DiscreteObservationSpace(ObservationSpace, DiscreteSpace):
    pass