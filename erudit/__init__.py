__version__ = '0.1.0'


from erudit.space.space import Space, DiscreteSpace
from erudit.action import Action, ActionSpace, DiscreteActionSpace
from erudit.observation import Observation, ObservationSpace, DiscreteObservationSpace
from erudit.feedback import Feedback
from erudit.environment.environment import Environment
from erudit.environment.examples.frozen_lake import FrozenLake
import erudit.utils