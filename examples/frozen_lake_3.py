"""
In this example, we train a SARSA agent on the FrozenLake
environment. 
"""

import time

from erudit import FrozenLake
from erudit.control.sarsa import sarsa

if __name__ == "__main__":

    frozen_lake = FrozenLake()
    Q, V, pi, Q_track, pi_track = sarsa(frozen_lake)

    win = False
    attempts = 0

    feedback = frozen_lake.reset()

    frozen_lake.render()
    
    while not win:
        attempts += 1
        done = False
        frozen_lake.render()
        
        while not done:
            action = pi(feedback.observation)
            feedback = frozen_lake.step(action)
            done = feedback.done
            win = feedback.reward == 1
            frozen_lake.render()
            time.sleep(0.1)

        feedback = frozen_lake.reset()

    print(f"Agent has reached the goal in {attempts} attempts.")