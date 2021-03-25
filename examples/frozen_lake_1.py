"""
In this example, we learn how to instantiate an environment
and sample a random policy. We can render the environment
in the GUI or in terminal.
Finally, we use the reset() function to restore all states
to their initial values.
"""

import time

from erudit import FrozenLake

if __name__ == "__main__":

    frozen_lake = FrozenLake()

    win = False
    attempts = 0

    while not win:
        attempts += 1
        done = False
        frozen_lake.render()
        
        while not done:
            action = frozen_lake.get_action_space().sample()
            feedback = frozen_lake.step(action)
            done = feedback.done
            win = feedback.reward == 1
            frozen_lake.render()
            time.sleep(0.1)

        frozen_lake.reset()

print(f"Agent has reached the goal in {attempts} attempts.")