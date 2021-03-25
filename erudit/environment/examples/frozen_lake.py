import numpy as np
from enum import Enum

from erudit import Environment, Action, ActionSpace, DiscreteActionSpace, Feedback

class Direction(Enum):
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

class State(Enum):
    ROBOT = "⚉"
    NORMAL = " "
    HOLE = "🔾"
    GOAL = "⚐"
    LOST = "⨯"
    WIN = "⚑"

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class FrozenLake(Environment):
    def __init__(self, config: dict = {}):
        super().__init__(config)

        self.current_state = 0
        # 4 possible actions: 0: top, 1: right, 2: bottom, 3: left
        self.action_space = DiscreteActionSpace(4)

        # we create a 4x4 grid
        self.grid_size = 4
        self.holes_states = [Position(1, 1), Position(3, 1), Position(3, 2), Position(0, 3)]
        
        self.reset()

    def get_current_position(self) -> Position:
        for y, col in enumerate(self.states):
            for x, _ in enumerate(col):
                if self.states[x][y] in [State.ROBOT, State.WIN, State.LOST]:
                    return Position(x, y)

    def set_current_position(self, position: Position):
        current_pos = self.get_current_position()
        self.states[current_pos.x][current_pos.y] = State.NORMAL
        # Bounce back if we hit a wall
        x = min(max(position.x, 0), self.grid_size-1)
        y = min(max(position.y, 0), self.grid_size-1)
        self.states[x][y] = State.ROBOT

        if self.is_in_hole():
            self.states[x][y] = State.LOST
        
        elif self.has_reach_goal():
            self.states[x][y] = State.WIN
        
    def is_in_hole(self) -> bool:
        return self.get_current_position() in self.holes_states
    
    def has_reach_goal(self) -> bool:
        goal_pos = Position(self.grid_size - 1, self.grid_size - 1)
        return self.get_current_position() == goal_pos

    def step(self, action: Action) -> Feedback:
        self.time_step += 1

        # Take the action
        action = Direction(action)
        prob = np.random.randint(low=-1, high=2)
        current_pos = self.get_current_position()


        # We take the expected action
        if action == Direction.TOP:
            new_pos = Position(current_pos.x, current_pos.y - 1)

        elif action == Direction.RIGHT:
            new_pos = Position(current_pos.x + 1, current_pos.y)

        elif action == Direction.BOTTOM:
            new_pos = Position(current_pos.x, current_pos.y + 1)

        elif action == Direction.LEFT:
            new_pos = Position(current_pos.x - 1, current_pos.y)

        else:
            raise Exception(f"Action {action} is not valid.")

        # agent slip and instead turns left (if prob == -1) or right (== 1)
        if prob in [-1, 1]:
            # if movement is vertical
            if new_pos.y - current_pos.y != 0:
                new_pos = Position(current_pos.x + prob, current_pos.y)

            # mouvement is horizontal
            else:
                new_pos = Position(current_pos.x, current_pos.y - prob)

        self.set_current_position(new_pos)

        # Compute the feedback
        obs = self.get_current_position()

        if self.is_in_hole():
            reward = -1
            done = True

        elif self.has_reach_goal():
            reward = 1
            done = True

        else:
            reward = 0
            done = False

        return Feedback(obs, reward, done)

    def render(self, mode="terminal"):
        flat_states = []
        for y, col in enumerate(self.states):
            for x, _ in enumerate(col):
                flat_states.append(f" {self.states[x][y].value} ")

        grid = """
            ┌───┬───┬───┬───┐
            │{0}│{1}│{2}│{3}│
            ├───├───├───├───┤
            │{4}│{5}│{6}│{7}│
            ├───├───├───├───┤
            │{8}│{9}│{10}│{11}│
            ├───├───├───├───┤
            │{12}│{13}│{14}│{15}│
            └───┴───┴───┴───┘
        """.format(
            *flat_states
        )

        print(f"           === Time step {self.time_step} ===")
        print(grid)



    def reset(self):
        self.time_step = 0
        self.states = [[State.NORMAL] * self.grid_size for i in range(self.grid_size)]
        self.states[0][0] = State.ROBOT
        self.states[-1][-1] = State.GOAL

        for position in self.holes_states:
            self.states[position.x][position.y] = State.HOLE

    def get_action_space(self) -> ActionSpace:
        return self.action_space


# ┌────────┬────────┬────────┬────────┐
# │ 3.14   │ 3.14   │ 3.14   │ 3.14   │
# │        │        │        │        │
# │        │        │        │        │
# ├────────├────────├────────├────────┤
# │ 3.14   │ 3.14   │ 3.14   │ 3.14   │
# │        │        │        │        │
# │        │        │        │        │
# ├────────├────────├────────├────────┤
# │ 3.14   │ 3.14   │ 3.14   │ 3.14   │
# │        │        │        │        │
# │        │        │        │        │
# ├────────├────────├────────├────────┤
# │ 3.14   │ 3.14   │ 3.14   │ 3.14   │
# │        │        │        │        │
# │        │        │        │        │
# └────────┴────────┴────────┴────────┘