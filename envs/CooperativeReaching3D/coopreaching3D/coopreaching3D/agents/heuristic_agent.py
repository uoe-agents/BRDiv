import random
import numpy as np
from enum import IntEnum
import logging
import numpy as np

class Action(IntEnum):
    NONE = 0
    WEST = 1
    EAST = 2
    NORTH = 3
    SOUTH = 4
    UP = 5
    DOWN = 6


_MAX_INT = 999999

class H1(object):
    """
	H1 agent always goes to the closest rewarding goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            dist_to_goal = [
                abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) + abs(agent_pos[2]-goal[2])
                for goal in self.goals
            ]
            min_dist_to_goal = min(dist_to_goal)
            closest_goals = [idx for idx, dist in enumerate(dist_to_goal) if dist==min_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H2(object):
    """
	H2 agent always goes to the furthest rewarding goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            dist_to_goal = [
                abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) + abs(agent_pos[2]-goal[2])
                for goal in self.goals
            ]
            max_dist_to_goal = max(dist_to_goal)
            closest_goals = [idx for idx, dist in enumerate(dist_to_goal) if dist==max_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H3(object):
    """
	H3 agent always goes to the closest optimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            dist_to_goal = [
                (idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) + abs(agent_pos[2]-goal[2]))
                for idx, goal in enumerate(self.goals)
            ]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] == 1.0]
            min_dist_to_goal = min([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==min_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H4(object):
    """
	H4 agent always goes to the furthest optimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            dist_to_goal = [(
                idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) + abs(agent_pos[2]-goal[2])
            ) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] == 1.0]
            max_dist_to_goal = max([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==max_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H5(object):
    """
	H5 agent always goes to the closest suboptimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            dist_to_goal = [(
                idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) + abs(agent_pos[2]-goal[2])
            ) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] != 1.0]
            min_dist_to_goal = min([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==min_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H6(object):
    """
	H6 agent always goes to the furthest suboptimal goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            dist_to_goal = [(
                idx, abs(agent_pos[0]-goal[0]) + abs(agent_pos[1]-goal[1]) + abs(agent_pos[2]-goal[2])
            ) for idx, goal in enumerate(self.goals)]
            valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] != 1.0]
            max_dist_to_goal = max([dist for _, dist in valid_dists])
            closest_goals = [idx for idx, dist in valid_dists if dist==max_dist_to_goal]
            self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H7(object):
    """
	H7 goes to a randomly selected goal
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        if self.target_goal == None:
            self.target_goal = self.goals[np.random.choice(list(range(len(self.goals))), 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H8(object):
    """
	H8 goes to the goal closest to the learner
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        learner_pos = tuple(obs[3:6])

        dist_to_goal = [abs(learner_pos[0]-goal[0]) + abs(learner_pos[1]-goal[1]) + abs(learner_pos[2]-goal[2]) for goal in self.goals]
        min_dist_to_goal = min(dist_to_goal)
        closest_goals = [idx for idx, dist in enumerate(dist_to_goal) if dist==min_dist_to_goal]
        self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)


class H9(object):
    """
	H9 goes to the optimal goal closest to the learner
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        learner_pos = tuple(obs[3:6])

        dist_to_goal = [(
            idx, abs(learner_pos[0] - goal[0]) + abs(learner_pos[1] - goal[1]) + abs(learner_pos[2] - goal[2])
        ) for idx, goal in enumerate(self.goals)]
        valid_dists = [a for a in dist_to_goal if self.rewards[a[0]] == 1.0]
        min_dist_to_goal = min([dist for _, dist in valid_dists])
        closest_goals = [idx for idx, dist in valid_dists if dist == min_dist_to_goal]
        self.target_goal = self.goals[np.random.choice(closest_goals, 1)[0]]

        return self.move_towards(self.target_goal, agent_pos)

class H10(object):
    """
	H10 follows the learner
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def move_towards(self, target_coord, agent_pos):

        if target_coord[0] < agent_pos[0]:
            return Action.WEST
        elif target_coord[1] < agent_pos[1]:
            return Action.NORTH
        elif target_coord[0] > agent_pos[0]:
            return Action.EAST
        elif target_coord[1] > agent_pos[1]:
            return Action.SOUTH
        elif target_coord[2] < agent_pos[2]:
            return Action.UP
        elif target_coord[2] > agent_pos[2]:
            return Action.DOWN
        else:
            return Action.NONE

    def step(self, obs):
        agent_pos = tuple(obs[:3])
        learner_pos = tuple(obs[3:6])

        return self.move_towards(learner_pos, agent_pos)

class H11(object):
    """
	H11 moves randomly
	"""

    def __init__(self, arena_size, goals, rewards):
        self.arena_size = arena_size
        self.goals = goals
        self.rewards = rewards
        self.target_goal = None

    def step(self, obs):
        return np.random.choice([Action.NONE, Action.WEST, Action.EAST, Action.NORTH, Action.SOUTH, Action.UP, Action.DOWN], 1)[0]
