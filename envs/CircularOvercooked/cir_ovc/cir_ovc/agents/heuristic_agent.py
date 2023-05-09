import random
import numpy as np
from enum import IntEnum
import logging
import numpy as np

class Action(IntEnum):
    NONE = 0
    COUNTER_CLOCKWISE = 1
    CLOCKWISE = 2
    PUT_OUTSIDE = 3
    PUT_INSIDE = 4
    TAKE_INSIDE = 5
    TAKE_OUTSIDE = 6
    USE_ITEM = 7

class Collectibles(IntEnum):
    TOMATO = 0
    CHOPPED_TOMATO = 1
    CARROT = 2
    BLENDED_CARROT = 3
    PLATE = 4

class NonCollectibles(IntEnum):
    BLENDER = 5
    KNIFE = 6
    SERVING_AREA = 7

class H1(object):
    """
	H1 agent seeks the nearest food in the clockwise direction,
	picks it up, and continues to move clockwise to the appropriate food processing station,
	where it processes the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[12]
        carrot_loc = obs[14]
        knife_loc, blender_loc = obs[-2], obs[-3]

        tomato_possessed, carrot_possessed = obs[1], obs[3]

        if blender_loc == carrot_loc and agent_pos == blender_loc:
            return Action.USE_ITEM

        if knife_loc == tomato_loc and agent_pos == knife_loc:
            return Action.USE_ITEM

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return Action.CLOCKWISE

            dist_to_items = [(t_item, (i_pos-agent_pos) % self.arena_size if i_pos != self.arena_size else 0)
                for t_item, i_pos in zip(target_items_final, items_loc_final)
            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                return Action.CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if tomato_possessed == 1:
                target_position = knife_loc
            else:
                target_position = blender_loc

            if agent_pos != target_position:
                return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE


class H2(object):
    """
	H2 agent seeks the nearest food in the counter-clockwise direction,
	picks it up, and continues to move counter-clockwise to the appropriate food processing station,
	where it processes the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[12]
        carrot_loc = obs[14]
        knife_loc, blender_loc = obs[-2], obs[-3]

        tomato_possessed, carrot_possessed = obs[1], obs[3]

        if blender_loc == carrot_loc and agent_pos == blender_loc:
            return Action.USE_ITEM

        if knife_loc == tomato_loc and agent_pos == knife_loc:
            return Action.USE_ITEM

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return Action.COUNTER_CLOCKWISE

            dist_to_items = [(t_item, (agent_pos - i_pos) % self.arena_size if i_pos != self.arena_size else 0)
                             for t_item, i_pos in zip(target_items_final, items_loc_final)
                             ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if tomato_possessed == 1:
                target_position = knife_loc
            else:
                target_position = blender_loc

            if agent_pos != target_position:
                return Action.COUNTER_CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H3(object):
    """
	H3 agent seeks the nearest food,
	picks it up, and continues to move to the appropriate food processing station,
	where it processes the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[12]
        carrot_loc = obs[14]
        knife_loc, blender_loc = obs[-2], obs[-3]

        tomato_possessed, carrot_possessed = obs[1], obs[3]

        if blender_loc == carrot_loc and agent_pos == blender_loc:
            return Action.USE_ITEM

        if knife_loc == tomato_loc and agent_pos == knife_loc:
            return Action.USE_ITEM

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return np.random.choice([Action.COUNTER_CLOCKWISE, Action.CLOCKWISE])

            dist_to_items = [(t_item,
                              min((agent_pos - i_pos) % self.arena_size, (i_pos-agent_pos) % self.arena_size)
                              if i_pos != self.arena_size else 0)
                              for t_item, i_pos in zip(target_items_final, items_loc_final)
                            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                if (agent_pos + min_dist) % self.arena_size in items_loc_final:
                    return Action.CLOCKWISE
                else:
                    return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if tomato_possessed == 1:
                target_position = knife_loc
            else:
                target_position = blender_loc

            if agent_pos != target_position:
                if (agent_pos - target_position) % self.arena_size < (target_position-agent_pos) % self.arena_size:
                    return Action.COUNTER_CLOCKWISE
                return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H4(object):
    """
	H4 agent seeks the furthest food in the clockwise direction,
	picks it up, and continues to move clockwise to the appropriate food processing station,
	where it processes the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[12]
        carrot_loc = obs[14]
        knife_loc, blender_loc = obs[-2], obs[-3]

        tomato_possessed, carrot_possessed = obs[1], obs[3]

        if blender_loc == carrot_loc and agent_pos == blender_loc:
            return Action.USE_ITEM

        if knife_loc == tomato_loc and agent_pos == knife_loc:
            return Action.USE_ITEM

        if tomato_possessed == -1 and carrot_possessed == -1:
            if not self.target_goal is None:
                if self.target_goal == "TOMATO" and tomato_loc == -1:
                    self.target_goal = None
                elif self.target_goal == "CARROT" and carrot_loc == -1:
                    self.target_goal = None

            if self.target_goal is None:
                target_items = ["TOMATO", "CARROT"]
                items_loc = [tomato_loc, carrot_loc]

                target_items_final, items_loc_final = [], []
                for _item, i_pos in zip(target_items, items_loc):
                    if i_pos != -1:
                        target_items_final.append(_item)
                        items_loc_final.append(i_pos)

                if len(target_items_final) == 0:
                    collectible_locations = obs[-3:]
                    if not agent_pos in collectible_locations:
                        return Action.NONE
                    return Action.CLOCKWISE

                dist_to_items = [(t_item, (i_pos-agent_pos) % self.arena_size if i_pos != self.arena_size else 0)
                    for t_item, i_pos in zip(target_items_final, items_loc_final)
                ]

                max_dist = -2
                collected_items = []
                for t_item, dist in dist_to_items:
                    if dist > max_dist:
                        max_dist = dist
                        collected_items = [t_item]
                    elif dist == max_dist:
                        collected_items.append(t_item)

                self.target_goal = np.random.choice(collected_items)

            if self.target_goal == "TOMATO":
                dist_to_item = (tomato_loc-agent_pos) % self.arena_size
            elif self.target_goal == "CARROT":
                dist_to_item = (carrot_loc-agent_pos) % self.arena_size

            if dist_to_item != 0:
                return Action.CLOCKWISE

            if self.target_goal == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            elif self.target_goal == "CARROT":
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if tomato_possessed == 1:
                target_position = knife_loc
            else:
                target_position = blender_loc

            if agent_pos != target_position:
                return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H5(object):
    """
	H5 agent seeks the furthest food in the counter-clockwise direction,
	picks it up, and continues to move counter-clockwise to the appropriate food processing station,
	where it processes the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[12]
        carrot_loc = obs[14]
        knife_loc, blender_loc = obs[-2], obs[-3]

        tomato_possessed, carrot_possessed = obs[1], obs[3]

        if blender_loc == carrot_loc and agent_pos == blender_loc:
            return Action.USE_ITEM

        if knife_loc == tomato_loc and agent_pos == knife_loc:
            return Action.USE_ITEM

        if tomato_possessed == -1 and carrot_possessed == -1:

            if not self.target_goal is None:
                if self.target_goal == "TOMATO" and tomato_loc == -1:
                    self.target_goal = None
                elif self.target_goal == "CARROT" and carrot_loc == -1:
                    self.target_goal = None

            if self.target_goal is None:
                target_items = ["TOMATO", "CARROT"]
                items_loc = [tomato_loc, carrot_loc]

                target_items_final, items_loc_final = [], []
                for _item, i_pos in zip(target_items, items_loc):
                    if i_pos != -1:
                        target_items_final.append(_item)
                        items_loc_final.append(i_pos)

                if len(target_items_final) == 0:
                    collectible_locations = obs[-3:]
                    if not agent_pos in collectible_locations:
                        return Action.NONE
                    return Action.COUNTER_CLOCKWISE

                dist_to_items = [(t_item, (agent_pos - i_pos) % self.arena_size if i_pos != self.arena_size else 0)
                                 for t_item, i_pos in zip(target_items_final, items_loc_final)
                                 ]

                max_dist = -2
                collected_items = []
                for t_item, dist in dist_to_items:
                    if dist > max_dist:
                        max_dist = dist
                        collected_items = [t_item]
                    elif dist == max_dist:
                        collected_items.append(t_item)

                self.target_goal = np.random.choice(collected_items)

            if self.target_goal == "TOMATO":
                dist_to_item = (agent_pos - tomato_loc) % self.arena_size
            elif self.target_goal == "CARROT":
                dist_to_item = (agent_pos - carrot_loc) % self.arena_size

            if dist_to_item != 0:
                return Action.COUNTER_CLOCKWISE

            if self.target_goal == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            elif self.target_goal == "CARROT":
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if tomato_possessed == 1:
                target_position = knife_loc
            else:
                target_position = blender_loc

            if agent_pos != target_position:
                return Action.COUNTER_CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H6(object):
    """
	    H6 is similar to H3 except that it chooses a random action 25% of the time
    """

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[12]
        carrot_loc = obs[14]
        knife_loc, blender_loc = obs[-2], obs[-3]

        tomato_possessed, carrot_possessed = obs[1], obs[3]

        if np.random.uniform() <= 0.25:
            return np.random.choice(list(range(8)))

        if blender_loc == carrot_loc and agent_pos == blender_loc:
            return Action.USE_ITEM

        if knife_loc == tomato_loc and agent_pos == knife_loc:
            return Action.USE_ITEM

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return np.random.choice([Action.COUNTER_CLOCKWISE, Action.CLOCKWISE])

            dist_to_items = [(t_item,
                              min((agent_pos - i_pos) % self.arena_size, (i_pos-agent_pos) % self.arena_size)
                              if i_pos != self.arena_size else 0)
                              for t_item, i_pos in zip(target_items_final, items_loc_final)
                            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                if (agent_pos + min_dist) % self.arena_size in items_loc_final:
                    return Action.CLOCKWISE
                else:
                    return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if tomato_possessed == 1:
                target_position = knife_loc
            else:
                target_position = blender_loc

            if agent_pos != target_position:
                if (agent_pos - target_position) % self.arena_size < (target_position-agent_pos) % self.arena_size:
                    return Action.COUNTER_CLOCKWISE
                return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H7(object):
    """
	H7 agent seeks the nearest processed food in the clockwise direction,
	picks it up, and continues to move clockwise to the plate or (if plate is collected) serving counter,
	where it puts the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[13]
        carrot_loc = obs[15]
        plate_loc = obs[16]
        serving_counter_loc = obs[-1]

        tomato_possessed, carrot_possessed = obs[2], obs[4]
        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1 and i_pos != plate_loc and i_pos != serving_counter_loc:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return Action.CLOCKWISE

            dist_to_items = [(t_item, (i_pos-agent_pos) % self.arena_size
                if i_pos != self.arena_size else 0)
                for t_item, i_pos in zip(target_items_final, items_loc_final)
            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                return Action.CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if plate_loc != -1 :
                target_position = plate_loc
            else:
                target_position = serving_counter_loc

            if target_position == self.arena_size:
                return Action.PUT_INSIDE

            if agent_pos != target_position:
                return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE


class H8(object):
    """
	H8 agent seeks the nearest processed food in the counter-clockwise direction,
	picks it up, and continues to move counter-clockwise to put it on a plate or (if plate is collected) serving counter,
	where it puts the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[13]
        carrot_loc = obs[15]
        plate_loc = obs[16]
        serving_counter_loc = obs[-1]

        tomato_possessed, carrot_possessed = obs[2], obs[4]

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1 and i_pos != plate_loc and i_pos != serving_counter_loc:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return Action.COUNTER_CLOCKWISE

            dist_to_items = [(t_item, (agent_pos - i_pos) % self.arena_size if i_pos != self.arena_size else 0)
                             for t_item, i_pos in zip(target_items_final, items_loc_final)
                             ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if plate_loc != -1:
                target_position = plate_loc
            else:
                target_position = serving_counter_loc

            if target_position == self.arena_size:
                return Action.PUT_INSIDE

            if agent_pos != target_position:
                return Action.COUNTER_CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H9(object):
    """
	H9 agent seeks the nearest processed food,
	picks it up, and continues to move to the plate or (if plate unavailable serving counter),
	where it puts the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[13]
        carrot_loc = obs[15]
        plate_loc = obs[16]
        serving_counter_loc = obs[-1]

        tomato_possessed, carrot_possessed = obs[2], obs[4]

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1 and i_pos != plate_loc and i_pos != serving_counter_loc:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return np.random.choice([Action.COUNTER_CLOCKWISE, Action.CLOCKWISE])

            dist_to_items = [(t_item,
                              min((agent_pos - i_pos) % self.arena_size, (i_pos-agent_pos) % self.arena_size)
                              if i_pos != self.arena_size else 0)
                              for t_item, i_pos in zip(target_items_final, items_loc_final)
                            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                if (agent_pos + min_dist) % self.arena_size in items_loc_final:
                    return Action.CLOCKWISE
                else:
                    return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = -1
            if plate_loc != -1:
                target_position = plate_loc
            else:
                target_position = serving_counter_loc

            if target_position == self.arena_size:
                return Action.PUT_INSIDE

            if agent_pos != target_position:
                if agent_pos != target_position:
                    if (agent_pos - target_position) % self.arena_size < (
                            target_position - agent_pos) % self.arena_size:
                        return Action.COUNTER_CLOCKWISE
                    return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H10(object):
    """
	H10 agent seeks the nearest processed food in the clockwise direction,
	picks it up, and continues to move clockwise to the serving counter,
	where it puts the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[13]
        carrot_loc = obs[15]
        plate_loc = obs[16]
        serving_counter_loc = obs[-1]

        tomato_possessed, carrot_possessed = obs[2], obs[4]
        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1 and i_pos != plate_loc and i_pos != serving_counter_loc:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return Action.CLOCKWISE

            dist_to_items = [(t_item, (i_pos-agent_pos) % self.arena_size
                if i_pos != self.arena_size else 0)
                for t_item, i_pos in zip(target_items_final, items_loc_final)
            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                return Action.CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = serving_counter_loc
            if target_position == self.arena_size:
                return Action.PUT_INSIDE

            if agent_pos != target_position:
                return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE


class H11(object):
    """
	H11 agent seeks the nearest processed food in the counter-clockwise direction,
	picks it up, and continues to move counter-clockwise to put it on the serving counter,
	where it puts the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[13]
        carrot_loc = obs[15]
        plate_loc = obs[16]
        serving_counter_loc = obs[-1]

        tomato_possessed, carrot_possessed = obs[2], obs[4]

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1 and i_pos != plate_loc and i_pos != serving_counter_loc:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return Action.COUNTER_CLOCKWISE

            dist_to_items = [(t_item, (agent_pos - i_pos) % self.arena_size if i_pos != self.arena_size else 0)
                             for t_item, i_pos in zip(target_items_final, items_loc_final)
                             ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = serving_counter_loc

            if target_position == self.arena_size:
                return Action.PUT_INSIDE

            if agent_pos != target_position:
                return Action.COUNTER_CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

class H12(object):
    """
	H12 agent seeks the nearest processed food,
	picks it up, and continues to move to the serving counter,
	where it puts the food. Gets out of the way afterwards.
	"""

    def __init__(self, arena_size):
        self.arena_size = arena_size
        self.target_goal = None

    def step(self, obs):
        agent_pos = obs[0]
        tomato_loc = obs[13]
        carrot_loc = obs[15]
        plate_loc = obs[16]
        serving_counter_loc = obs[-1]

        tomato_possessed, carrot_possessed = obs[2], obs[4]

        if tomato_possessed == -1 and carrot_possessed == -1:

            target_items = ["TOMATO", "CARROT"]
            items_loc = [tomato_loc, carrot_loc]

            target_items_final, items_loc_final = [], []
            for _item, i_pos in zip(target_items, items_loc):
                if i_pos != -1 and i_pos != plate_loc and i_pos != serving_counter_loc:
                    target_items_final.append(_item)
                    items_loc_final.append(i_pos)

            if len(target_items_final) == 0:
                collectible_locations = obs[-3:]
                if not agent_pos in collectible_locations:
                    return Action.NONE
                return np.random.choice([Action.COUNTER_CLOCKWISE, Action.CLOCKWISE])

            dist_to_items = [(t_item,
                              min((agent_pos - i_pos) % self.arena_size, (i_pos-agent_pos) % self.arena_size)
                              if i_pos != self.arena_size else 0)
                              for t_item, i_pos in zip(target_items_final, items_loc_final)
                            ]

            min_dist = 100
            collected_items = []
            for t_item, dist in dist_to_items:
                if dist < min_dist:
                    min_dist = dist
                    collected_items = [t_item]
                elif dist == min_dist:
                    collected_items.append(t_item)

            if min_dist != 0:
                if (agent_pos + min_dist) % self.arena_size in items_loc_final:
                    return Action.CLOCKWISE
                else:
                    return Action.COUNTER_CLOCKWISE

            choice_item_collected = np.random.choice(collected_items)
            if choice_item_collected == "TOMATO":
                if tomato_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE
            else:
                if carrot_loc == self.arena_size:
                    return Action.TAKE_INSIDE
                return Action.TAKE_OUTSIDE

        else:
            target_position = serving_counter_loc
            if target_position == self.arena_size:
                return Action.PUT_INSIDE

            if agent_pos != target_position:
                if agent_pos != target_position:
                    if (agent_pos - target_position) % self.arena_size < (
                            target_position - agent_pos) % self.arena_size:
                        return Action.COUNTER_CLOCKWISE
                    return Action.CLOCKWISE
            return Action.PUT_OUTSIDE

        return Action.NONE

