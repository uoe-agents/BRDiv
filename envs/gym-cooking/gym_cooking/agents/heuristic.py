import torch
import numpy as np
from d_network import AgentNetworkFC

class FullActionScheme:

    WALK_UP = 4
    WALK_DOWN = 3
    WALK_RIGHT = 2
    WALK_LEFT = 1

    NO_OP = 0

    INTERACT_PRIMARY = 5
    INTERACT_PICK_UP_SPECIAL = 6
    EXECUTE_ACTION = 7

    WALK_ACTIONS = [WALK_UP, WALK_DOWN, WALK_RIGHT, WALK_LEFT]
    INTERACT_ACTIONS = [INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL, EXECUTE_ACTION]
    ACTIONS = [NO_OP, WALK_LEFT, WALK_RIGHT, WALK_DOWN, WALK_UP, INTERACT_PRIMARY, INTERACT_PICK_UP_SPECIAL, EXECUTE_ACTION]

class Direction:
    NORTH = 3
    SOUTH = 2
    EAST = 1
    WEST = 0
    ANY = -1 # (when direction doesn't matter)


class RandomAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation) -> int:
        return self.action_space.sample()

class DeepAgent:

    def __init__(self, model_path):
        model = torch.load(model_path)
        self.joint_policy = AgentNetworkFC(
            obs_dim = model["linear_obs_rep.0.weight"].shape[1],
            obs_u_dim = 0,
            mid_dim1 = model["linear_obs_rep.2.weight"].shape[0],
            mid_dim2 = model["linear_obs_rep.2.weight"].shape[1],
            gnn_hdim1 = model["gnn.2.weight"].shape[0],
            gnn_hdim2 = model["gnn.2.weight"].shape[1],
            gnn_out_dim = model["gnn.4.weight"].shape[0],
            num_acts = model["act_logits.weight"].shape[0],
            device="cpu",
        )
        self.joint_policy.load_state_dict(model)

    def get_action(self, observation):
        logits = self.joint_policy(torch.Tensor(observation))
        dist = torch.distributions.categorical.Categorical(logits=logits)
        act = dist.sample().item()
        return act

class HAgent:
    def __init__(self):
        self.pos_to_loc_dict = {
            (1, 1): 0,
            (2, 1): 1,
            (3, 1): 2,
            (3, 2): 3,
            (3, 3): 4,
            (3, 4): 5,
            (3, 5): 6,
            (2, 5): 7,
            (1, 5): 8,
            (1, 4): 9,
            (1, 3): 10,
            (1, 2): 11,
            }
        self.loc_to_pos_dict = {v: k for k,v in self.pos_to_loc_dict.items()}
        self.pos_to_nav_dict = {
            (1, 0): [(0, Direction.NORTH)],
            (2, 0): [(1, Direction.NORTH)],
            (3, 0): [(2, Direction.NORTH)],

            (4, 1): [(2, Direction.EAST)],
            (4, 2): [(3, Direction.EAST)],
            (4, 3): [(4, Direction.EAST)],
            (4, 4): [(5, Direction.EAST)],
            (4, 5): [(6, Direction.EAST)],

            (3, 6): [(6, Direction.SOUTH)],
            (2, 6): [(7, Direction.SOUTH)],
            (1, 6): [(8, Direction.SOUTH)],

            (0, 5): [(8, Direction.WEST)],
            (0, 4): [(9, Direction.WEST)],
            (0, 3): [(10, Direction.WEST)],
            (0, 2): [(11, Direction.WEST)],
            (0, 1): [(0, Direction.WEST)],

            (2, 2): [(1, Direction.SOUTH),
                     (3, Direction.WEST),
                     (11, Direction.EAST)],
            (2, 3): [(4, Direction.WEST),
                     (10, Direction.EAST)],
            (2, 4): [(5, Direction.WEST),
                     (7, Direction.NORTH),
                     (9, Direction.EAST)],

            **{pos: [(loc, Direction.ANY)]
               for pos, loc in self.pos_to_loc_dict.items()}
            }

        self.is_left = None

    def parse_obs(self, observation):
        if self.is_left is None:
            self.is_left = ( ((5*observation[18]+1 == 0) and 5*observation[0]==1)
                            or ((5*observation[18]+1 == -2) and 5*observation[0]==3))
        if self.is_left:
            ego_pos_x, ego_pos_y, ego_orient_W, ego_orient_E, ego_orient_S, ego_orient_N = \
                    observation[0:6]
            opp_pos_x, opp_pos_y, opp_orient_W, opp_orient_E, opp_orient_S, opp_orient_N = \
                    observation[6:12]
        else:
            ego_pos_x, ego_pos_y, ego_orient_W, ego_orient_E, ego_orient_S, ego_orient_N = \
                    observation[6:12]
            opp_pos_x, opp_pos_y, opp_orient_W, opp_orient_E, opp_orient_S, opp_orient_N = \
                    observation[0:6]

        ego_orient = (Direction.NORTH*ego_orient_N
                      + Direction.SOUTH*ego_orient_S
                      + Direction.EAST*ego_orient_E
                      + Direction.WEST*ego_orient_W)
        opp_orient = (Direction.NORTH*opp_orient_N
                      + Direction.SOUTH*opp_orient_S
                      + Direction.EAST*opp_orient_E
                      + Direction.WEST*opp_orient_W)
        blender_pos_x, blender_pos_y = observation[12:14]
        carrot_pos_x, carrot_pos_y, carrot_chopped, carrot_mashed = observation[14:18]
        chopboard_pos_x, chopboard_pos_y = observation[18:20]
        deliver_pos_x, deliver_pos_y = observation[20:22]
        plate_pos_x, plate_pos_y = observation[22:24]
        tomato_pos_x, tomato_pos_y, tomato_chopped= observation[24:27]

        obs_dict = {
            "ego": {"x": self.parse_loc(x=ego_pos_x),
                    "y": self.parse_loc(y=ego_pos_y),
                    "orientation": round(ego_orient)},
            "opp": {"x": self.parse_loc(x=opp_pos_x + ego_pos_x),
                    "y": self.parse_loc(y=opp_pos_y + ego_pos_y),
                    "orientation": round(opp_orient)},
            "blender": {"x": self.parse_loc(x=blender_pos_x + ego_pos_x),
                        "y": self.parse_loc(y=blender_pos_y + ego_pos_y)},
            "chopboard": {"x": self.parse_loc(x=chopboard_pos_x + ego_pos_x),
                         "y": self.parse_loc(y=chopboard_pos_y + ego_pos_y)},
            "deliver": {"x": self.parse_loc(x=deliver_pos_x + ego_pos_x),
                        "y": self.parse_loc(y=deliver_pos_y + ego_pos_y)},
            "plate": {"x": self.parse_loc(x=plate_pos_x + ego_pos_x),
                      "y": self.parse_loc(y=plate_pos_y + ego_pos_y)},
            "tomato": {"x": self.parse_loc(x=tomato_pos_x + ego_pos_x),
                       "y": self.parse_loc(y=tomato_pos_y + ego_pos_y),
                       "chopped": bool(tomato_chopped)},
            "carrot": {"x": self.parse_loc(x=carrot_pos_x + ego_pos_x),
                       "y": self.parse_loc(y=carrot_pos_y + ego_pos_y),
                       "chopped": bool(carrot_chopped),
                       "mashed": bool(carrot_mashed)},
        }
        self.state = obs_dict
        return obs_dict

    def parse_loc(self, loc=None, x=None, y=None):
        if loc is not None:
            return round(5*loc[0]), round(7*loc[1])
        if (x is not None) and (y is not None):
            return round(5*x), round(7*y)
        if x is not None:
            return round(5*x)
        if y is not None:
            return round(7*y)

    def pos_to_loc(self, loc):
        return self.pos_to_loc_dict[(loc[0],loc[1])]

    def navigate_to(self, target_loc, target_orientation=None, direction="cw"):
        #
        #  █████
        #  █012█
        #  █B█3█
        #  █A█4█
        #  █9█5█
        #  █876█
        #  █████
        #

        ego_loc = self.pos_to_loc((self.state["ego"]["x"], self.state["ego"]["y"]))
        ego_orientation = self.state["ego"]["orientation"]
        if ego_loc == target_loc:
            if ((target_orientation is None)
                 or (ego_orientation == target_orientation)):
                return FullActionScheme.NO_OP
            else:
                return self.orient(target_orientation)

        if direction == "adapt":
            cw_distance = self.distance_to_loc(target_loc, direction="cw")
            acw_distance = self.distance_to_loc(target_loc, direction="acw")
            direction = "cw" if cw_distance < acw_distance else "acw"


        if direction == "cw":
            if ego_loc in [0,1]:
                return FullActionScheme.WALK_RIGHT
            if ego_loc in [2,3,4,5]:
                return FullActionScheme.WALK_DOWN
            if ego_loc in [6,7]:
                return FullActionScheme.WALK_LEFT
            if ego_loc in [8,9,10,11]:
                return FullActionScheme.WALK_UP
        elif direction == "acw":
            if ego_loc in [0,11,10,9]:
                return FullActionScheme.WALK_DOWN
            if ego_loc in [8,7]:
                return FullActionScheme.WALK_RIGHT
            if ego_loc in [6,5,4,3]:
                return FullActionScheme.WALK_UP
            if ego_loc in [2,1]:
                return FullActionScheme.WALK_LEFT

    def orient(self, target_orientation):
        if target_orientation == Direction.NORTH:
            return FullActionScheme.WALK_UP
        if target_orientation == Direction.SOUTH:
            return FullActionScheme.WALK_DOWN
        if target_orientation == Direction.EAST:
            return FullActionScheme.WALK_RIGHT
        if target_orientation == Direction.WEST:
            return FullActionScheme.WALK_LEFT

    def determine_orientation(self, pos, object):
        x,y = self.loc_to_pos_dict[pos]
        dx, dy = self.state[object]["x"]-x, self.state[object]["y"]-y
        if (dx == 0) and (dy == 1):
            return Direction.NORTH
        elif (dx == 0) and (dy == -1):
            return Direction.SOUTH
        elif (dx == 1) and (dy == 0):
            return Direction.EAST
        elif (dx == 0) and (dy == 1):
            return Direction.WEST
        else:
            return Direction.ANY

    def seek_object(self, object, direction="cw"):
        # navigates to the location of the object and changes orientation to face
        target_pos = self.state[object]["x"], self.state[object]["y"]
        return self.seek_pos(target_pos, direction=direction)

    def seek_pos(self, target_pos, direction="cw"):
        possible_navs = self.pos_to_nav_dict[target_pos]
        nav_loc, nav_orientation = self.choose_nav(possible_navs, direction=direction)
        return self.navigate_to(target_loc=nav_loc,
                                target_orientation=nav_orientation,
                                direction=direction)

    def choose_nav(self, possible_navs, direction="cw"):
        if direction == "cw":
            distances = [self.distance_to_loc(loc, direction=direction)
                         for loc, _ in possible_navs]
            idx = np.argmin(distances)
            return possible_navs[idx]
        if direction == "acw":
            distances = [self.distance_to_loc(loc, direction=direction)
                         for loc, _ in possible_navs]
            idx = np.argmin(distances)
            return possible_navs[idx]
        if direction == "adapt":
            distances = [self.distance_to_loc(loc, direction=direction)
                         for loc, _ in possible_navs]
            idx = np.argmin(distances)
            return possible_navs[idx]

    def distance_to_loc(self, loc, direction="cw"):
        ego_loc = self.pos_to_loc((self.state["ego"]["x"], self.state["ego"]["y"]))
        if direction == "cw":
            return (loc - ego_loc) % 12
        if direction == "acw":
            return (ego_loc - loc) % 12
        if direction == "adapt":
            return abs(ego_loc - loc)

    def distance_to_object(self, object, direction="cw"):
        pos = self.state[object]["x"], self.state[object]["y"]
        possible_navs = self.pos_to_nav_dict[pos]
        loc, _ = self.choose_nav(possible_navs, direction=direction)
        return self.distance_to_loc(loc, direction=direction)

    def holding_object(self, object):
        return self.objects_at_same_pos(object, "ego")

    def on_chopboard(self, object):
        return self.objects_at_same_pos(object, "chopboard")

    def in_blender(self, object):
        return self.objects_at_same_pos(object, "blender")

    def seek_and_pick_object(self, object, direction="cw"):
        if self.holding_object(object):
            return FullActionScheme.NO_OP
        seek_action = self.seek_object(object, direction=direction)
        if seek_action == FullActionScheme.NO_OP:
            return FullActionScheme.INTERACT_PRIMARY
        return seek_action

    def seek_and_chop_object(self, object, direction="cw"):
        if self.state[object]["chopped"]:
            if self.holding_object(object):
                return FullActionScheme.NO_OP # already chopped and holding
            else:
                return self.seek_and_pick_object(object, direction="cw")
        if self.on_chopboard(object):
            return FullActionScheme.EXECUTE_ACTION
        else:
            chopboard_pos = self.state["chopboard"]["x"], self.state["chopboard"]["y"]
            return self.move_object_to_pos(object,
                                           target_pos=chopboard_pos,
                                           direction=direction)

    def seek_and_blend_object(self, object, direction="cw"):
        if self.state[object]["mashed"]:
            if self.holding_object(object):
                return FullActionScheme.NO_OP # already mashed and holding
            else:
                return self.seek_and_pick_object(object, direction="cw")
        if self.in_blender(object):
            return FullActionScheme.EXECUTE_ACTION
        else:
            blender_pos = self.state["blender"]["x"], self.state["blender"]["y"]
            return self.move_object_to_pos(object,
                                           target_pos=blender_pos,
                                           direction=direction)

    def object_at_pos(self, object, pos):
        return (self.state[object]["x"], self.state[object]["y"]) == pos

    def objects_at_same_pos(self, object_1, object_2):
        return (self.state[object_1]["x"] == self.state[object_2]["x"]
                and self.state[object_1]["y"] == self.state[object_2]["y"])

    def wait_for_delivery(self, object, delivery_pos, direction="cw"):
        seek = self.seek_pos(delivery_pos, direction=direction)
        if seek == FullActionScheme.NO_OP:
            # wait until the item is delivered
            if self.object_at_pos(object, delivery_pos):
                return FullActionScheme.INTERACT_PRIMARY
            else:
                return FullActionScheme.NO_OP
        else:
            return seek

    def move_object_to_pos(self, object, target_pos, direction="cw"):
        object_pos = self.state[object]["x"], self.state[object]["y"]
        if object_pos == target_pos:
            return FullActionScheme.NO_OP # already at location
        if self.holding_object(object):
            return self.place_held_object(target_pos, direction=direction)
        else:
            return self.seek_and_pick_object(object, direction=direction)

    def place_held_object(self, target_pos, direction="cw"):
        nav_action = self.seek_pos(target_pos, direction=direction)
        if nav_action == FullActionScheme.NO_OP:
            return FullActionScheme.INTERACT_PRIMARY
        else:
            return nav_action

    def random_action(self):
        return np.random.randint(8)

    def get_action(self, obs):
        self.parse_obs(obs)

class HAgent1(HAgent):
    def __init__(self, food="tomato", state="chopped", direction="cw", epsilon=0.05):
        super().__init__()
        self.food = food
        self.food_target_state = state
        self.direction = direction
        self.achieved_goal = False
        self.epsilon = epsilon

    def get_action(self, obs):
        super().get_action(obs)
        if self.food == "nearest":
            tomato_distance = self.distance_to_object("tomato", direction=self.direction)
            carrot_distance = self.distance_to_object("carrot", direction=self.direction)
            self.food = "tomato" if tomato_distance < carrot_distance else "carrot"
            self.food_target_state = "chopped" if self.food == "tomato" else "mashed"
        elif self.food == "farthest":
            tomato_distance = self.distance_to_object("tomato", direction=self.direction)
            carrot_distance = self.distance_to_object("carrot", direction=self.direction)
            self.food = "tomato" if tomato_distance > carrot_distance else "carrot"
            self.food_target_state = "chopped" if self.food == "tomato" else "mashed"

        if (np.random.random() < epsilon) or self.achieved_goal:
            return self.random_action()

        self.achieved_goal = (self.achieved_goal
                              or (self.state[self.food][self.food_target_state]
                                  and self.holding_object(self.food)))
        if self.food_target_state == "chopped":
            return self.seek_and_chop_object(self.food, direction=self.direction)
        if self.food_target_state == "mashed":
            return self.seek_and_blend_object(self.food, direction=self.direction)

        return self.random_action()

class HAgent2(HAgent):
    def __init__(self, deliver_food="nearest", state="chopped", direction="cw", epsilon=0.05):
        super().__init__()
        self.deliver_food = deliver_food
        self.food_target_state = state
        self.direction = direction
        self.achieved_goal = False
        self.finished_waiting = False
        self.moved_object = False
        self.epsilon = epsilon

    def get_action(self, obs):
        super().get_action(obs)
        if self.deliver_food == "nearest":
            tomato_distance = self.distance_to_object("tomato", direction=self.direction)
            carrot_distance = self.distance_to_object("carrot", direction=self.direction)
            self.deliver_food = "tomato" if tomato_distance < carrot_distance else "carrot"
            self.target_food = "carrot" if self.deliver_food == "tomato" else "tomato"
            self.food_target_state = "chopped" if self.target_food == "tomato" else "mashed"
            self.deliver_pos = 2, self.state[self.deliver_food]["y"]
            self.target_pos = 2, self.state[self.target_food]["y"]
        elif self.deliver_food == "farthest":
            tomato_distance = self.distance_to_object("tomato", direction=self.direction)
            carrot_distance = self.distance_to_object("carrot", direction=self.direction)
            self.deliver_food = "tomato" if tomato_distance > carrot_distance else "carrot"
            self.target_food = "carrot" if self.deliver_food == "tomato" else "tomato"
            self.food_target_state = "chopped" if self.target_food == "tomato" else "mashed"
            self.deliver_pos = 2, self.state[self.deliver_food]["y"]
            self.target_pos = 2, self.state[self.target_food]["y"]

        if (np.random.random() < self.epsilon) or self.achieved_goal:
            return self.random_action()


        move_act = self.move_object_to_pos(self.deliver_food, self.deliver_pos, direction=self.direction)
        if move_act == FullActionScheme.NO_OP:
            self.moved_object = True
        if not self.moved_object:
            return move_act

        wait_act = self.wait_for_delivery(self.target_food, self.target_pos, direction=self.direction)
        if wait_act == FullActionScheme.INTERACT_PRIMARY:
            self.finished_waiting = True
        if not self.finished_waiting:
            return wait_act

        if self.food_target_state == "chopped":
            seek_act = self.seek_and_chop_object(self.target_food, direction=self.direction)
        if self.food_target_state == "mashed":
            seek_act = self.seek_and_blend_object(self.target_food, direction=self.direction)

        self.achieved_goal = (self.achieved_goal
                              or (self.state[self.target_food][self.food_target_state]
                                  and self.holding_object(self.target_food)))
        if not self.achieved_goal:
            return seek_act

        return self.random_action()

# Seek and chop carrot (CW, ACW, adapt)
# Seek and chop tomato (CW, ACW, adapt)
# Seek and chop nearest
# Seek and chop furthest
# Pass nearest on counter and wait
