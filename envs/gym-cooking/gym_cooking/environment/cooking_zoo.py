# Other core modules
import copy
import functools

from gym_cooking.cooking_world.cooking_world import CookingWorld
from gym_cooking.cooking_world.world_objects import *
from gym_cooking.cooking_world.actions import *
from gym_cooking.cooking_book.recipe_drawer import RECIPES, NUM_GOALS

import numpy as np
from collections import namedtuple, defaultdict
from gym import Env
import gym
from pettingzoo.utils import agent_selector

from gym.spaces import Discrete, Box, MultiBinary, Dict
from gym.utils import colorize

COLORS = ['blue', 'magenta', 'yellow', 'green']

class CookingEnvironment(Env):
    """Environment object for Overcooked."""

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "cooking_zoo"
    }
    action_scheme_map = {
        "full_action_scheme": FullActionScheme,
        "ego_turn_scheme": EgoTurnScheme,
        "simplified_cardinal_scheme": SimplifiedCardinalScheme}

    def __init__(
            self,
            level,
            num_agents,
            record,
            max_steps,
            recipes,
            obs_spaces=["numeric"],
            allowed_objects=None,
            action_scheme="full_action_scheme",
            ghost_agents=0,
            completion_reward_frac=0.2,
            time_penalty=0.0,
            mode="MARL"
        ):

        super().__init__()
        self.action_scheme = action_scheme
        self.action_scheme_class = self.action_scheme_map[self.action_scheme]
        obs_spaces = obs_spaces or ["numeric"]
        self.allowed_obs_spaces = set([
            "symbolic",
            "numeric",
            "numeric_main",
            "feature_vector",
            "feature_vector_nc",  # Feature vector excluding counters
            ])
        assert set(obs_spaces).issubset(self.allowed_obs_spaces), \
            f"Selected invalid obs spaces. Allowed {self.allowed_obs_spaces}"
        assert len(obs_spaces) > 0, \
            f"Please select an observation space from: {self.allowed_obs_spaces}"
        self.obs_spaces = obs_spaces
        self.allowed_objects = allowed_objects or []
        self.possible_agents = [f"player_{r}" for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.num_agents = num_agents

        self.level = level
        self.record = record
        self.max_steps = max_steps
        self.t = 0
        self.filename = ""
        self.set_filename()
        self.world = CookingWorld(self.action_scheme_class)
        self.seed()
        self.recipes = recipes
        self.game = None
        self.recipe_graphs = [RECIPES[recipe]() for recipe in recipes]
        self.ghost_agents = ghost_agents

        self.mode = mode

        self.terminated = False
        self.truncated = False
        self.completion_reward_frac = completion_reward_frac
        self.time_penalty = time_penalty
        self.world.load_level(level=self.level, num_agents=num_agents)
        self.graph_representation_length = sum([cls.state_length() for cls in GAME_CLASSES])
        self.has_reset = True

        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))
        # self.agent_selection = None
        # self._agent_selector = agent_selector(self.agents)
        self.done = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.accumulated_actions = []
        self.current_tensor_observation = dict(zip(self.agents, [np.zeros((self.world.width, self.world.height,
                                                                           self.graph_representation_length))
                                                                 for _ in self.agents]))
        self.observations = dict(zip(self.agents, [self.observe(agent) for agent in self.agents]))

        if self.mode == "adhoc-eval":
            self.teammate_policy = None
            self.teammate_obs = None

    def seed(self, seed=None):
        return self.world.seed(seed)

    @property
    def observation_space(self):
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents

        feature_vec_len = sum(
            [obj.feature_vector_length()
             for cls in GAME_CLASSES
             for obj in objects[ClassToString[cls]]
             ]
            )
        ghost_agent_feature_vec_len = self.ghost_agents * StringToClass["Agent"].feature_vector_length()
        feature_vec_len += ghost_agent_feature_vec_len

        feature_vec_nc_len = sum(
            [obj.feature_vector_length()
             for cls in GAME_CLASSES
             for obj in objects[ClassToString[cls]]
             if ClassToString[cls] != "Counter"
             ]
            )
        feature_vec_nc_len += ghost_agent_feature_vec_len

        numeric_obs_space = gym.spaces.Dict({
            'symbolic_observation': Box(low=0, high=10,
                                                   shape=(self.world.width, self.world.height,
                                                          self.graph_representation_length),
                                                   dtype=np.int32),
            'agent_location': Box(low=0, high=max(self.world.width, self.world.height),
                                             shape=(2,)),
            'goal_vector': MultiBinary(NUM_GOALS)
        })
        feature_vec_obs_space = Box(low=-1, high=1,
                                    shape=(len(self.possible_agents), feature_vec_len,))
        feature_vec_nc_obs_space = Box(low=-1, high=1,
                                       shape=(len(self.possible_agents), feature_vec_nc_len,))
        numeric_main_obs_space = Box(low=0, high=10,
                                     shape=(len(self.possible_agents), self.world.width, self.world.height,
                                            self.graph_representation_length))
        obs_space_dict = gym.spaces.Dict({
            "numeric": numeric_obs_space,
            "numeric_main": numeric_main_obs_space,
            "feature_vector": feature_vec_obs_space,
            "feature_vector_nc": feature_vec_nc_obs_space,
            })
        return obs_space_dict[self.obs_spaces[0]]


    @property
    def action_space(self):
        if self.mode == "adhoc-eval":
            return gym.spaces.Discrete(len(self.action_scheme_class.ACTIONS))
        return gym.spaces.MultiDiscrete([
            len(self.action_scheme_class.ACTIONS) for _ in range(len(self.possible_agents))
        ])

    def set_filename(self):
        self.filename = f"{self.level}_agents{self.num_agents}"

    def state(self):
        pass

    def reset(self, seed=None, return_info=False, options=None):
        self.world = CookingWorld(self.action_scheme_class)
        self.seed(seed)
        self.t = 0

        # For tracking data during an episode.
        self.truncated = False
        self.terminated = False

        # Load world & distances.
        self.world.load_level(level=self.level, num_agents=len(self.possible_agents))

        for recipe in self.recipe_graphs:
            recipe.update_recipe_state(self.world)

        self.agents = self.possible_agents[:]
        # self._agent_selector.reinit(self.agents)
        # self.agent_selection = self._agent_selector.next()

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        self.recipe_mapping = dict(zip(self.possible_agents, self.recipe_graphs))
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.world_agent_mapping = dict(zip(self.possible_agents, self.world.agents))
        self.world_agent_to_env_agent_mapping = dict(zip(self.world.agents, self.possible_agents))

        self.current_tensor_observation = {agent: self.get_tensor_representation(agent)
                                           for agent in self.agents}
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.dones = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.observations = np.asarray([self.observe(agent) for agent in self.agents])
        self.accumulated_actions = []

        if self.mode == "adhoc-eval":
            self.teammate_obs = np.array([a[1] for a in self.observations])
            self.teammate_policy_list = [
                    ... # TODO!!!
            ]
            self.teammate_policy = ...

        return self.observations, {}

    def close(self):
        return

    def step(self, action):
        if self.mode == "adhoc-eval":
            actions = [action]
            actions.append(self.teammate_policy.get_action(self.teammate_obs))
        for id, agent in enumerate(self.agents):
            self.accumulated_actions.append(action[id])
            for idx, agent in enumerate(self.agents):
                self.rewards[agent] = 0
            self._cumulative_rewards[agent] = 0

        self.accumulated_step(self.accumulated_actions)
        self.accumulated_actions = []

        total_rewards = 0
        for value in self.rewards.values():
            total_rewards += value

        all_done = all(list(self.dones.values()))
        return self.observations, total_rewards, all_done, False, {}

    def accumulated_step(self, actions):
        # Track internal environment info.
        self.t += 1
        # translated_actions = [action_translation_dict[actions[f"player_{idx}"]] for idx in range(len(actions))]
        self.world.perform_agent_actions(self.world.agents, actions)

        # Get an image observation
        # image_obs = self.game.get_image_obs()
        for agent in self.agents:
            self.current_tensor_observation[agent] = self.get_tensor_representation(agent)

        done, rewards, goals = self.compute_rewards()
        info = {"t": self.t, "terminated": self.terminated, "truncated": self.truncated}
        for idx, agent in enumerate(self.agents):
            self.observations = np.asarray([self.observe(agent) for agent in self.agents])
            self.dones[agent] = done
            self.rewards[agent] = rewards[idx]
            self.infos[agent] = info

    def observe(self, agent):
        observation = []
        if "numeric" in self.obs_spaces:
            num_observation = {'numeric_observation': self.current_tensor_observation[agent],
                               'agent_location': np.asarray(self.world_agent_mapping[agent].location, np.int32),
                               'goal_vector': self.recipe_mapping[agent].goals_completed(NUM_GOALS)}
            observation.append(num_observation)
        if "symbolic" in self.obs_spaces:
            objects = defaultdict(list)
            objects.update(self.world.world_objects)
            objects["Agent"] = self.world.agents
            sym_observation = copy.deepcopy(objects)
            observation.append(sym_observation)
        if "numeric_main" in self.obs_spaces:
            observation.append(self.current_tensor_observation)
        if "feature_vector" in self.obs_spaces:
            observation.append(self.get_feature_vector(agent, ignore=[]))
        if "feature_vector_nc" in self.obs_spaces:
            observation.append(self.get_feature_vector(agent, ignore=["Counter"]))
        returned_observation = observation if not len(observation) == 1 else observation[0]
        return returned_observation

    def compute_rewards(self):
        done = False
        rewards = [0] * len(self.recipes)
        open_goals = [[0]] * len(self.recipes)
        # Done if the episode maxes out
        if self.t >= self.max_steps and self.max_steps:
            self.truncated = True
            done = True

        for idx, recipe in enumerate(self.recipe_graphs):
            goals_before = recipe.goals_completed(NUM_GOALS)
            recipe.update_recipe_state(self.world)
            open_goals[idx] = recipe.goals_completed(NUM_GOALS)
            n_completed_goals = sum(goals_before) - sum(open_goals[idx])
            rewards[idx] = ((1-self.completion_reward_frac)*n_completed_goals/len(recipe.node_list)
                            + self.completion_reward_frac*recipe.completed())
            rewards[idx] -= self.time_penalty

            # objects_to_seek = recipe.get_objects_to_seek()
            # if objects_to_seek:
            #     distances = []
            #     for cls in objects_to_seek:
            #         world_objects = self.world.world_objects[ClassToString[cls]]
            #         min_distance = min([abs(self.world.agents[idx].location[0] - obj.location[0]) / self.world.height +
            #                             abs(self.world.agents[idx].location[1] - obj.location[1]) / self.world.width
            #                             for obj in world_objects])
            #         distances.append(min_distance)
            #
            #     rewards[idx] -= min(distances)

        # for idx, agent in enumerate(self.world.agents):
        #     if not agent.interacts_with:
        #         rewards[idx] -= 0.01

        if all((recipe.completed() for recipe in self.recipe_graphs)):
            self.terminated = True
            done = True
        return done, rewards, open_goals

    def feature_vector_semantics(self, ignore=[]):
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents
        semantics = []
        for cls in GAME_CLASSES:
            if ClassToString[cls] in ignore:
                continue
            for obj in objects[ClassToString[cls]]:
                features = list(obj.feature_vector_representation())
                semantics.extend([ClassToString[cls]]*len(features))
        for idx in range(self.ghost_agents):
            features = self.world_agent_mapping[agent].feature_vector_representation()
            semantics.extend(["GhostAgent"]*len(features))
        return semantics

    def get_feature_vector(self, agent, ignore=[]):
        feature_vector = []
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents
        x, y = self.world_agent_mapping[agent].location
        for cls in GAME_CLASSES:
            if ClassToString[cls] in ignore:
                continue
            for obj in objects[ClassToString[cls]]:
                features = list(obj.feature_vector_representation())
                if features and obj is not self.world_agent_mapping[agent]:
                    features[0] = (features[0] - x) / self.world.width
                    features[1] = (features[1] - y) / self.world.height
                if obj is self.world_agent_mapping[agent]:
                    features[0] = features[0] / self.world.width
                    features[1] = features[1] / self.world.height
                feature_vector.extend(features)
        for idx in range(self.ghost_agents):
            features = self.world_agent_mapping[agent].feature_vector_representation()
            features[0] = 0
            features[1] = 0
            feature_vector.extend(features)

        return np.array(feature_vector)

    def get_tensor_representation(self, agent=None):
        tensor = np.zeros((self.world.width, self.world.height, self.graph_representation_length))
        objects = defaultdict(list)
        objects.update(self.world.world_objects)
        objects["Agent"] = self.world.agents
        state_idx = 0
        for cls in GAME_CLASSES:
            for obj in objects[ClassToString[cls]]:
                x, y = obj.location
                for idx, value in enumerate(obj.numeric_state_representation()):
                    tensor[x, y, state_idx + idx] = value
            state_idx += cls.state_length()
        return tensor

    def get_agent_names(self):
        return [agent.name for agent in self.world.agents]

    def render(self, mode='human'):
        if mode == "ansi":
            return self._render_ansi()

    def _render_ansi(self):
        # TODO this is kind hacky and confusing, and will only work with tomato + lettuce.
        grid = np.full((self.world.width, self.world.height), " ", dtype=object)
        # render counters
        for counter in self.world.world_objects["Counter"]:
            x, y = counter.location
            grid[x, y] = colorize(" ", color="gray", highlight=True)
        for cut_board in self.world.world_objects["Cutboard"]:
            x, y = cut_board.location
            grid[x, y] = colorize(" ", color="yellow", highlight=True)
        for deliver_square in self.world.world_objects["Deliversquare"]:
            x, y = deliver_square.location
            grid[x, y] = colorize(" ", color="cyan", highlight=True)
        # render counter items
        # spawn tomatoes first
        for tomato in self.world.world_objects["Tomato"]:
            x, y = tomato.location
            tomato_color = "magenta" if tomato.chop_state == ChopFoodStates.CHOPPED else "red"
            tomato_string = colorize("●", tomato_color)
            grid[x, y] = grid[x, y].replace(" ", tomato_string)
        for lettuce in self.world.world_objects["Lettuce"]:
            x, y = lettuce.location
            lettuce_color = "blue" if lettuce.chop_state == ChopFoodStates.CHOPPED else "green"
            # cover the case where there's already a tomato there (i.e. chopped lettuce + tomato)
            # won't do anything if there isn't already a chopped tomato there
            lettuce_tomato_string = colorize("●", "yellow")
            grid[x, y] = grid[x, y].replace("●", lettuce_tomato_string)
            # add the lettuce in the case that it's an empty space
            lettuce_string = colorize("●", lettuce_color)
            grid[x, y] = grid[x, y].replace(" ", lettuce_string)
        # then spawn plates, which take on the color of any lettuce/tomato on them
        for plate in self.world.world_objects["Plate"]:
            x, y = plate.location
            grid[x, y] = grid[x, y].replace(" ", "O")
            grid[x, y] = grid[x, y].replace("●", "O")
        # render agents
        for agent in self.world.agents:
            x, y = agent.location
            if agent.holding is None:
                symbols = "▲▼▶◀"
                symbol = symbols[agent.orientation-1]
                grid[x, y] = grid[x, y].replace(" ", symbol)
            elif isinstance(agent.holding, Tomato):
                tomato = agent.holding
                symbols = "▲▼▶◀"
                symbol = symbols[agent.orientation-1]
                grid[x, y] = grid[x, y].replace("●", symbol)
            elif isinstance(agent.holding, Plate):
                symbols = "△▽▷◁"
                symbol = symbols[agent.orientation-1]
                grid[x, y] = grid[x, y].replace("O", symbol)
        return "\n".join(("".join(row) for row in grid)) + "\n"
