import logging
from collections import namedtuple
from enum import IntEnum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
from cir_ovc.cir_ovc.agents.heuristic_agent import H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12

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

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.reward = 0
        self.history = None
        self.possession = None
        self.active = False

    def setup(self, position):
        self.history = []
        self.position = position
        self.reward = 0
        self.possession = []

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"

class MARLCircularOvercookedEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    action_set = [
        Action.NONE, Action.COUNTER_CLOCKWISE, Action.CLOCKWISE,
        Action.PUT_OUTSIDE, Action.PUT_INSIDE, Action.TAKE_INSIDE,
        Action.TAKE_OUTSIDE, Action.USE_ITEM
    ]

    Observation = namedtuple(
        "Observation",
        ["actions", "players", "game_over", "current_step", "collectibles", "noncollectibles"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "possession", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        corridor_length,
        max_episode_steps,
        mode,
        seed = 1235,
        teammate_id = -1
    ):
        self.logger = logging.getLogger(__name__)
        self.corridor_length = corridor_length
        self.seed_val = seed
        self.viewer = None
        self.seed(seed)
        self.players = [Player() for _ in range(2)]
        self._max_episode_steps = max_episode_steps
        self._game_over = None
        self._rendering_initialized = False

        self.num_collectible_items = 5
        self.num_non_collectible_items = 3
        self.collectible_locations = None
        self.non_collectible_locations = None
        self.food_assembled = None

        self.mode = mode
        self.teammate_id = teammate_id

        if self.mode == "adhoc-eval":
            self.teammate_policy = None
            self.teammate_obs = None

    @property
    def action_space(self):
        if self.mode == "adhoc-eval":
            return gym.spaces.Discrete(8)
        else:
            return gym.spaces.MultiDiscrete([8 for _ in range(len(self.players))])

    @property
    def observation_space_base(self):
        return self._get_observation_space()

    @property
    def observation_space(self):
        return self._get_observation_space_real()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """

        min_obs = [0, 0, 0, 0, 0, 0] * len(self.players)
        max_obs = [self.corridor_length - 1, 1, 1, 1, 1, 1] * len(self.players)

        min_collectibles_location = [-1] * self.num_collectible_items
        max_collectibles_location = [self.corridor_length] * self.num_collectible_items

        min_non_collectible_locations = [0] * self.num_non_collectible_items
        max_non_collectible_locations = [self.corridor_length - 1] * self.num_non_collectible_items

        min_obs.extend(min_collectibles_location)
        min_obs.extend(min_non_collectible_locations)

        max_obs.extend(max_collectibles_location)
        max_obs.extend(max_non_collectible_locations)

        return gym.spaces.Dict({'all_information' : gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)})

    def _get_observation_space_real(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """

        return gym.spaces.Box(
            low=0, high=self.corridor_length-1,
            shape=(len(self.players), self._get_observation_space()['all_information'].low.shape[0],)
        )

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, self.players, action)
            ]
            for player in self.players
        }

    def spawn_players(self):

        player_locations = self.np_random.choice(list(range(self.corridor_length)), len(self.players), replace=False)
        for idx, player in enumerate(self.players):
            if not player.active:
                continue

            player.reward = 0
            player.position = player_locations[idx]
            player.possession = []

    def _is_valid_action(self, player, players_list, action):
        if action == Action.NONE:
            return True
        elif action == Action.COUNTER_CLOCKWISE:
            return not any(
                [p.position == ((player.position-1) % self.corridor_length) for p in players_list]
            )
        elif action == Action.CLOCKWISE:
            return not any(
                [p.position == ((player.position+1) % self.corridor_length) for p in players_list]
            )
        elif action == Action.PUT_INSIDE or action == Action.PUT_OUTSIDE:
            return len(player.possession) != 0
        elif action == Action.TAKE_INSIDE:
            return any([
                self.collectible_locations[k] == self.corridor_length for k in self.collectible_locations.keys()
            ])
        elif action == Action.TAKE_OUTSIDE:
            return any([
                self.collectible_locations[k] == player.position for k in self.collectible_locations.keys()
            ])
        elif action == Action.USE_ITEM:
            return (
                    (
                        self.collectible_locations[Collectibles.TOMATO] == self.non_collectible_locations[NonCollectibles.KNIFE]
                    ) and (
                        player.position == self.non_collectible_locations[NonCollectibles.KNIFE]
                    )
                ) or (
                    (
                        self.collectible_locations[Collectibles.CARROT] == self.non_collectible_locations[NonCollectibles.BLENDER]
                    ) and (
                        player.position == self.non_collectible_locations[NonCollectibles.BLENDER]
                    )
                )

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        if not player.active:
            return None

        return self.Observation(
                actions=self._valid_actions[player],
                players=[
                    self.PlayerObservation(
                        position=a.position,
                        is_self=a == player,
                        possession= a.possession,
                        history=a.history,
                        reward=a.reward if a == player else None,
                    ) for a in self.players
                ],
                # todo also check max?
                game_over=self.game_over,
                current_step=self.current_step,
                collectibles=self.collectible_locations,
                noncollectibles=self.non_collectible_locations
        )


    def _make_gym_obs(self, observations):
        def make_obs_array(observation):
            obs = -np.ones(self.observation_space_base['all_information'].shape)
            if observation is None:
                return obs
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p and p.is_self] + [
                 p for p in observation.players if (p and not p.is_self) or (not p)
            ]

            for i in range(len(self.players)):
                obs[(1+self.num_collectible_items)*i:(1+self.num_collectible_items)*(i+1)] = -1

            for i, p in enumerate(seen_players):
                if p:
                    obs[(1+self.num_collectible_items)*i] = p.position
                    for possessed_item in p.possession:
                        obs[(1 + self.num_collectible_items)*i + possessed_item + 1] = 1

            offset = (1+self.num_collectible_items)*len(self.players)

            for collectible_id in observation.collectibles.keys():
                obs[offset+collectible_id] = observation.collectibles[collectible_id]
            for non_collectible_id in observation.noncollectibles.keys():
                obs[offset+non_collectible_id] = observation.noncollectibles[non_collectible_id]
            return obs

        def get_player_reward(observation, idx):
            if not self.pre_rew_storage is None:
                if observation is None and self.pre_rew_storage[idx] == 0.0:
                    return 0.0
                if observation is None and self.pre_rew_storage[idx] != 0.0:
                    return self.pre_rew_storage[idx]
                for p in observation.players:
                    if p and p.is_self:
                        return p.reward
            else:
                if observation is None:
                    return 0.0
                for p in observation.players:
                    if p and p.is_self:
                        return p.reward


        nobs = [make_obs_array(ob) if self.players[idx].active else make_obs_array(None)
                for idx, ob in enumerate(observations)]
        nreward = [get_player_reward(obs, idx) for idx, obs in enumerate(observations)]
        ndone = all([obs.game_over if obs else True for obs in observations])
        ntruncated = False
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = [{} for obs in observations]

        # todo this?:
        # return nobs, nreward, ndone, ninfo
        # use this line to enable heuristic agents:
        return list(zip(observations, nobs)), nreward, ndone, ntruncated, ninfo

    def _make_gym_obs_returns(self, observations):
        def get_player_reward(observation):
            if observation is None:
                return 0.0
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nreward = [get_player_reward(obs) for obs in observations]
        return nreward

    def reset(self, seed=None):

        if seed != None:
            self.seed_val = seed
        elif self.seed_val != None:
            self.seed_val = self.seed_val + 123
        else:
            self.seed_val = 0

        self.seed(self.seed_val)

        for idx in range(len(self.players)):
            self.players[idx].active = True

        self.spawn_players()

        if self.corridor_length == 8:
            self.collectible_locations = {
                Collectibles.TOMATO: 1,
                Collectibles.CHOPPED_TOMATO: -1,
                Collectibles.CARROT: 5,
                Collectibles.BLENDED_CARROT: -1,
                Collectibles.PLATE: 8
            }
            self.non_collectible_locations = {
                NonCollectibles.BLENDER: 0,
                NonCollectibles.KNIFE: 4,
                NonCollectibles.SERVING_AREA: 3
            }
        elif self.corridor_length == 9:
            self.collectible_locations = {
                Collectibles.TOMATO: 1,
                Collectibles.CHOPPED_TOMATO: -1,
                Collectibles.CARROT: 6,
                Collectibles.BLENDED_CARROT: -1,
                Collectibles.PLATE: 9
            }
            self.non_collectible_locations = {
                NonCollectibles.BLENDER: 0,
                NonCollectibles.KNIFE: 5,
                NonCollectibles.SERVING_AREA: 3
            }
        else:
            self.collectible_locations = {
                Collectibles.TOMATO: 1,
                Collectibles.CHOPPED_TOMATO: -1,
                Collectibles.CARROT: 6,
                Collectibles.BLENDED_CARROT: -1,
                Collectibles.PLATE: 10
            }
            self.non_collectible_locations = {
                NonCollectibles.BLENDER: 0,
                NonCollectibles.KNIFE: 5,
                NonCollectibles.SERVING_AREA: 8
            }

        self.current_step = 0
        self.pre_rew_storage = None
        self._game_over = False
        self.food_assembled = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ntruncated, ninfo = self._make_gym_obs(observations)

        if self.mode == "adhoc-eval":
            self.teammate_obs = np.array([a[1] for a in nobs])
            self.teammate_policy_list = [
                H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12
            ]
            if self.teammate_id != -1:
                self.teammate_policy_list = self.teammate_policy_list[self.teammate_id-1:self.teammate_id]
            self.teammate_policy = self.np_random.choice(self.teammate_policy_list, 1)[0](
                self.corridor_length
            )

        return np.array([a[1] for a in nobs]), {}

    def step(self, action):
        self.current_step += 1

        for p in self.players:
            p.reward = 0
        actions = action

        if self.mode == "adhoc-eval":
            actions = [action]
            actions.append(self.teammate_policy.step(self.teammate_obs[1,:]))

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions) if p.active
        ]

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                continue
            elif action == Action.COUNTER_CLOCKWISE:
                if not any(
                    [p.position == ((player.position-1) % self.corridor_length) for p in self.players]
                ):
                    player.position = ((player.position-1) % self.corridor_length)
            elif action == Action.CLOCKWISE:
                if not any(
                        [p.position == ((player.position+1) % self.corridor_length) for p in self.players]
                ):
                    player.position = ((player.position+1) % self.corridor_length)
            elif action == Action.PUT_OUTSIDE:
                for pos_id in player.possession:
                    self.collectible_locations[pos_id] = player.position
                player.possession = []
            elif action == Action.PUT_INSIDE:
                for pos_id in player.possession:
                    self.collectible_locations[pos_id] = self.corridor_length
                player.possession = []
            elif action == Action.TAKE_OUTSIDE:
                items_to_collect = [
                    k for k in self.collectible_locations.keys() if self.collectible_locations[k] == player.position
                ]
                player.possession.extend(items_to_collect)
                for k in items_to_collect:
                    self.collectible_locations[k] = -1

            elif action == Action.TAKE_INSIDE:
                items_to_collect = [
                    k for k in self.collectible_locations.keys() if self.collectible_locations[k] == self.corridor_length
                ]
                player.possession.extend(items_to_collect)
                for k in items_to_collect:
                    self.collectible_locations[k] = -1
            elif action == Action.USE_ITEM:
                if player.position == self.non_collectible_locations[NonCollectibles.KNIFE]:
                    if self.collectible_locations[Collectibles.TOMATO] == self.non_collectible_locations[NonCollectibles.KNIFE]:
                        self.collectible_locations[Collectibles.TOMATO] = -1
                        self.collectible_locations[Collectibles.CHOPPED_TOMATO] = player.position
                        for p in self.players:
                            p.reward += 0.25
                elif player.position == self.non_collectible_locations[NonCollectibles.BLENDER]:
                    if self.collectible_locations[Collectibles.CARROT] == self.non_collectible_locations[NonCollectibles.BLENDER]:
                        self.collectible_locations[Collectibles.CARROT] = -1
                        self.collectible_locations[Collectibles.BLENDED_CARROT] = player.position
                        for p in self.players:
                            p.reward += 0.25

        if (not self.food_assembled) and (
            self.collectible_locations[Collectibles.PLATE] == self.collectible_locations[Collectibles.CHOPPED_TOMATO]
        ) and (
            self.collectible_locations[Collectibles.PLATE] == self.collectible_locations[Collectibles.BLENDED_CARROT]
        ) and (
            self.collectible_locations[Collectibles.PLATE] != -1
        ):
            print("FOOD_ASSEMBLED")
            self.food_assembled = True
            for p in self.players:
                p.reward += 0.25

        completed_item = [Collectibles.CHOPPED_TOMATO, Collectibles.BLENDED_CARROT, Collectibles.PLATE]
        if (not self.food_assembled) and (
            self.collectible_locations[Collectibles.PLATE] == -1
        ) and (
            all([a in self.players[0].possession for a in completed_item]) or
            all([a in self.players[1].possession for a in completed_item])
        ):
            self.food_assembled = True
            for p in self.players:
                p.reward += 0.25

        task_completed = False
        if self.food_assembled and self.collectible_locations[Collectibles.PLATE] == self.non_collectible_locations[NonCollectibles.SERVING_AREA]:
            task_completed = True
            for p in self.players:
                p.reward += 0.25

        self._game_over = (
            task_completed or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        self.pre_rew_storage = [player.reward for player in self.players]

        observations_post_remove = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ntruncated, ninfo = self._make_gym_obs(observations_post_remove)
        self.teammate_obs = np.asarray([a[1] for a in nobs])

        return np.asarray([a[1] for a in nobs]), nreward[0], ndone, ntruncated, ninfo[0]

    # def render(self, mode="human"):
    #     for p in self.players:
    #         agent_str = "".join(["O" if p.position==idx else "X" for idx in range(self.corridor_length)])
    #         print(agent_str)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((1, self.corridor_length))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
