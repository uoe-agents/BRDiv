import logging
from collections import namedtuple
from enum import IntEnum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
from coopreaching.coopreaching.agents.heuristic_agent import H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11
import numpy as np

class Action(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.reward = 0
        self.history = None

        self.active = False

    def setup(self, position):
        self.history = []
        self.position = position
        self.reward = 0

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


class MARLCooperativeReachingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    action_set = [Action.NONE, Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
    Observation = namedtuple(
        "Observation",
        ["actions", "players", "game_over", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        world_length,
        world_height,
        max_episode_steps,
        mode,
        seed = 1235,
        teammate_id = -1
    ):
        self.logger = logging.getLogger(__name__)
        self.world_length = world_length
        self.world_height = world_height
        self.seed_val = seed
        self.viewer = None
        self.seed(seed)
        self.players = [Player() for _ in range(2)]
        self._max_episode_steps = max_episode_steps
        self._game_over = None
        self._rendering_initialized = False
        self.mode = mode
        self.teammate_id = teammate_id

        if self.mode == "adhoc-eval":
            self.teammate_policy = None
            self.teammate_obs = None


    @property
    def action_space(self):
        if self.mode == "adhoc-eval":
            return gym.spaces.Discrete(5)
        else:
            return gym.spaces.MultiDiscrete([5 for _ in range(len(self.players))])

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
        - player description (x, y)*player_count
        """

        min_obs = [0, 0] * len(self.players)
        max_obs = [self.world_length-1, self.world_height-1] * len(self.players)

        return gym.spaces.Dict({'all_information' : gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)})

    def _get_observation_space_real(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y)*player_count
        """
        return gym.spaces.Box(
            low=0, high=max(self.world_height-1, self.world_length-1),
            shape=(len(self.players), self._get_observation_space()['all_information'].low.shape[0],)
        )

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def spawn_players(self):

        for player in self.players:
            if not player.active:
                continue

            player.reward = 0
            player.position = (
                self.np_random.choice(list(range(1,self.world_length-1)), 1),
                self.np_random.choice(list(range(1, self.world_height - 1)), 1)
            )

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.LEFT:
            return (
                player.position[0] > 0
            )
        elif action == Action.RIGHT:
            return (
                player.position[0] < self.world_length - 1
            )
        elif action == Action.UP:
            return (
                    player.position[1] > 0
            )
        elif action == Action.DOWN:
            return (
                    player.position[1] < self.world_height - 1
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
                        history=a.history,
                        reward=a.reward if a == player else None,
                    ) for a in self.players
                ],
                # todo also check max?
                game_over=self.game_over,
                current_step=self.current_step,
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
                obs[2*i] = -1
                obs[2*i+1] = -1

            for i, p in enumerate(seen_players):
                if p:
                    obs[2*i] = p.position[0]
                    obs[2*i+1]= p.position[1]
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

        self.current_step = 0
        self.pre_rew_storage = None
        self._game_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, n_truncated, ninfo = self._make_gym_obs(observations)

        if self.mode == "adhoc-eval":
            self.teammate_obs = np.array([a[1] for a in nobs])
            self.teammate_policy_list = [
                H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11
            ]
            if self.teammate_id != -1:
                self.teammate_policy_list = self.teammate_policy_list[self.teammate_id-1:self.teammate_id]
            self.teammate_policy = self.np_random.choice(self.teammate_policy_list, 1)[0](
                (self.world_length, self.world_height),
                [(0,0), (self.world_length-1, self.world_height-1), (0, self.world_height-1), (self.world_length-1, 0)],
                [1.0, 1.0, 0.75, 0.75]
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
            elif action == Action.LEFT:
                player.position = (player.position[0]-1, player.position[1])
            elif action == Action.RIGHT:
                player.position = (player.position[0]+1, player.position[1])
            elif action == Action.UP:
                player.position = (player.position[0], player.position[1]-1)
            elif action == Action.DOWN:
                player.position = (player.position[0], player.position[1]+1)

        all_agents_finished = False
        player_position = None
        for player in self.players:
            if player_position is None:
                player_position = player.position
                continue
            elif player_position != player.position:
                break
            elif player_position == (0,0) or \
                 player_position == (self.world_length-1, self.world_height-1) or \
                 player_position == (0, self.world_height-1) or \
                 player_position == (self.world_length-1, 0) :
                all_agents_finished=True

        self._game_over = (
            all_agents_finished or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            if all_agents_finished:
                if player_position == (0,0) or player_position == (self.world_length-1, self.world_height-1):
                    p.reward = int(all_agents_finished)
                else:
                    p.reward = 0.75 * int(all_agents_finished)

        self.pre_rew_storage = [player.reward for player in self.players]

        observations_post_remove = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, n_truncated, ninfo = self._make_gym_obs(observations_post_remove)
        self.teammate_obs = np.asarray([a[1] for a in nobs])

        return np.asarray([a[1] for a in nobs]), nreward[0], ndone, n_truncated, ninfo[0]

    # def render(self, mode="human"):
    #     for p in self.players:
    #         agent_str = "".join(["O" if p.position==idx else "X" for idx in range(self.corridor_length)])
    #         print(agent_str)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.world_length, self.world_height))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
