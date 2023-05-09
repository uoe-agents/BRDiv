import logging
from collections import namedtuple
from enum import IntEnum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np

class Action(IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    PICK_ITEM = 5
    #ATTACK = 6
    #USE_ITEM = 7
    USE_ITEM = 6
    ATTACK = 7

class GameItem(IntEnum):
    NONE = -1
    LONG_SWORD = 0
    GREAT_SWORD = 1
    BOW = 2
    HEALING_STAFF = 3
    SHIELD = 4
    CHAINS = 5
    TELEPORTATION_SCROLL = 6
    BUFF_SPELLBOOK = 7

class PreyController:
    def __init__(self, config):
        self.prey_movement_params = None
        self.config = config
        self.mode = None
        self.time_to_end_mode = 0

    def _step(self, obs):
        position_x = int(obs[0])
        position_y = int(obs[1])
        health = obs[2]
        is_aggravated, is_immobilised,  is_debuffed= obs[4] > 0, obs[5] > 0, obs[6] > 0

        agent_data = {
            "position_x" : position_x,
            "position_y" : position_y,
            "health" : health,
            "is_aggravated": is_aggravated,
            "is_immobilised": is_immobilised,
            "is_debuffed": is_debuffed
        }

        predator_data = [
            {
                "position_x" : int(obs[(7+self.config["num_items"])*(i+self.config["num_prey"])]),
                "position_y" : int(obs[(7+self.config["num_items"])*(i+self.config["num_prey"])+1]),
                "item_cooldown" : int(obs[(7+self.config["num_items"])*(i+self.config["num_prey"])+2]),
                "is_buffed": int(obs[(7+self.config["num_items"])*(i+self.config["num_prey"])+6]) > 0,
                "item_in_possession": np.where(
                    obs[
                        (7+self.config["num_items"])*(i+self.config["num_prey"])+7: (7+self.config["num_items"])*(i+self.config["num_prey"]+1)
                    ] == 1
                )
            }
            for i in range(self.config["num_predator"])
        ]

        decision = self.decision(agent_data, predator_data)
        return decision

    def decision(self, agent_data, predator_data):

        # Select prey behaviour mode
        # If prey is aggravated make it behave in aggravated mode (chasing shield wielder)
        if agent_data["is_aggravated"] and self.mode != "aggravated":
            self.mode = "aggravated"
            self.time_to_end_mode = 0
        # If aggravated state ends, choose new behaviour mode
        elif self.mode == "aggravated" and (not agent_data["is_aggravated"]):
            choice_probabilities = None
            # If health below threshold, make agent prefer to run away
            if agent_data["health"] < self.config["health_threshold"]:
                choice_probabilities = [0.2, 0.3, 0.5]
            # If agent is previously attacking/running, make it more likely to wait in next mode
            elif self.mode=="attacking" or self.mode=="running":
                choice_probabilities = [0.15, 0.7, 0.15]
            # Otherwise, make prey more aggresive in attacking predators
            else:
                choice_probabilities = [0.8, 0.1, 0.1]
            self.mode = self.config["randomizer"].choice(["attacking", "waiting", "running"], p=choice_probabilities)

            # Set time till mode ends
            if self.mode == "attacking":
                self.time_to_end_mode = self.config["attacking_duration"]
            elif self.mode == "waiting":
                self.time_to_end_mode = self.config["waiting_duration"]
            else:
                self.time_to_end_mode = self.config["running_duration"]
        # If mode ends, choose new behaviour mode
        elif self.time_to_end_mode == 0:
            # Same with the above block
            choice_probabilities = None
            if agent_data["health"] < self.config["health_threshold"]:
                choice_probabilities = [0.2, 0.3, 0.5]
            elif self.mode == "attacking" or self.mode == "running":
                choice_probabilities = [0.15, 0.7, 0.15]
            else:
                choice_probabilities = [0.7, 0.2, 0.1]
            self.mode = self.config["randomizer"].choice(["attacking", "waiting", "running"], p=choice_probabilities)

            if self.mode == "attacking":
                self.time_to_end_mode = self.config["attacking_duration"]
            elif self.mode == "waiting":
                self.time_to_end_mode = self.config["waiting_duration"]
            else:
                self.time_to_end_mode = self.config["running_duration"]

        #print(self.mode, self.time_to_end_mode)
        self.time_to_end_mode = max(0, self.time_to_end_mode-1)
        #print("Mode and time to end: ", self.mode, self.time_to_end_mode)
        #self.mode = "aggravated"

        if self.mode == "aggravated":
            return self.get_aggravated_action(agent_data, predator_data)
        elif self.mode == "attacking":
            return self.get_attacking_action(agent_data, predator_data)
        elif self.mode == "waiting":
            return self.get_waiting_action(agent_data, predator_data)

        return self.get_running_action(agent_data, predator_data)

    def get_aggravated_action(self, agent_data, predator_data):
        target_agent = None
        for p in predator_data:
            if p["item_in_possession"][0].size != 0 and p["item_in_possession"][0].item() == GameItem.SHIELD:
                target_agent = p
                break

        if target_agent == None:
            return self.get_attacking_action(agent_data, predator_data)
        return self.get_attacking_action(agent_data, [target_agent])

    def get_attacking_action(self, agent_data, predator_data):
        dist_to_predator = [
            (abs(agent_data["position_x"]-p["position_x"]) + abs(agent_data["position_y"]-p["position_y"]))
            for p in predator_data
        ]

        if min(dist_to_predator) <= self.config["prey_attack_range"]:
            return Action.ATTACK

        delta_x = [-1, 1, 0, 0]
        delta_y = [0, 0, 1, -1]

        # Find action that minimises the distance with closest predator
        agent_actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
        min_agent_act_distances = []
        for action, del_x, del_y in zip(agent_actions, delta_x, delta_y):
            new_agent_position = (
                max(0, min(agent_data["position_x"] + del_x, self.config["world_length"] - 1)),
                max(0, min(agent_data["position_y"] + del_y, self.config["world_height"] - 1))
            )

            closest_dist_to_new_pos = min([
                abs(new_agent_position[0] - p_loc["position_x"]) + abs(new_agent_position[1] - p_loc["position_y"])
                for p_loc in predator_data
            ])

            if len(min_agent_act_distances) == 0:
                min_agent_act_distances.append((action, closest_dist_to_new_pos))
            else:
                if min_agent_act_distances[0][1] > closest_dist_to_new_pos:
                    min_agent_act_distances = [(action, closest_dist_to_new_pos)]

        return min_agent_act_distances[0][0]

    def get_waiting_action(self, agent_data, predator_data):
        return Action.NONE

    def get_running_action(self, agent_data, predator_data):
        delta_x = [-1,1,0,0]
        delta_y = [0,0,1,-1]

        # Find action that maximises the distance with closest predator
        agent_actions = [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
        max_agent_act_distances = []
        for action, del_x, del_y in zip(agent_actions, delta_x, delta_y):
            new_agent_position = (
                max(0, min(agent_data["position_x"]+del_x, self.config["world_length"]-1)),
                max(0, min(agent_data["position_y"]+del_y, self.config["world_height"]-1))
            )

            closest_dist_to_new_pos = min([
                abs(new_agent_position[0] - p_loc["position_x"]) + abs(new_agent_position[1] - p_loc["position_y"])
                for p_loc in predator_data
            ])

            if len(max_agent_act_distances) == 0:
                max_agent_act_distances.append((action, closest_dist_to_new_pos))
            else:
                if max_agent_act_distances[0][1] < closest_dist_to_new_pos:
                    max_agent_act_distances = [(action, closest_dist_to_new_pos)]

        #print(max_agent_act_distances[0][0])
        return max_agent_act_distances[0][0]

class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.item = None
        self.hp = None
        self.agent_type = None
        self.reward = 0
        self.history = None
        self.immobilised_period = 0
        self.item_cooldown_period = 0
        self.aggravated_period = 0
        self.buffed_period = 0

        self.active = False

    def setup(self, position, hp, agent_type):
        self.history = []
        self.position = position
        self.reward = 0
        self.item = GameItem.NONE
        self.hp = hp
        self.agent_type = agent_type
        self.item_cooldown_period = 0
        self.aggravated_period = 0
        self.immobilised_period = 0
        self.buffed_period = 0

    def pick_item(self, item_id):
        self.item = item_id

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


class MARLEnrichedPredPreyEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    action_set = [Action.NONE, Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]
    Observation = namedtuple(
        "Observation",
        ["field","actions", "players", "game_over", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["agent_type","position", "hp", "item", "item_cooldown_period", "aggravated_period",
                              "immobilised_period", "buffed_period", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        world_length,
        world_height,
        max_player_hp,
        max_prey_hp,
        player_normal_attack_accuracy,
        prey_normal_attack_accuracy,
        player_normal_attack_damage,
        prey_normal_attack_damage,
        player_normal_attack_range,
        prey_normal_attack_range,
        max_episode_steps,
        longsword_attack_accuracy,
        longsword_attack_damage,
        longsword_attack_range,
        greatsword_attack_accuracy,
        greatsword_attack_damage,
        greatsword_attack_range,
        greatsword_movement_penalty,
        bow_attack_accuracies,
        bow_attack_damage,
        bow_attack_range,
        healing_staff_cooldown_period,
        healing_staff_replenished_hp,
        healing_staff_range,
        shield_aggro_period,
        shield_aggro_cooldown_period,
        shield_range,
        shield_bash_accuracy,
        shield_damage_reduction,
        chain_immobilise_period,
        chain_immobilise_cooldown_period,
        chain_range,
        chain_accuracy,
        scroll_nearby_dist,
        scroll_range,
        scroll_cooldown_period,
        spellbook_range,
        spellbook_accuracy,
        spellbook_buff_increment,
        spellbook_buff_decrement,
        spellbook_buff_accuracy,
        spellbook_debuff_accuracy,
        spellbook_buff_effect_period,
        spellbook_buff_cooldown_period,
        seed = 1235
    ):
        self.field = -np.ones((world_height, world_length), np.int32)
        self.logger = logging.getLogger(__name__)
        self.world_length = world_length
        self.world_height = world_height
        self.seed_val = seed
        self.viewer = None
        self.max_player_hp = max_player_hp
        self.max_prey_hp = max_prey_hp
        self.seed(seed)
        self.players = [Player() for _ in range(3)]
        self._max_episode_steps = max_episode_steps
        self._game_over = None
        self._rendering_initialized = False

        # Define number of prey, predator, and items
        self.num_prey = 1
        self.num_predators = 2
        self.num_items = 8

        # Initialise normal attack params
        self.player_normal_attack_accuracy = player_normal_attack_accuracy
        self.prey_normal_attack_accuracy = prey_normal_attack_accuracy
        self.player_normal_attack_damage = player_normal_attack_damage
        self.prey_normal_attack_damage = prey_normal_attack_damage
        self.player_normal_attack_range = player_normal_attack_range
        self.prey_normal_attack_range = prey_normal_attack_range

        # Initialise item specifications
        # Longsword
        self.longsword_attack_accuracy = longsword_attack_accuracy
        self.longsword_attack_damage = longsword_attack_damage
        self.longsword_attack_range = longsword_attack_range
        # Greatsword
        self.greatsword_attack_accuracy = greatsword_attack_accuracy
        self.greatsword_attack_damage = greatsword_attack_damage
        self.greatsword_attack_range = greatsword_attack_range
        self.greatsword_movement_penalty = greatsword_movement_penalty

        # Bow
        self.bow_attack_accuracies = bow_attack_accuracies
        self.bow_attack_damage = bow_attack_damage
        self.bow_attack_range = bow_attack_range

        # To check
        # Healing staff
        self.healing_staff_cooldown_period = healing_staff_cooldown_period
        self.healing_staff_replenished_hp = healing_staff_replenished_hp
        self.healing_staff_range = healing_staff_range

        # Shield
        self.shield_aggro_period = shield_aggro_period
        self.shield_aggro_cooldown_period = shield_aggro_cooldown_period
        self.shield_bash_range = shield_range
        self.shield_bash_accuracy = shield_bash_accuracy
        self.shield_damage_reduction = shield_damage_reduction

        # Chain
        self.chain_immobilise_cooldown_period = chain_immobilise_cooldown_period
        self.chain_immobilise_period = chain_immobilise_period
        self.chain_range = chain_range
        self.chain_accuracy = chain_accuracy

        # Teleportation scroll
        self.scroll_nearby_dist = scroll_nearby_dist
        self.scroll_range = scroll_range
        self.scroll_cooldown_period = scroll_cooldown_period

        # Buff spellbook
        self.spellbook_range = spellbook_range
        self.spellbook_accuracy = spellbook_accuracy
        self.spellbook_buff_increment = spellbook_buff_increment
        self.spellbook_buff_decrement = spellbook_buff_decrement
        self.spellbook_buff_cooldown_period = spellbook_buff_cooldown_period
        self.spellbook_buff_effect_period = spellbook_buff_effect_period
        self.spellbook_buff_accuracy = spellbook_buff_accuracy
        self.spellbook_debuff_accuracy = spellbook_debuff_accuracy

        # Prey normal
        # Initialise prey obs
        self.prey_obs = None
        self.prey_config = {
            "num_items": self.num_items,
            "num_predator": self.num_predators,
            "num_prey": self.num_prey,
            "health_threshold": 0.375 * self.max_prey_hp,
            "randomizer": self.np_random,
            "attacking_duration": 8,
            "waiting_duration": 5,
            "running_duration": 6,
            "world_length": self.world_length,
            "world_height": self.world_height,
            "prey_attack_range": self.prey_normal_attack_range
        }

    @property
    def action_space(self):
        #return gym.spaces.MultiDiscrete([8 for _ in range(self.num_predators)])
        return gym.spaces.MultiDiscrete([7 for _ in range(self.num_predators)])

    @property
    def observation_space_base(self):
        return self._get_observation_space()

    @property
    def observation_space(self):
        return self._get_observation_space_real()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.world_height

    @property
    def cols(self):
        return self.world_length

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y)*player_count
        """

        min_obs = []
        max_obs = []

        item_cooldown_periods = [
            self.healing_staff_cooldown_period, self.shield_aggro_cooldown_period,
            self.chain_immobilise_cooldown_period, self.scroll_cooldown_period,
            self.spellbook_buff_cooldown_period
        ]

        # Observation description:
        # Agent observation includes --> [x position, y position, remaining hp, item in possession, item_cooldown_duration, status_duration, item locations]
        for _ in range(self.num_prey):
            min_obs.extend([0,0,0,0,0,0,0])
            min_obs.extend([0]*self.num_items)
            max_obs.extend([self.world_length-1, self.world_height-1, self.max_prey_hp,0,self.shield_aggro_period, self.chain_immobilise_period, self.spellbook_buff_effect_period])
            max_obs.extend([0]*self.num_items)

        for _ in range(self.num_predators):
            min_obs.extend([0,0,0,0,0,0,0])
            min_obs.extend([0] * self.num_items)
            max_obs.extend([self.world_length-1, self.world_height-1, self.max_player_hp, max(item_cooldown_periods), 0, 0, self.spellbook_buff_effect_period])
            max_obs.extend([1] * self.num_items)

        min_obs.extend([0, 0] * self.num_items)
        max_obs.extend([self.world_length-1, self.world_height-1] * self.num_items)

        return gym.spaces.Dict({'all_information' : gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)})

    def _get_observation_space_real(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y)*player_count
        """
        return gym.spaces.Box(
            np.repeat(
                np.expand_dims(self._get_observation_space()['all_information'].low, axis=0),
                self.num_predators,
                axis=0
            ), np.repeat(
                np.expand_dims(self._get_observation_space()['all_information'].high, axis=0),
                self.num_predators,
                axis=0
            )
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

        spawned_location = []
        for player in self.players[:self.num_prey]:
            if not player.active:
                continue

            player.reward = 0
            player.hp = self.max_prey_hp
            player.agent_type = "prey"
            player.item = GameItem.NONE
            player.controller = PreyController(config=self.prey_config)
            player.item_cooldown_period = 0
            player.aggravated_period = 0
            player.immobilised_period = 0
            player.buffed_period = 0

            sampled_position = (
                self.np_random.choice(list(range(self.world_length)), 1),
                self.np_random.choice(list(range(self.world_height)), 1)
            )

            while sampled_position in spawned_location:
                sampled_position = (
                    self.np_random.choice(list(range(self.world_length)), 1),
                    self.np_random.choice(list(range(self.world_height)), 1)
                )

                sampled_position = (
                    max(0, min(sampled_position[0], self.world_length-1)),
                    max(0, min(sampled_position[1], self.world_height-1))
                )
            player.position = sampled_position
            spawned_location.append(sampled_position)

        for player in self.players[self.num_prey:]:
            if not player.active:
                continue

            player.reward = 0
            player.hp = self.max_player_hp
            player.agent_type = "predator"
            player.item = GameItem.NONE
            player.item_cooldown_period = 0
            player.aggravated_period = 0
            player.immobilised_period = 0
            player.buffed_period = 0

            sampled_position = (
                self.np_random.choice(list(range(self.world_length)), 1)[0],
                self.np_random.choice(list(range(self.world_height)), 1)[0]
            )

            sampled_position = (
                max(0, min(sampled_position[0], self.world_length - 1)),
                max(0, min(sampled_position[1], self.world_height - 1))
            )

            while sampled_position in spawned_location:
                sampled_position = (
                    self.np_random.choice(list(range(self.world_length)), 1)[0],
                    self.np_random.choice(list(range(self.world_height)), 1)[0]
                )

                sampled_position = (
                    max(0, min(sampled_position[0], self.world_length - 1)),
                    max(0, min(sampled_position[1], self.world_height - 1))
                )

            player.position = sampled_position
            spawned_location.append(sampled_position)

    # Add method to add items to env
    def spawn_items(self):
        unavail_locations = [player.position for player in self.players]
        for item_id in range(self.num_items):
            sampled_item_position = (
                self.np_random.choice(list(range(self.world_length)), 1)[0],
                self.np_random.choice(list(range(self.world_height)), 1)[0]
            )

            sampled_item_position = (
                max(0, min(sampled_item_position[0], self.world_length - 1)),
                max(0, min(sampled_item_position[1], self.world_height - 1))
            )

            while sampled_item_position in unavail_locations:
                sampled_item_position = (
                    self.np_random.choice(list(range(self.world_length)), 1)[0],
                    self.np_random.choice(list(range(self.world_height)), 1)[0]
                )

                sampled_item_position = (
                    max(0, min(sampled_item_position[0], self.world_length - 1)),
                    max(0, min(sampled_item_position[1], self.world_height - 1))
                )

            unavail_locations.append(sampled_item_position)
            self.field[sampled_item_position[1]][sampled_item_position[0]] = item_id

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.LEFT:
            return (
                player.position[0] > 0 and (not player.immobilised_period > 0)
            )
        elif action == Action.RIGHT:
            return (
                (player.position[0] < self.world_length - 1) and (not player.immobilised_period > 0)
            )
        elif action == Action.UP:
            return (
                (player.position[1] < self.world_height - 1) and (not player.immobilised_period > 0)
            )
        elif action == Action.DOWN:
            return (
                (player.position[1] > 0) and (not player.immobilised_period > 0)
            )
        elif action == Action.PICK_ITEM:
            # Only predator can pick items

            return(
                self.field[player.position[1], player.position[0]] != -1 and player.agent_type != "prey"
            )
        elif action == Action.ATTACK:
            # Uncomment this if return to original
            if player.agent_type == "prey":
                for other_player in self.players:
                    player_dist = abs(player.position[0]-other_player.position[0]) + \
                                  abs(player.position[1]-other_player.position[1])
                    if (other_player.agent_type != player.agent_type and player_dist <= self.player_normal_attack_range) and (not player.immobilised_period > 0):
                        return (True)
            return (False)
        elif action == Action.USE_ITEM:
            return (
                player.item != GameItem.NONE and player.agent_type== "predator" and player.item_cooldown_period == 0
            )

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")


    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    # TODO Check this
    def _make_obs(self, player):
        if not player.active:
            return None

        return self.Observation(
                actions=self._valid_actions[player],
                players=[
                    self.PlayerObservation(
                        position=a.position,
                        hp=a.hp,
                        item=a.item,
                        item_cooldown_period=a.item_cooldown_period,
                        aggravated_period=a.aggravated_period,
                        immobilised_period=a.immobilised_period,
                        buffed_period=a.buffed_period,
                        agent_type=a.agent_type,
                        is_self=a == player,
                        history=a.history,
                        reward=a.reward if a == player else None,
                    ) for a in self.players
                ],
                # todo also check max?
                field=np.copy(self.field),
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
            agent_type = None
            for p in observation.players:
                if p and p.is_self:
                    agent_type = p.agent_type
                    break

            # Make sure that prey info is always put last for predators
            if agent_type == "prey":
                seen_players = [p for p in observation.players if p and p.is_self] + [
                     p for p in observation.players if (p and not p.is_self) or (not p)
                ]
            else:
                seen_players = [p for p in observation.players if p and p.is_self] + [
                    p for p in observation.players if ((p and (p.agent_type == "predator")) and not p.is_self) or (not p)
                ] + [p for p in observation.players if p and p.agent_type == "prey"]

            # TODO
            for i in range(len(self.players)):
                obs[(7+self.num_items)*i] = -1
                obs[(7+self.num_items)*i+1] = -1
                obs[(7+self.num_items)*i+2] = -1
                obs[(7+self.num_items)*i+3] = 0
                obs[(7+self.num_items)*i+4] = 0
                obs[(7+self.num_items)*i+5] = 0
                obs[(7+self.num_items)*i+6] = 0
                obs[(7+self.num_items)*i+7:(7+self.num_items)*(i+1)] = 0

            # Set all item locations to -1
            obs[(7+self.num_items)*len(self.players):] = -1

            for i, p in enumerate(seen_players):
                if p:
                    obs[(7+self.num_items)*i] = p.position[0]
                    obs[(7+self.num_items)*i+1]= p.position[1]
                    obs[(7+self.num_items)*i+2] = p.hp
                    obs[(7+self.num_items)*i+3] = p.item_cooldown_period
                    obs[(7+self.num_items)*i+4] = p.aggravated_period
                    obs[(7+self.num_items)*i+5] = p.immobilised_period
                    obs[(7+self.num_items)*i+6] = p.buffed_period
                    if p.item != GameItem.NONE:
                        obs[(7+self.num_items)*i+7+p.item] = 1

            # Add position of items belonging to no one
            for item_idx in range(self.num_items):
                item_location_y, item_location_x = np.where(observation.field==item_idx)
                obs[(7+self.num_items) * len(self.players) + 2*(item_idx)] = item_location_x[0] if len(item_location_x) != 0 else -1
                obs[(7+self.num_items) * len(self.players) + 2*(item_idx) + 1] = item_location_y[0] if len(item_location_y) != 0 else -1

            # Add position of items belonging to someone
            for player in observation.players:
                if player.item != GameItem.NONE:
                    obs[(7+self.num_items) * len(self.players) + 2 * (player.item)] = player.position[0]
                    obs[(7+self.num_items) * len(self.players) + 2 * (player.item) + 1] = player.position[1]

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

        all_obs = [
            make_obs_array(ob) if self.players[idx].active else make_obs_array(None)
            for idx, ob in enumerate(observations)
        ]

        self.prey_obs = all_obs[:self.num_prey]
        nobs = all_obs[self.num_prey:]
        nreward = [get_player_reward(obs, idx) for idx, obs in enumerate(observations)][self.num_prey:]
        ndone = all([obs.game_over if obs else True for obs in observations])
        ntruncated = False
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = [{} for obs in observations][self.num_prey:]

        # return nobs, nreward, ndone, ninfo
        # use this line to enable heuristic agents:
        return list(zip(observations, nobs)), nreward, ndone, ninfo

    def _make_gym_obs_returns(self, observations):
        def get_player_reward(observation):
            if observation is None:
                return 0.0
            for p in observation.players:
                if p.is_self:
                    return p.reward

        nreward = [get_player_reward(obs) for obs in observations]
        return nreward

    # TODO
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

        self.field = -np.ones((self.world_height, self.world_length), np.int32)
        self.spawn_players()
        self.spawn_items()

        self.current_step = 0
        self.pre_rew_storage = None
        self.prey_obs = None
        self._game_over = False
        self._gen_valid_moves()

        observations = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ntruncated, ninfo = self._make_gym_obs(observations)

        return np.array([a[1] for a in nobs]), {}

    # TODO
    def step(self, action):
        self.current_step += 1

        sum_player_previous_hp = sum([pl.hp for pl in self.players[self.num_prey:]])
        sum_prey_previous_hp = sum([pl.hp for pl in self.players[:self.num_prey]])

        # # Prey treated as NPC that decides their actions first
        actions = []
        for player, player_ob in zip(self.players[:self.num_prey], self.prey_obs):
            actions.append(player.step(player_ob))

        for p in self.players:
            p.reward = 0
        actions.extend(action)
        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions) if p.active
        ]

        # Implement effects of agent actions here
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                continue
            elif action == Action.LEFT:
                if player.item == GameItem.GREAT_SWORD and self.np_random.uniform() < self.greatsword_movement_penalty:
                        continue
                player.position = (max(0,player.position[0]-1), player.position[1])
            elif action == Action.RIGHT:
                if player.item == GameItem.GREAT_SWORD and self.np_random.uniform() < self.greatsword_movement_penalty:
                        continue
                player.position = (min(self.world_length-1,player.position[0]+1), player.position[1])
            elif action == Action.UP:
                if player.item == GameItem.GREAT_SWORD and self.np_random.uniform() < self.greatsword_movement_penalty:
                        continue
                player.position = (player.position[0], min(self.world_height-1, player.position[1]+1))
            elif action == Action.DOWN:
                if player.item == GameItem.GREAT_SWORD and self.np_random.uniform() < self.greatsword_movement_penalty:
                        continue
                player.position = (player.position[0], max(0,player.position[1]-1))
            elif action == Action.PICK_ITEM:
                # Drop old item and use new one
                if player.agent_type != "prey" and self.field[player.position[1], player.position[0]] != -1:
                    selected_item = self.field[player.position[1], player.position[0]]
                    self.field[player.position[1], player.position[0]] = player.item
                    player.item = selected_item
                    player.item_cooldown_period = 0
                    print("ITEM PICKED UP")
            elif action == Action.ATTACK:
                attack_accuracy_increase = 0
                attack_bonus = 0
                attack_accuracy_decrease = 0
                attack_penalty = 0
                if player.buffed_period > 0:
                    if player.agent_type == "predator":
                        attack_bonus = self.spellbook_buff_increment
                        attack_accuracy_increase = self.spellbook_buff_accuracy
                    else:
                        attack_penalty = self.spellbook_buff_decrement
                        attack_accuracy_decrease = self.spellbook_debuff_accuracy

                for other_player in self.players:
                    player_dist = abs(player.position[0] - other_player.position[0]) + \
                                  abs(player.position[1] - other_player.position[1])

                    # Deal damage to prey from opposing team
                    # Attack accuracy and damage determined by normal attack parameters.
                    normal_attack_range = self.player_normal_attack_range if player.agent_type == "predator" else self.prey_normal_attack_range
                    if other_player.agent_type != player.agent_type and player_dist <= normal_attack_range:
                        # If prey is not aggravated, every agent is damaged by prey attack (with shield wielder's damage reduced)
                        if player.agent_type == "prey":
                            attack_power = self.prey_normal_attack_damage - attack_penalty
                            attack_accuracy = self.prey_normal_attack_accuracy - attack_accuracy_decrease
                            if not player.aggravated_period > 0:
                                rand_uni = self.np_random.uniform()
                                if rand_uni < attack_accuracy:
                                    if not other_player.item == GameItem.SHIELD:
                                        other_player.hp -= attack_power
                                    else:
                                        other_player.hp -= (attack_power - self.shield_damage_reduction)
                            else:
                                # If prey is aggaravated, only attack shield wielder.
                                if other_player.item == GameItem.SHIELD:
                                    rand_uni = self.np_random.uniform()
                                    if rand_uni < self.prey_normal_attack_accuracy:
                                        other_player.hp -= (
                                                attack_power - self.shield_damage_reduction
                                        )
                        else:
                            attack_power = self.player_normal_attack_damage + attack_bonus
                            attack_accuracy = self.player_normal_attack_accuracy + attack_accuracy_increase
                            rand_uni = self.np_random.uniform()
                            if rand_uni < attack_accuracy:
                                print("NORMAL ATTACK USED SUCCESSFULLY!!!")
                                other_player.hp -= attack_power

            elif action == Action.USE_ITEM:
                if player.item == GameItem.LONG_SWORD:
                    added_damage = 0
                    added_accuracy = 0
                    if player.buffed_period > 0:
                        added_damage = self.spellbook_buff_increment
                        added_accuracy = self.spellbook_buff_accuracy

                    total_damage = self.longsword_attack_damage + added_damage
                    total_accuracy = self.longsword_attack_accuracy + added_accuracy
                    for other_player in self.players:
                        player_dist = abs(player.position[0] - other_player.position[0]) + \
                                      abs(player.position[1] - other_player.position[1])

                        # Deal damage to prey from opposing team
                        # Attack accuracy and damage determined by longsword attack parameters.
                        if other_player.agent_type != player.agent_type and player_dist <= self.longsword_attack_range:
                            rand_uni = self.np_random.uniform()
                            if rand_uni < total_accuracy:
                                print("SWORD USED SUCCESSFULLY!!!")
                                other_player.hp -= total_damage

                elif player.item == GameItem.GREAT_SWORD:
                    added_damage = 0
                    added_accuracy = 0
                    if player.buffed_period > 0:
                        added_damage = self.spellbook_buff_increment
                        added_accuracy = self.spellbook_buff_accuracy

                    total_damage = self.greatsword_attack_damage + added_damage
                    total_accuracy = self.greatsword_attack_accuracy + added_accuracy
                    for other_player in self.players:
                        player_dist = abs(player.position[0] - other_player.position[0]) + \
                                      abs(player.position[1] - other_player.position[1])

                        # Deal damage to prey from opposing team
                        # Attack accuracy and damage determined by longsword attack parameters.
                        if other_player.agent_type != player.agent_type and player_dist <= self.greatsword_attack_range:
                            rand_uni = self.np_random.uniform()
                            if rand_uni < total_accuracy:
                                print("GREATSWORD USED SUCCESSFULLY!!!")
                                other_player.hp -= total_damage

                elif player.item == GameItem.BOW:
                    added_damage = 0
                    added_accuracy = 0
                    if player.buffed_period > 0:
                        added_damage = self.spellbook_buff_increment
                        added_accuracy = self.spellbook_buff_accuracy

                    total_damage = self.bow_attack_damage + added_damage
                    for other_player in self.players:
                        player_dist = abs(player.position[0] - other_player.position[0]) + \
                                      abs(player.position[1] - other_player.position[1])

                        if other_player.agent_type != player.agent_type and player_dist <= self.bow_attack_range:
                            rand_uni = self.np_random.uniform()
                            if rand_uni < self.bow_attack_accuracies[int(player_dist)] + added_accuracy:
                                print("BOW USED SUCCESSFULLY!!!")
                                other_player.hp -= total_damage

                elif player.item == GameItem.HEALING_STAFF:
                    if player.item_cooldown_period == 0:
                        for other_player in self.players:
                            player_dist = abs(player.position[0] - other_player.position[0]) + \
                                          abs(player.position[1] - other_player.position[1])

                            if other_player.agent_type == player.agent_type and player_dist <= self.healing_staff_range:
                                print("STAFF USED SUCCESSFULLY!!!")
                                other_player.hp = min(
                                    other_player.hp + self.healing_staff_replenished_hp, self.max_player_hp
                                )
                        player.item_cooldown_period = self.healing_staff_cooldown_period

                elif player.item == GameItem.SHIELD:
                    if player.item_cooldown_period == 0:
                        added_accuracy = 0
                        if player.buffed_period > 0:
                            added_accuracy = self.spellbook_buff_accuracy

                        for other_player in self.players:
                            player_dist = abs(player.position[0] - other_player.position[0]) + \
                                          abs(player.position[1] - other_player.position[1])

                            if other_player.agent_type != player.agent_type and player_dist <= self.shield_bash_range:
                                rand_uni = self.np_random.uniform()
                                if rand_uni < self.shield_bash_accuracy + added_accuracy:
                                    print("SHIELD USED SUCCESSFULLY!!!")
                                    if other_player.aggravated_period == 0:
                                        other_player.aggravated_period = self.shield_aggro_period
                        player.item_cooldown_period = self.shield_aggro_cooldown_period


                elif player.item == GameItem.CHAINS:
                    if player.item_cooldown_period == 0:
                        added_accuracy = 0
                        if player.buffed_period > 0:
                            added_accuracy = self.spellbook_buff_accuracy

                        for other_player in self.players:
                            player_dist = abs(player.position[0] - other_player.position[0]) + \
                                          abs(player.position[1] - other_player.position[1])

                            if other_player.agent_type != player.agent_type and player_dist <= self.chain_range:
                                rand_uni = self.np_random.uniform()
                                if rand_uni < self.chain_accuracy + added_accuracy:
                                    print("CHAINS USED SUCCESSFULLY!!!")
                                    if other_player.immobilised_period == 0:
                                        other_player.immobilised_period = self.chain_immobilise_period
                        player.item_cooldown_period = self.chain_immobilise_cooldown_period

                elif player.item == GameItem.TELEPORTATION_SCROLL:
                    if player.item_cooldown_period == 0:
                        target_players = []
                        player.item_cooldown_period = self.scroll_cooldown_period
                        # Find player to teleport. Must be someone within range
                        for other_player in self.players:
                            if other_player.agent_type == "predator" and other_player != player:
                                teammate_dist = abs(player.position[0] - other_player.position[0]) + \
                                                abs(player.position[1] - other_player.position[1])

                                if teammate_dist <= self.scroll_range:
                                    target_players.append(other_player)

                        if not len(target_players) == 0:

                            teleported_player = self.np_random.choice(target_players)

                            prey_list = [player for player in self.players if player.agent_type=="prey"]
                            is_within_teleport_distance_to_prey = [
                                (abs(teleported_player.position[0] - prey.position[0]) + abs(teleported_player.position[1] - prey.position[1])) <= self.scroll_nearby_dist
                                for prey in prey_list
                            ]

                            # If teleported player already nearby prey, teleport them out
                            if any(is_within_teleport_distance_to_prey):
                                new_sampled_position = (
                                    self.np_random.choice(list(range(self.world_length)), 1)[0],
                                    self.np_random.choice(list(range(self.world_height)), 1)[0]
                                )

                                new_sampled_position = (
                                    max(0, min(new_sampled_position[0], self.world_length)),
                                    max(0, min(new_sampled_position[1], self.world_height))
                                )

                                is_sampled_position_within_dist_to_prey = [
                                    (abs(new_sampled_position[0] - prey.position[0]) +
                                     abs(new_sampled_position[1] - prey.position[1])) <= self.scroll_nearby_dist
                                    for prey in prey_list
                                ]

                                while any(is_sampled_position_within_dist_to_prey):
                                    new_sampled_position = (
                                        self.np_random.choice(list(range(self.world_length)), 1)[0],
                                        self.np_random.choice(list(range(self.world_height)), 1)[0]
                                    )

                                    new_sampled_position = (
                                        max(0, min(new_sampled_position[0], self.world_length-1)),
                                        max(0, min(new_sampled_position[1], self.world_height-1))
                                    )

                                    is_sampled_position_within_dist_to_prey = [
                                        (abs(new_sampled_position[0] - prey.position[0]) +
                                         abs(new_sampled_position[1] - prey.position[1])) <= self.scroll_nearby_dist
                                        for prey in prey_list
                                    ]

                                print("SCROLL USED SUCCESSFULLY!!!")
                                teleported_player.position = new_sampled_position
                            # Otherwise teleport them next to prey
                            else:
                                dest_prey = self.np_random.choice(prey_list)
                                added_val_x = self.np_random.choice(list(range(
                                    -self.scroll_nearby_dist, self.scroll_nearby_dist+1
                                )))
                                target_x_coord = dest_prey.position[0] + added_val_x
                                target_y_coord = dest_prey.position[1] + self.np_random.choice(list(range(
                                    -abs(added_val_x), abs(added_val_x)+1
                                )))

                                target_coord = (
                                    max(0,min(target_x_coord, self.world_length-1)),
                                    max(0,min(target_y_coord, self.world_height-1))
                                )

                                print("SCROLL USED SUCCESSFULLY!!!")
                                teleported_player.position = target_coord

                elif player.item == GameItem.BUFF_SPELLBOOK:
                    if player.item_cooldown_period == 0:
                        added_accuracy = 0
                        if player.buffed_period > 0:
                            added_accuracy = self.spellbook_buff_accuracy

                        for other_player in self.players:
                            if other_player == player:
                                continue
                            player_dist = abs(player.position[0] - other_player.position[0]) + \
                                          abs(player.position[1] - other_player.position[1])

                            if other_player.agent_type != player.agent_type and player_dist <= self.spellbook_range:
                                rand_uni = self.np_random.uniform()
                                if rand_uni < self.spellbook_accuracy + added_accuracy:
                                    print("BOOK USED SUCCESSFULLY!!!")
                                    if other_player.buffed_period == 0:
                                        other_player.buffed_period = self.spellbook_buff_effect_period

                            elif other_player.agent_type == player.agent_type and player_dist <= self.spellbook_range:
                                if other_player.buffed_period == 0:
                                    print("BOOK USED SUCCESSFULLY!!!")
                                    other_player.buffed_period = self.spellbook_buff_effect_period

                        player.item_cooldown_period = self.spellbook_buff_cooldown_period

        for player in self.players:
            player.hp = max(player.hp, 0)
            player.item_cooldown_period = max(player.item_cooldown_period-1, 0)
            player.aggravated_period = max(player.aggravated_period-1, 0)
            player.immobilised_period = max(player.immobilised_period - 1, 0)
            player.buffed_period = max(player.buffed_period - 1, 0)

        # Since we do not consider closed environments
        # end game when any agent dies.

        prey_is_dead = []
        player_is_dead = []
        dead_player_type = []
        dead_true_player = []
        for player in self.players:
            if max(player.hp, 0) == 0:
                player_is_dead.append(True)
                dead_player_type.append(player.agent_type)
                if player.agent_type == "prey":
                    prey_is_dead.append(True)
                else:
                    dead_true_player.append(player)
            else:
                player_is_dead.append(False)

        any_prey_is_dead = any(prey_is_dead)
        any_player_is_dead = any(player_is_dead)
        # Game ends if all prey are slain or if episode ends
        self._game_over = (
            any_prey_is_dead or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        agent_death_penalty = 0
        if len(dead_player_type) > 0:
            agent_death_penalty = sum([self.max_prey_hp if agent_type == "prey" else -self.max_player_hp for agent_type in dead_player_type])

        sum_player_curr_hp = sum([pl.hp for pl in self.players[self.num_prey:]])
        sum_prey_curr_hp = sum([pl.hp for pl in self.players[:self.num_prey]])

        for p in self.players:
            if player.agent_type == "predator":
                # Reward delta in hp alongside with bonus for slaying prey or being slain by prey
                p.reward = 0.1 * (agent_death_penalty + (sum_player_curr_hp - sum_player_previous_hp) + (sum_prey_previous_hp - sum_prey_curr_hp))

        self.pre_rew_storage = [player.reward for player in self.players]
        self.reanimate_dead_players(dead_true_player)

        observations_post_remove = [self._make_obs(player) for player in self.players]
        nobs, nreward, ndone, ntruncated, ninfo = self._make_gym_obs(observations_post_remove)

        return np.asarray([a[1] for a in nobs]), nreward[0], ndone, ntruncated, ninfo[0]

    # def render(self, mode="human"):
    #     for p in self.players:
    #         agent_str = "".join(["O" if p.position==idx else "X" for idx in range(self.corridor_length)])
    #         print(agent_str)

    def reanimate_dead_players(self, dead_true_player):
        if len(dead_true_player) == 0:
            return

        player_previous_item = GameItem.NONE
        spawned_locations = [(p.position, p.agent_type) for p in self.players if p.hp != 0]

        for player in dead_true_player:
            if not player.active:
                continue

            # Reinitialise player previous item somewhere else
            player_previous_item = player.item

            player.hp = self.max_player_hp
            player.agent_type = "predator"
            player.item = GameItem.NONE
            player.item_cooldown_period = 0
            player.aggravated_period = 0
            player.immobilised_period = 0
            player.buffed_period = 0

            sampled_position = (
                self.np_random.choice(list(range(self.world_length)), 1)[0],
                self.np_random.choice(list(range(self.world_height)), 1)[0]
            )

            sampled_position = (
                max(0, min(sampled_position[0], self.world_length - 1)),
                max(0, min(sampled_position[1], self.world_height - 1))
            )

            redo_sampling = False
            for p_loc_type in spawned_locations :
                if p_loc_type[1] == "predator" and sampled_position == p_loc_type[0]:
                    redo_sampling = True
                elif p_loc_type[1] == "prey" and (
                    abs(p_loc_type[0][0]- sampled_position[0]) + abs(
                    p_loc_type[0][1]-sampled_position[1]) <= self.prey_normal_attack_range
                ):
                    redo_sampling = True
                else:
                    redo_sampling = False

            while redo_sampling:
                sampled_position = (
                    self.np_random.choice(list(range(self.world_length)), 1),
                    self.np_random.choice(list(range(self.world_height)), 1)
                )

                sampled_position = (
                    max(0, min(sampled_position[0], self.world_length - 1)),
                    max(0, min(sampled_position[1], self.world_height - 1))
                )

                for p_loc_type in spawned_locations:
                    if p_loc_type[1] == "predator" and sampled_position == p_loc_type[0]:
                        redo_sampling = True
                    elif p_loc_type[1] == "prey" and (
                            abs(p_loc_type[0][0] - sampled_position[0]) + abs(
                        p_loc_type[0][1] - sampled_position[1]) <= self.prey_normal_attack_range
                    ):
                        redo_sampling = True
                    else:
                        redo_sampling = False

            player.position = sampled_position
            spawned_locations.append((player.position, player.agent_type))

            unavail_item_locations = [player.position for player in self.players]
            sampled_item_position = (
                self.np_random.choice(list(range(self.world_length)), 1)[0],
                self.np_random.choice(list(range(self.world_height)), 1)[0]
            )

            sampled_item_position = (
                max(0, min(sampled_item_position[0], self.world_length - 1)),
                max(0, min(sampled_item_position[1], self.world_height - 1))
            )

            while (sampled_item_position in unavail_item_locations) or self.field[sampled_item_position[1]][sampled_item_position[0]] != GameItem.NONE:
                sampled_item_position = (
                    self.np_random.choice(list(range(self.world_length)), 1)[0],
                    self.np_random.choice(list(range(self.world_height)), 1)[0]
                )

                sampled_item_position = (
                    max(0, min(sampled_item_position[0], self.world_length - 1)),
                    max(0, min(sampled_item_position[1], self.world_height - 1))
                )

            self.field[sampled_item_position[1]][sampled_item_position[0]] = player_previous_item



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


