import gym
import corridor
import lbforaging
import coopreaching
import coopreaching3D
import enrichedpredprey
import gym_cooking
import torch
import string
import json
import numpy as np
from Agents import Agents
import time

class PolicyRender(object):
    """
        A class that runs an experiment on learning with Upside Down Reinforcement Learning (UDRL).
    """

    def __init__(self, config):
        """
            Constructor for UDRLTraining class
                Args:
                    config : A dictionary containing required hyperparameters for UDRL training
        """
        self.config = config
        self.device = torch.device("cuda" if config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
        self.env_name = config.env["name"]

    def get_obs_sizes(self, obs_space):
        """
            Method to get the size of the envs' obs space and length of obs features. Must be defined for every envs.
        """
        out_shape = list(obs_space.shape)
        out_shape[-1] += self.config.populations["num_populations"]
        if "Corridor" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        if "Foraging" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        if "Reaching" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        if "Enriched" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        if "cooking" in self.env_name:
            return tuple(out_shape), obs_space.shape[-1] + self.config.populations["num_populations"]
        return None

    def to_one_hot_population_id(self, indices):
        act_indices = np.asarray(indices).astype(int)
        one_hot_ids = np.eye(self.config.populations["num_populations"])[act_indices]

        return one_hot_ids

    def eval(self):
        """
            A method that encompasses the main training loop for UDRL.
        """

        # Initialize environment, agent population model & experience replay based on obs vector sizes
        env1 = gym.make(
            self.config.env["name"]
        )

        obs, _ = env1.reset()
        obs = np.asarray([obs])
        env1.render()

        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)
        act_sizes = env1.action_space.nvec[0]
        act_sizes_all = (len(env1.action_space.nvec), act_sizes)

        obs_size_list = list(obs_sizes)
        real_obs_size = tuple(obs_size_list)

        device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
        obs_size = env1.observation_space.shape[-1] + self.config.populations["num_populations"]

        agent_population = Agents(obs_size, agent_o_size, obs_sizes[0], self.config.populations["num_populations"],
                                  self.config, act_sizes, device, None, mode="eval")

        agent_population.load_model(int(self.config.run["load_from_checkpoint"]))

        # Continuation
        timesteps_elapsed = 0
        sp_selected_agent_idx = [self.config.render["rendered_pop_id"]]

        done = False
        while not done:
            one_hot_id_shape = list(obs.shape)[:-1]
            one_hot_ids = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(sp_selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape))
            # acts = agent_population.decide_acts(np.concatenate([obs, remaining_target, time_elapsed], axis=-1))
            # Decide agent's action based on model and execute.

            real_input = np.concatenate([obs, one_hot_ids], axis=-1)
            acts, act_log_prob = agent_population.decide_acts(real_input, True)
            nobs, rews, done, _, infos = env1.step(acts[0])
            time.sleep(0.5)
            env1.render()
            nobs = np.asarray([nobs])
            real_n_input = np.concatenate([nobs, one_hot_ids], axis=-1)

            obs = nobs
