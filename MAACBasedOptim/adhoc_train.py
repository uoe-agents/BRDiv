import gym
import random
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
from ExpReplay import EpisodicSelfPlayExperienceReplay, EpisodicCrossPlayExperienceReplay
from AdhocAgent import AdhocAgent
from Agents import Agents
# from train import Logger
import os
import wandb
from omegaconf import OmegaConf


class AdhocTraining(object):
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

        self.logger = Logger(config)
        #self.logger= None

        # Other experiment related variables
        self.exp_replay = None

        self.sp_selected_agent_idx = None

        self.stored_obs = None
        self.stored_nobs = None
        self.prev_cell_values = None
        self.prev_agent_actions = None
        self.agent_representation_list = None

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

    def create_directories(self):
        """
            A method that creates the necessary directories for storing resulting logs & parameters.
        """
        if not os.path.exists("adhoc_model"):
            os.makedirs("adhoc_model")

    def to_one_hot_population_id(self, indices):
        act_indices = np.asarray(indices).astype(int)
        one_hot_ids = np.eye(self.config.populations["num_populations"])[act_indices]

        return one_hot_ids

    def adhoc_data_gathering(self, env, adhoc_agent, agent_population, real_obs_size, act_sizes_all):
        # Get required data from selected agents
        target_timesteps_elapsed = self.config.train["timesteps_per_update"]
        self.agent_representation_list = []

        timesteps_elapsed = 0
        if not self.sp_selected_agent_idx:
            self.sp_selected_agent_idx = [
                np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0] for _ in
                range(self.config.env.parallel["adhoc_collection"])]

        real_obs_header_size = [self.config.env.parallel["adhoc_collection"], target_timesteps_elapsed]
        act_header_size = [self.config.env.parallel["adhoc_collection"], target_timesteps_elapsed]
        batch_size = self.config.env.parallel["adhoc_collection"]

        real_obs_header_size.extend(list(real_obs_size))
        act_header_size.extend(list(act_sizes_all))

        stored_real_obs = np.zeros(real_obs_header_size)
        stored_next_real_obs = np.zeros(real_obs_header_size)
        stored_acts = np.zeros(act_header_size)
        stored_rewards = np.zeros([self.config.env.parallel["adhoc_collection"], target_timesteps_elapsed])
        stored_dones = np.zeros([self.config.env.parallel["adhoc_collection"], target_timesteps_elapsed])

        cell_values = self.prev_cell_values
        agent_prev_action = self.prev_agent_actions

        while timesteps_elapsed < target_timesteps_elapsed:
            one_hot_id_shape = list(self.stored_obs.shape)[:-1]
            # print((selected_agent_idx * np.ones(one_hot_id_shape)).shape)
            one_hot_ids = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(self.sp_selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape))

            # Decide agent's action based on model and execute.
            real_input = np.concatenate([self.stored_obs, one_hot_ids], axis=-1)
            acts, act_log_prob = agent_population.decide_acts(real_input, True)

            # Compute teammate_representation
            if agent_prev_action is None:
                agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])

            encoder_representation_input = np.concatenate(
                [
                    self.stored_obs, agent_prev_action
                ], axis=-1
            )
            agent_representation, cell_values = adhoc_agent.get_teammate_representation(
                encoder_representation_input, cell_values
            )

            self.agent_representation_list.append(agent_representation.unsqueeze(1))

            rl_agent_representation = agent_representation.detach()
            ah_obs = torch.tensor(self.stored_obs).double().to(self.device)[:, 0, :]
            ah_policy_input = torch.cat([ah_obs, rl_agent_representation], dim=-1)
            ah_agents_acts, ah_agent_log_prob = adhoc_agent.decide_acts(ah_policy_input, True)

            final_acts = [(a1, a2[0]) for a1, a2 in zip(ah_agents_acts, acts)]
            self.stored_nobs, rews, dones, _, infos = env.step(final_acts)

            # Store data from most recent timestep into tracking variables
            one_hot_acts = agent_population.to_one_hot(final_acts)
            next_encoder_representation_input = np.concatenate(
                [
                    self.stored_nobs, one_hot_acts
                ], axis=-1
            )

            stored_real_obs[:, timesteps_elapsed] = encoder_representation_input
            stored_next_real_obs[:, timesteps_elapsed] = next_encoder_representation_input
            stored_acts[:, timesteps_elapsed] = one_hot_acts
            stored_rewards[:, timesteps_elapsed] = rews
            stored_dones[:, timesteps_elapsed] = dones

            agent_prev_action = one_hot_acts
            self.stored_obs = self.stored_nobs

            timesteps_elapsed += 1

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if flag:
                    self.sp_selected_agent_idx[idx] = \
                    np.random.choice(list(range(self.config.populations["num_populations"])), 1)[0]
                    cell_values[0][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                        self.device)
                    cell_values[1][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                        self.device)
                    agent_prev_action[idx] = np.zeros(list(agent_prev_action[idx].shape))

        self.prev_cell_values = cell_values
        self.prev_agent_actions = agent_prev_action

        encoder_representation_input = np.concatenate(
            [
                self.stored_obs, agent_prev_action
            ], axis=-1
        )

        agent_representation, _ = adhoc_agent.get_teammate_representation(
            encoder_representation_input, cell_values
        )

        self.agent_representation_list.append(agent_representation.unsqueeze(1))
        for r_obs, nr_obs, acts, rewards, dones in zip(stored_real_obs, stored_next_real_obs, stored_acts,
                                                       stored_rewards, stored_dones):
            self.exp_replay.add_episode(r_obs, acts, rewards, dones, nr_obs)

    def eval_train_policy_performance(self, adhoc_agent, agent_population, logger, logging_id):
        # Get required data from selected agents
        target_num_episodes = self.config.run["num_eval_episodes"] // self.config.env.parallel["eval_collection"]
        env1 = gym.make(
            self.config.env["name"]
        )

        act_sizes = env1.action_space.nvec[0]
        act_sizes_all = (len(env1.action_space.nvec), act_sizes)

        def make_env(env_name):
            def _make():
                env = gym.make(
                    env_name
                )
                return env

            return _make

        env = gym.vector.SyncVectorEnv([
            make_env(
                self.config.env["name"]
            ) for idx in range(self.config.env.parallel["eval_collection"])
        ])

        eval_obses, _ = env.reset(
            seed=[self.config.run["seed"] + idx for idx in range(self.config.env.parallel["eval_collection"])]
        )

        episodes_elapsed_per_thread = [0] * self.config.env.parallel["eval_collection"]
        episode_length_counter = [0] * self.config.env.parallel["eval_collection"]
        randomizer = np.random.RandomState(seed=self.config.run["seed"])

        # Ensure that evaluated seed remains the same between evals
        teammate_ids = randomizer.choice(
            self.config.populations["num_populations"],
            (self.config.env.parallel["eval_collection"], target_num_episodes),
        )

        total_returns_discounted = np.zeros([self.config.env.parallel["eval_collection"], target_num_episodes])
        total_returns_undiscounted = np.zeros([self.config.env.parallel["eval_collection"], target_num_episodes])

        # Set initial values for interaction
        selected_agent_idx = teammate_ids[:,0].tolist()
        batch_size = self.config.env.parallel["eval_collection"]
        cell_values = None
        agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])

        while any([eps < target_num_episodes for eps in episodes_elapsed_per_thread]):
            one_hot_id_shape = list(eval_obses.shape)[:-1]
            one_hot_ids = self.to_one_hot_population_id(
                np.expand_dims(np.asarray(selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape))

            # Decide agent's action based on model and execute.
            real_input = np.concatenate([eval_obses, one_hot_ids], axis=-1)
            acts, act_log_prob = agent_population.decide_acts(real_input, True)

            # Compute teammate_representation
            if agent_prev_action is None:
                agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])

            encoder_representation_input = np.concatenate(
                [
                    eval_obses, agent_prev_action
                ], axis=-1
            )
            agent_representation, cell_values = adhoc_agent.get_teammate_representation(
                encoder_representation_input, cell_values
            )

            rl_agent_representation = agent_representation.detach()
            ah_obs = torch.tensor(eval_obses).double().to(self.device)[:, 0, :]
            ah_policy_input = torch.cat([ah_obs, rl_agent_representation], dim=-1)
            ah_agents_acts, ah_agent_log_prob = adhoc_agent.decide_acts(ah_policy_input, True)

            final_acts = [(a1, a2[0]) for a1, a2 in zip(ah_agents_acts, acts)]
            eval_nobses, rews, dones, _, infos = env.step(final_acts)

            agent_prev_action = agent_population.to_one_hot(final_acts)
            eval_obses = eval_nobses

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if episodes_elapsed_per_thread[idx] < total_returns_undiscounted.shape[1]:
                    total_returns_undiscounted[idx][episodes_elapsed_per_thread[idx]] += rews[idx]
                    total_returns_discounted[idx][episodes_elapsed_per_thread[idx]] += (
                        self.config.train["gamma"]**episode_length_counter[idx]
                    )*rews[idx]
                    episode_length_counter[idx] += 1

                if flag:
                    if episodes_elapsed_per_thread[idx] + 1 < target_num_episodes:
                        selected_agent_idx[idx] = teammate_ids[idx][episodes_elapsed_per_thread[idx] + 1]

                    cell_values[0][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                        self.device)
                    cell_values[1][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                        self.device)
                    agent_prev_action[idx] = np.zeros(list(agent_prev_action[idx].shape))
                    episodes_elapsed_per_thread[idx] = min(episodes_elapsed_per_thread[idx] + 1, target_num_episodes)
                    episode_length_counter[idx] = 0

        logger.log_item(
            f"Returns/train/discounted",
            np.mean(total_returns_discounted),
            checkpoint=logging_id)
        logger.log_item(
            f"Returns/train/nondiscounted",
            np.mean(total_returns_undiscounted),
            checkpoint=logging_id)


    def eval_gen_policy_performance(self, adhoc_agent, agent_population, logger, logging_id):
        # Get required data from selected agents
        target_num_episodes = self.config.run["num_eval_episodes"] // self.config.env.parallel["eval_collection"]
        env1 = gym.make(
            self.config.env["name"]
        )

        act_sizes = env1.action_space.nvec[0]
        act_sizes_all = (len(env1.action_space.nvec), act_sizes)

        def make_env(env_name):
            def _make():
                env = gym.make(
                    env_name
                )
                return env

            return _make

        print(self.config.env_eval["name"])
        env = gym.vector.SyncVectorEnv([
            make_env(
                self.config.env_eval["name"]
            ) for idx in range(self.config.env.parallel["eval_collection"])
        ])

        eval_obses, _ = env.reset(
            seed=[self.config.run["seed"] + idx for idx in range(self.config.env.parallel["eval_collection"])]
        )

        episodes_elapsed_per_thread = [0] * self.config.env.parallel["eval_collection"]
        episode_length_counter = [0] * self.config.env.parallel["eval_collection"]

        total_returns_discounted = np.zeros([self.config.env.parallel["eval_collection"], target_num_episodes])
        total_returns_undiscounted = np.zeros([self.config.env.parallel["eval_collection"], target_num_episodes])

        # Set initial values for interaction
        batch_size = self.config.env.parallel["eval_collection"]
        cell_values = None
        agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])

        while any([eps < target_num_episodes for eps in episodes_elapsed_per_thread]):
            # Compute teammate_representation
            if agent_prev_action is None:
                agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])

            encoder_representation_input = np.concatenate(
                [
                    eval_obses, agent_prev_action
                ], axis=-1
            )
            agent_representation, cell_values = adhoc_agent.get_teammate_representation(
                encoder_representation_input, cell_values
            )

            rl_agent_representation = agent_representation.detach()
            ah_obs = torch.tensor(eval_obses).double().to(self.device)[:, 0, :]
            ah_policy_input = torch.cat([ah_obs, rl_agent_representation], dim=-1)
            ah_agents_acts, ah_agent_log_prob = adhoc_agent.decide_acts(ah_policy_input, True)

            final_acts = ah_agents_acts
            eval_nobses, rews, dones, _, infos = env.step(final_acts)

            agent_prev_action = np.zeros([batch_size, act_sizes_all[0], act_sizes_all[-1]])
            agent_prev_action[:,0,:] = agent_population.to_one_hot(final_acts)
            eval_obses = eval_nobses

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if episodes_elapsed_per_thread[idx] < total_returns_undiscounted.shape[1]:
                    total_returns_undiscounted[idx][episodes_elapsed_per_thread[idx]] += rews[idx]
                    total_returns_discounted[idx][episodes_elapsed_per_thread[idx]] += (
                        self.config.train["gamma"]**episode_length_counter[idx]
                    )*rews[idx]
                    episode_length_counter[idx] += 1

                if flag:
                    cell_values[0][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                        self.device)
                    cell_values[1][idx] = torch.zeros([self.config.model["agent_rep_size"]]).double().to(
                        self.device)
                    agent_prev_action[idx] = np.zeros(list(agent_prev_action[idx].shape))
                    episodes_elapsed_per_thread[idx] = min(episodes_elapsed_per_thread[idx] + 1, target_num_episodes)
                    episode_length_counter[idx] = 0

        print("total_returns_discounted : ", total_returns_discounted)
        print("episodes_elapsed_per_thread : ", episodes_elapsed_per_thread)
        logger.log_item(
            f"Returns/generalise/discounted",
            np.mean(total_returns_discounted),
            checkpoint=logging_id)
        logger.log_item(
            f"Returns/generalise/nondiscounted",
            np.mean(total_returns_undiscounted),
            checkpoint=logging_id)

    def run(self):
        """
            A method that encompasses the main training loop for UDRL.
        """

        # Initialize environment, agent population model & experience replay based on obs vector sizes
        env1 = gym.make(
            self.config.env["name"]
        )

        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)
        act_sizes = env1.action_space.nvec[0]
        act_sizes_all = (len(env1.action_space.nvec), act_sizes)

        obs_size_list = list(obs_sizes)
        real_obs_size = tuple(obs_size_list)

        def make_env(env_name):
            def _make():
                env = gym.make(
                    env_name
                )
                return env

            return _make

        env = gym.vector.SyncVectorEnv([
            make_env(
                self.config.env["name"]
            ) for idx in range(self.config.env.parallel["adhoc_collection"])
        ])

        test_obses, _ = env.reset(
            seed=[self.config.run["seed"] + idx for idx in range(self.config.env.parallel["adhoc_collection"])])

        device = torch.device("cuda" if self.config.run['use_cuda'] and torch.cuda.is_available() else "cpu")
        obs_size = test_obses.shape[-1] + self.config.populations["num_populations"]

        self.create_directories()

        # Create teammates' neural networks that will be loaded for AHT training
        agent_population = Agents(obs_size, agent_o_size, obs_sizes[0], self.config.populations["num_populations"],
                                  self.config, act_sizes, device, self.logger, mode="load")

        agent_population.load_model(self.config.env["model_id"])

        exp_replay_obs_size = list(real_obs_size)
        exp_replay_obs_size[-1] = exp_replay_obs_size[-1] - self.config.populations["num_populations"] + act_sizes

        self.exp_replay = EpisodicSelfPlayExperienceReplay(
            exp_replay_obs_size, list(act_sizes_all),
            max_episodes=self.config.env.parallel["adhoc_collection"],
            max_eps_length=self.config.train["timesteps_per_update"]
        )

        adhoc_agent = AdhocAgent(
            obs_size-self.config.populations["num_populations"],
            agent_o_size-self.config.populations["num_populations"],
            self.config, act_sizes, device, self.logger
        )

        # Save randomly initialized NN or load from pre-existing parameters if specified in argparse.
        # TODO Change to ad hoc agent
        if self.config.run["load_from_checkpoint"] == -1:
            adhoc_agent.save_model(0, save_model=self.logger.save_model)
            self.eval_gen_policy_performance(adhoc_agent, agent_population, self.logger, 0)
            self.eval_train_policy_performance(adhoc_agent, agent_population, self.logger, 0)
        else:
            adhoc_agent.load_model(self.config.run["load_from_checkpoint"])

        # Compute number of episodes required for training in each checkpoint.
        checkpoints_elapsed = self.config.run["load_from_checkpoint"] if self.config.run["load_from_checkpoint"] != -1 else 0
        total_checkpoints = self.config.run["total_checkpoints"]
        timesteps_per_checkpoint = self.config.run["num_timesteps"] // (total_checkpoints * (self.config.env.parallel["adhoc_collection"]))

        for ckpt_id in range(checkpoints_elapsed, total_checkpoints):
            # Record number of episodes that has elapsed in a checkpoint

            self.stored_obs, _ = env.reset()
            self.prev_cell_values = None
            self.prev_agent_actions = None

            print(f"Checkpoint: {ckpt_id}")
            timesteps_elapsed = 0

            while timesteps_elapsed < timesteps_per_checkpoint:
                # Gather self play data
                print(timesteps_elapsed, timesteps_per_checkpoint)
                self.adhoc_data_gathering(
                    env, adhoc_agent, agent_population, tuple(exp_replay_obs_size), act_sizes_all
                )

                timesteps_elapsed += self.config.train["timesteps_per_update"]
                batches = self.exp_replay.sample_all()

                all_agent_representations = torch.cat(self.agent_representation_list, dim=1)
                adhoc_agent.update(batches, all_agent_representations)
                self.prev_cell_values = (
                    self.prev_cell_values[0].detach(),
                    self.prev_cell_values[1].detach()
                )
                self.exp_replay = EpisodicSelfPlayExperienceReplay(
                    exp_replay_obs_size, list(act_sizes_all),
                    max_episodes=self.config.env.parallel["adhoc_collection"],
                    max_eps_length=self.config.train["timesteps_per_update"]
                )

            # Eval policy after sufficient number of episodes were collected.
            adhoc_agent.save_model(ckpt_id + 1, save_model=self.logger.save_model)
            self.eval_train_policy_performance(adhoc_agent, agent_population, self.logger, ckpt_id + 1)
            self.eval_gen_policy_performance(adhoc_agent, agent_population, self.logger, ckpt_id + 1)

            if self.logger:
                self.logger.commit()

        
        exp_replay_obs_size = list(real_obs_size)    
        exp_replay_obs_size[-1] = exp_replay_obs_size[-1] - self.config.populations["num_populations"] + act_sizes    
        self.exp_replay = EpisodicSelfPlayExperienceReplay(    
            exp_replay_obs_size, list(act_sizes_all),    
            max_episodes=self.config.env.parallel["adhoc_collection"],    
            max_eps_length=self.config.train["timesteps_per_update"]    
        )    
        adhoc_agent = AdhocAgent(    
            obs_size-self.config.populations["num_populations"],    
            agent_o_size-self.config.populations["num_populations"],    
            self.config, act_sizes, device, self.logger    
        )    
        # Save randomly initialized NN or load from pre-existing parameters if specified in argparse.    
        # TODO Change to ad hoc agent    
        if self.config.run["load_from_checkpoint"] == -1:    
            adhoc_agent.save_model(0)    
            self.eval_gen_policy_performance(adhoc_agent, agent_population, self.logger, 0)    
            self.eval_train_policy_performance(adhoc_agent, agent_population, self.logger, 0)    
        else:    
            adhoc_agent.load_model(self.config.run["load_from_checkpoint"])    
        # Compute number of episodes required for training in each checkpoint.    
        checkpoints_elapsed = self.config.run["load_from_checkpoint"] if self.config.run["load_from_checkpoint"] != -1 else 0    
        total_checkpoints = self.config.run["total_checkpoints"]    
        timesteps_per_checkpoint = self.config.run["num_timesteps"] // (total_checkpoints * (self.config.env.parallel["adhoc_collection"]))    
        for ckpt_id in range(checkpoints_elapsed, total_checkpoints):    
            # Record number of episodes that has elapsed in a checkpoint    
            self.stored_obs, _ = env.reset()    
            self.prev_cell_values = None    
            self.prev_agent_actions = None    
            print(f"Checkpoint: {ckpt_id}")    
            timesteps_elapsed = 0    
            while timesteps_elapsed < timesteps_per_checkpoint:    
                # Gather self play data    
                self.adhoc_data_gathering(    
                    env, adhoc_agent, agent_population, tuple(exp_replay_obs_size), act_sizes_all    
                )    
                timesteps_elapsed += self.config.train["timesteps_per_update"]    
                batches = self.exp_replay.sample_all()    
                all_agent_representations = torch.cat(self.agent_representation_list, dim=1)    
                adhoc_agent.update(batches, all_agent_representations)    
                self.prev_cell_values = (    
                    self.prev_cell_values[0].detach(),    
                    self.prev_cell_values[1].detach()    
                )    
                self.exp_replay = EpisodicSelfPlayExperienceReplay(    
                    exp_replay_obs_size, list(act_sizes_all),    
                    max_episodes=self.config.env.parallel["adhoc_collection"],    
                    max_eps_length=self.config.train["timesteps_per_update"]    
                )    
            # Eval policy after sufficient number of episodes were collected.    
            adhoc_agent.save_model(ckpt_id + 1)    
            self.eval_train_policy_performance(adhoc_agent, agent_population, self.logger, ckpt_id + 1)    
            self.eval_gen_policy_performance(adhoc_agent, agent_population, self.logger, ckpt_id + 1)    
            if self.logger:    
                self.logger.commit()    

class Logger:    
    def __init__(self, config):    
        logger_period = config.logger.logger_period    
        self.steps_per_update = (config.env.parallel.adhoc_collection) * config.train.timesteps_per_update    
        self.save_model = config.logger.get("save_model", False)
        if logger_period < 1:    
            # Frequency    
            self.train_log_period = int(logger_period * config.run.num_timesteps // self.steps_per_update)    
        else:    
            # Period    
            self.train_log_period = logger_period    
        self.verbose = config.logger.get("verbose", False)    
        self.run = wandb.init(    
            project=config.logger.project,    
            entity=config.logger.entity,    
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),    
            tags=config.logger.get("tags", None),    
            notes=config.logger.get("notes", None),    
            group=config.logger.get("group", None),    
            mode=config.logger.get("mode", None),    
            reinit=True,    
        )    
        self.define_metrics()    
    def log(self, data, step=None, commit=False):    
        wandb.log(data, step=step, commit=commit)    
    def log_item(self, tag, val, step=None, commit=False, **kwargs):    
        self.log({tag: val, **kwargs}, step=step, commit=commit)    
        if self.verbose:    
            print(f"{tag}: {val}")    
    def commit(self):    
        self.log({}, commit=True)    
    def define_metrics(self):    
        wandb.define_metric("train_step")    
        wandb.define_metric("checkpoint")    
        wandb.define_metric("Train/*", step_metric="train_step")    
        wandb.define_metric("Returns/*", step_metric="checkpoint")    

