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
from AdhocAgent import PLASTIC_Policy_Agent
from Agents import Agents
# from train import Logger
import os
import wandb
from omegaconf import OmegaConf


class AdhocEval(object):
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
        # self.logger= None

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

    def eval_train_policy_performance(self, adhoc_agent, agent_population, agent_population_dir, logger, logging_id):
        # Get required data from selected agents
        target_num_episodes = self.config.eval_params["per_seed_eval_eps"] // self.config.env.parallel["eval_collection"]
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

        all_seed_discounted = []
        all_seed_undiscounted = []
        eval_approach_name = agent_population_dir.split("/")[-1]

        for seed_id in range(self.config.eval_params["num_seeds"]):
            agent_population.load_model(self.config.env["model_id"], agent_population_dir + "/" +str(seed_id+1) +"/models")

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
            selected_agent_idx = teammate_ids[:, 0].tolist()
            batch_size = self.config.env.parallel["eval_collection"]

            adhoc_agent.type_belief_prior = None

            while any([eps < target_num_episodes for eps in episodes_elapsed_per_thread]):
                one_hot_id_shape = list(eval_obses.shape)[:-1]
                one_hot_ids = self.to_one_hot_population_id(
                    np.expand_dims(np.asarray(selected_agent_idx), axis=-1) * np.ones(one_hot_id_shape))

                # Decide agent's action based on model and execute.
                real_input = np.concatenate([eval_obses, one_hot_ids], axis=-1)
                acts, act_log_prob = agent_population.decide_acts(real_input, True)

                ah_obs = torch.tensor(eval_obses).double().to(self.device)
                ah_agents_acts= adhoc_agent.decide_acts(ah_obs)

                final_acts = [(a1, a2[1]) for a1, a2 in zip(ah_agents_acts, acts)]
                eval_nobses, rews, dones, _, infos = env.step(final_acts)
                adhoc_agent.get_teammate_representation(eval_obses, np.asarray([a2[1] for a2 in acts]))

                eval_obses = eval_nobses

                # TODO Change agent id in finished envs.
                for idx, flag in enumerate(dones):
                    # If an episode collected by one of the threads ends...
                    if episodes_elapsed_per_thread[idx] < total_returns_undiscounted.shape[1]:
                        total_returns_undiscounted[idx][episodes_elapsed_per_thread[idx]] += rews[idx]
                        total_returns_discounted[idx][episodes_elapsed_per_thread[idx]] += (
                                                                                                   self.config.train[
                                                                                                       "gamma"] **
                                                                                                   episode_length_counter[
                                                                                                       idx]
                                                                                           ) * rews[idx]
                        episode_length_counter[idx] += 1

                    if flag:
                        if episodes_elapsed_per_thread[idx] + 1 < target_num_episodes:
                            selected_agent_idx[idx] = teammate_ids[idx][episodes_elapsed_per_thread[idx] + 1]
                        episodes_elapsed_per_thread[idx] = min(episodes_elapsed_per_thread[idx] + 1, target_num_episodes)
                        adhoc_agent.type_belief_prior[idx] = torch.ones([self.config.populations["num_populations"]])/(
                                self.config.populations["num_populations"]+0.0
                        )
                        episode_length_counter[idx] = 0

            all_seed_discounted.append(np.mean(total_returns_discounted))
            all_seed_undiscounted.append(np.mean(total_returns_undiscounted))

        print(eval_approach_name, all_seed_discounted)
        logger.log_item(
            f"Returns/gen_xp/discounted/{eval_approach_name}",
            sum(all_seed_discounted)/(len(all_seed_discounted)+0.0),
            checkpoint=logging_id)
        logger.log_item(
            f"Returns/gen_xp/nondiscounted/{eval_approach_name}",
            sum(all_seed_undiscounted)/(len(all_seed_undiscounted)+0.0),
            checkpoint=logging_id)

    def eval_gen_policy_performance(self, adhoc_agent, agent_population, logger, logging_id, teammate_type=1):
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

        evaluated_env_name = self.config.env_eval["name"][:-3] + str(teammate_type) + self.config.env_eval["name"][-3:]
        env = gym.vector.SyncVectorEnv([
            make_env(
                evaluated_env_name
            ) for idx in range(self.config.env.parallel["eval_collection"])
        ])

        eval_obses, _ = env.reset(
            seed=[self.config.run["seed"] + idx for idx in range(self.config.env.parallel["eval_collection"])]
        )

        episodes_elapsed_per_thread = [0] * self.config.env.parallel["eval_collection"]
        episode_length_counter = [0] * self.config.env.parallel["eval_collection"]

        total_returns_discounted = np.zeros([self.config.env.parallel["eval_collection"], target_num_episodes])
        total_returns_undiscounted = np.zeros([self.config.env.parallel["eval_collection"], target_num_episodes])

        while any([eps < target_num_episodes for eps in episodes_elapsed_per_thread]):
            # Compute teammate_representation

            ah_obs = torch.tensor(eval_obses).double().to(self.device)
            ah_agents_acts = adhoc_agent.decide_acts(ah_obs)

            final_acts = ah_agents_acts
            eval_nobses, rews, dones, _, infos = env.step(final_acts)
            if not "teammate_actions" in infos.keys():
                infos["teammate_actions"] = np.zeros(self.config.env.parallel["eval_collection"])
            adhoc_agent.get_teammate_representation(eval_obses, infos["teammate_actions"])
            eval_obses = eval_nobses

            # TODO Change agent id in finished envs.
            for idx, flag in enumerate(dones):
                # If an episode collected by one of the threads ends...
                if episodes_elapsed_per_thread[idx] < total_returns_undiscounted.shape[1]:
                    total_returns_undiscounted[idx][episodes_elapsed_per_thread[idx]] += rews[idx]
                    total_returns_discounted[idx][episodes_elapsed_per_thread[idx]] += (
                                                                                               self.config.train[
                                                                                                   "gamma"] **
                                                                                               episode_length_counter[
                                                                                                   idx]
                                                                                       ) * rews[idx]
                    episode_length_counter[idx] += 1

                if flag:
                    episodes_elapsed_per_thread[idx] = min(episodes_elapsed_per_thread[idx] + 1, target_num_episodes)
                    episode_length_counter[idx] = 0
                    adhoc_agent.type_belief_prior[idx] = torch.ones([self.config.populations["num_populations"]]) / (
                            self.config.populations["num_populations"] + 0.0
                    )

        print("total_returns_discounted : ", total_returns_discounted)
        print("episodes_elapsed_per_thread : ", episodes_elapsed_per_thread)
        logger.log_item(
            f"Returns/generalise/discounted/H"+str(teammate_type),
            np.mean(total_returns_discounted),
            checkpoint=logging_id)
        logger.log_item(
            f"Returns/generalise/nondiscounted/H"+str(teammate_type),
            np.mean(total_returns_undiscounted),
            checkpoint=logging_id)

    def evaluate(self):
        """
            A method that encompasses the main training loop for UDRL.
        """

        # Initialize environment, agent population model & experience replay based on obs vector sizes
        env1 = gym.make(
            self.config.env["name"]
        )

        obs_sizes, agent_o_size = self.get_obs_sizes(env1.observation_space)
        act_sizes = env1.action_space.nvec[0]

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

        adhoc_agent = PLASTIC_Policy_Agent(
            agent_population, self.config, device
        )

        # Save randomly initialized NN or load from pre-existing parameters if specified in argparse.
        # TODO Change to ad hoc agent

        if self.config.eval_params["eval_mode"] == "heuristic":
            # Compute number of episodes required for training in each checkpoint.
            for teammate_type_id in range(self.config.eval_params["num_eval_heuristics"]):
                # Record number of episodes that has elapsed in a checkpoint
                # Eval policy after sufficient number of episodes were collected.
                adhoc_agent.type_belief_prior = None
                self.eval_gen_policy_performance(
                    adhoc_agent, agent_population, self.logger, 0, teammate_type=teammate_type_id+1
                )
        else:
            parameter_directory = self.config.eval_params["all_params_dir"]
            subfolders = [f.path for f in os.scandir(parameter_directory) if f.is_dir()]

            eval_agent_population = Agents(
                obs_size, agent_o_size, obs_sizes[0], self.config.populations["num_populations"],
                self.config, act_sizes, device, self.logger, mode="load"
            )

            for evaluated_folder in subfolders:
                self.eval_train_policy_performance(
                    adhoc_agent, eval_agent_population, evaluated_folder, self.logger, 0
                )

        if self.logger:
            self.logger.commit()

class Logger:
    def __init__(self, config):
        logger_period = config.logger.logger_period
        #self.steps_per_update = (config.env.parallel.adhoc_collection) * config.train.timesteps_per_update
        self.save_model = config.logger.get("save_model", False)
        #if logger_period < 1:
            # Frequency
        #    self.train_log_period = int(logger_period * config.run.num_timesteps // self.steps_per_update)
        #else:
            # Period
            #self.train_log_period = logger_period
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

