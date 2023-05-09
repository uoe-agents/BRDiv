import numpy as np
import copy

class EpisodicSelfPlayExperienceReplay(object):
    """
        Class that encapsulates the experience replay used for UDRL.
    """
    def __init__(self, ob_shape, act_shape, max_episodes=100000, max_eps_length=20):
        """
            Constructor of the experience replay class.
            :param ob_shape: Observation shape.
            :param act_shape: Action space dimensionality.
            :param max_episodes: Maximum number of stored episodes.
            :param max_eps_length: Maximum length of each episode.
        """

        real_ob_shape = [ob_shape[0]]
        real_ob_shape.extend(list(ob_shape))

        self.num_episodes = 0
        self.pointer = 0
        self.size = 0
        self.max_episodes = max_episodes
        self.max_eps_length = max_eps_length
        self.ob_shape = ob_shape

        obs_shape, acts_shape = [max_episodes, max_eps_length], [max_episodes, max_eps_length]
        obs_shape.extend(list(ob_shape))
        acts_shape.extend(list(act_shape))

        stored_orders_shape = copy.deepcopy(obs_shape)
        stored_orders_shape[-1] = 1

        self.obs = np.zeros(obs_shape)
        self.actions = np.zeros(acts_shape)
        self.next_obs = np.zeros(obs_shape)
        self.dones = np.zeros([max_episodes, max_eps_length])
        self.rewards = np.zeros([max_episodes, max_eps_length])

    def get_num_episodes(self):
        """
            A method that returns the number of episodes stored in the experience replay.
            :return: Number of episodes stored in the replay.
        """
        return self.num_episodes

    def get_size(self):
        """
            A method that returns the number of transitions stored in the experience replay.
            :return: Number of transitions stored in the replay.
        """
        return self.size

    def add_episode(self, obses, acts, rewards, dones, next_obses):
    #def add_episode(self, obses, acts, rewards, total_rew, eps_length, gamma):
        """
            A method to add an episode of experiences into the replay buffer. Target returns that are appended
            with obs are changed in hindsight according to the achieved returns.
            :param obses: List of stored obses.
            :param acts: List of stored acts.
            :param rewards: List of rewards per timestep.
            :param total_rew: The total returns for an episode.
            :param eps_length: The length of an episode.
            :param gamma: The discount rate used.
        """

        eps_length = self.max_eps_length

        rewards = rewards[:eps_length]
        self.obs[self.pointer,:eps_length] = obses[:eps_length]
        self.actions[self.pointer,:eps_length] = acts[:eps_length]
        self.rewards[self.pointer,:eps_length] = rewards[:eps_length]
        self.next_obs[self.pointer,:eps_length] = next_obses[:eps_length]
        self.dones[self.pointer, :eps_length] = dones[:eps_length]

        self.size = min(self.size + eps_length + self.max_eps_length, self.max_episodes * self.max_eps_length)
        self.num_episodes = min(self.num_episodes + 1, self.max_episodes)
        self.pointer = (self.pointer + 1) % self.max_episodes

    def sample_all(self):
        """
            A method to return everything stored in the buffer.
            :return: Everything contained in buffer.
        """
        return self.obs, self.actions, self.next_obs, self.dones, self.rewards

    def save(self, dir_location):
        """
            A method that stores every variable into disk.
        """
        with open(dir_location+'/obs.npy', 'wb') as f:
            np.save(f, self.obs)
        with open(dir_location + '/n_obs.npy', 'wb') as f:
            np.save(f, self.next_obs)
        with open(dir_location+'/actions.npy', 'wb') as f:
            np.save(f, self.actions)
        with open(dir_location+'/dones.npy', 'wb') as f:
            np.save(f, self.dones)
        with open(dir_location+'/num_episodes.npy', 'wb') as f:
            np.save(f, np.asarray([self.num_episodes]))
        with open(dir_location + '/size.npy', 'wb') as f:
            np.save(f, np.asarray([self.size]))
        with open(dir_location+'/pointer.npy', 'wb') as f:
            np.save(f, np.asarray([self.pointer]))
        with open(dir_location + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)

    def load(self, dir_location):
        """
            A method that loads experiences stored within a disk.
        """
        self.obs = np.load(dir_location+"/obs.npy")
        self.next_obs = np.load(dir_location+"n_obs.npy")
        self.actions = np.load(dir_location+"/actions.npy")
        self.num_episodes = np.load(dir_location + "/num_episodes.npy")[0]
        self.pointer = np.load(dir_location + "/pointer.npy")[0]
        self.dones = np.load(dir_location + "/dones.npy")
        self.rewards = np.load(dir_location + "/rewards.npy")
        self.size = np.load(dir_location+"/size.npy")

class EpisodicCrossPlayExperienceReplay(object):
    """
        Class that encapsulates the experience replay used for UDRL.
    """
    def __init__(self, ob_shape, act_shape, max_episodes=100000, max_eps_length=20):
        """
            Constructor of the experience replay class.
            :param ob_shape: Observation shape.
            :param act_shape: Action space dimensionality.
            :param max_episodes: Maximum number of stored episodes.
            :param max_eps_length: Maximum length of each episode.
        """

        real_ob_shape = [ob_shape[0]]
        real_ob_shape.extend(list(ob_shape))

        self.num_episodes = 0
        self.pointer = 0
        self.size = 0
        self.max_episodes = max_episodes
        self.max_eps_length = max_eps_length
        self.ob_shape = real_ob_shape

        obs_shape, acts_shape = [max_episodes, max_eps_length], [max_episodes, max_eps_length]
        obs_shape.extend(list(real_ob_shape))
        acts_shape.extend(list(act_shape))

        self.obs = np.zeros(obs_shape)
        self.actions = np.zeros(acts_shape)
        self.next_obs = np.zeros(obs_shape)
        self.dones = np.zeros([max_episodes, max_eps_length])
        self.rewards = np.zeros([max_episodes, max_eps_length])

    def get_num_episodes(self):
        """
            A method that returns the number of episodes stored in the experience replay.
            :return: Number of episodes stored in the replay.
        """
        return self.num_episodes

    def get_size(self):
        """
            A method that returns the number of transitions stored in the experience replay.
            :return: Number of transitions stored in the replay.
        """
        return self.size

    def add_episode(self, obses, acts, rewards, next_obses, dones):
    #def add_episode(self, obses, acts, rewards, total_rew, eps_length, gamma):
        """
            A method to add an episode of experiences into the replay buffer. Target returns that are appended
            with obs are changed in hindsight according to the achieved returns.
            :param obses: List of stored obses.
            :param acts: List of stored acts.
            :param rewards: List of rewards per timestep.
            :param total_rew: The total returns for an episode.
            :param eps_length: The length of an episode.
            :param gamma: The discount rate used.
        """
        eps_length = self.max_eps_length

        rewards = rewards[:eps_length]
        self.obs[self.pointer,:eps_length] = obses[:eps_length]
        self.actions[self.pointer,:eps_length] = acts[:eps_length]
        self.rewards[self.pointer,:eps_length] = rewards[:eps_length]
        self.next_obs[self.pointer,:eps_length] = next_obses[:eps_length]
        self.dones[self.pointer, :eps_length] = dones[:eps_length]

        self.size = min(self.size + self.max_eps_length, self.max_episodes * self.max_eps_length)
        self.num_episodes = min(self.num_episodes + 1, self.max_episodes)
        self.pointer = (self.pointer + 1) % self.max_episodes

    def sample_all(self):
        """
            A method to return everything stored in the buffer.
            :return: Everything contained in buffer.
        """
        return self.obs, self.actions, self.rewards, self.dones, self.next_obs

    def save(self, dir_location):
        """
            A method that stores every variable into disk.
        """
        with open(dir_location+'/obs.npy', 'wb') as f:
            np.save(f, self.obs)
        with open(dir_location + '/n_obs.npy', 'wb') as f:
            np.save(f, self.next_obs)
        with open(dir_location+'/actions.npy', 'wb') as f:
            np.save(f, self.actions)
        with open(dir_location+'/dones.npy', 'wb') as f:
            np.save(f, self.dones)
        with open(dir_location + '/size.npy', 'wb') as f:
            np.save(f, np.asarray([self.size]))
        with open(dir_location+'/pointer.npy', 'wb') as f:
            np.save(f, np.asarray([self.pointer]))
        with open(dir_location + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)

    def load(self, dir_location):
        """
            A method that loads experiences stored within a disk.
        """
        self.obs = np.load(dir_location+"/obs.npy")
        self.next_obs = np.load(dir_location+"n_obs.npy")
        self.actions = np.load(dir_location+"/actions.npy")
        self.pointer = np.load(dir_location + "/pointer.npy")[0]
        self.dones = np.load(dir_location + "/dones.npy")
        self.rewards = np.load(dir_location + "/rewards.npy")
        self.size = np.load(dir_location+"/size.npy")

class SelfPlayExperienceReplay(object):
    """
        Class that encapsulates the experience replay used for diversity training.
    """
    def __init__(self, ob_shape, act_shape, max_transitions=100000):
        """
            Constructor of the experience replay class.
            :param ob_shape: Observation shape.
            :param act_shape: Action space dimensionality.
            :param max_episodes: Maximum number of stored episodes.
            :param max_eps_length: Maximum length of each episode.
        """

        real_ob_shape = [ob_shape[0]]
        real_ob_shape.extend(list(ob_shape))

        self.num_episodes = 0
        self.pointer = 0
        self.size = 0
        self.max_transitions = max_transitions
        self.ob_shape = ob_shape

        obs_shape, acts_shape = [max_transitions], [max_transitions]
        obs_shape.extend(list(ob_shape))
        acts_shape.extend(list(act_shape))

        stored_orders_shape = copy.deepcopy(obs_shape)
        stored_orders_shape[-1] = 1

        self.obs = np.zeros(obs_shape)
        self.actions = np.zeros(acts_shape)
        self.next_obs = np.zeros(obs_shape)
        self.dones = np.zeros([max_transitions, 1])
        self.rewards = np.zeros([max_transitions, 1])

    def get_size(self):
        """
            A method that returns the number of transitions stored in the experience replay.
            :return: Number of transitions stored in the replay.
        """
        return self.size

    def add_experience(self, obs, act, rew, done, n_obs):
        """
            A method to add an experience into the replay buffer.
            :param obs: Stored obses.
            :param act: Stored acts.
            :param rew: Stored reward.
            :param n_obs: Stored next observation.
            :param done: Stored done flag.
        """

        self.obs[self.pointer] = obs
        self.actions[self.pointer] = act
        self.rewards[self.pointer,0] = rew
        self.next_obs[self.pointer] = n_obs
        self.dones[self.pointer,0] = done

        self.size = min(self.size + 1, self.max_transitions)
        self.pointer = (self.pointer + 1) % self.max_transitions

    def sample(self, num_samples):
        """
            A method to sample (obs, acts) from replay buffer.
            :param num_samples: Number of sampled (obs, acts) tuples.
            :return: A numpy array of sampled obses and sampled acts.
        """
        sampled_trx_idx = np.random.randint(self.size, size=num_samples)

        sampled_obs = np.concatenate([np.expand_dims(self.obs[trx_id], axis=0) for trx_id in sampled_trx_idx], axis=0)
        sampled_acts = np.concatenate([np.expand_dims(self.actions[trx_id], axis=0) for trx_id in sampled_trx_idx], axis=0)
        sampled_rews = np.concatenate([self.rewards[trx_id] for trx_id in sampled_trx_idx], axis=0)
        sampled_n_obses = np.concatenate([np.expand_dims(self.next_obs[trx_id], axis=0) for trx_id in sampled_trx_idx], axis=0)
        sampled_dones = np.concatenate([self.dones[trx_id] for trx_id in sampled_trx_idx], axis=0)

        return sampled_obs, sampled_acts, sampled_n_obses, sampled_dones, sampled_rews

    def sample_all(self):
        """
            A method to return everything stored in the buffer.
            :return: Everything contained in buffer.
        """
        #return self.obs[:self.size], self.actions[:self.size], self.next_obs[:self.size], self.dones[:self.size], self.rewards[:self.size]

        sampled_obs = np.concatenate([np.expand_dims(self.obs[trx_id], axis=0) for trx_id in range(self.size)], axis=0)
        sampled_acts = np.concatenate([np.expand_dims(self.actions[trx_id], axis=0) for trx_id in range(self.size)],
                                      axis=0)
        sampled_rews = np.concatenate([self.rewards[trx_id] for trx_id in range(self.size)], axis=0)
        sampled_n_obses = np.concatenate([np.expand_dims(self.next_obs[trx_id], axis=0) for trx_id in range(self.size)],
                                         axis=0)
        sampled_dones = np.concatenate([self.dones[trx_id] for trx_id in range(self.size)], axis=0)

        return sampled_obs, sampled_acts, sampled_n_obses, sampled_dones, sampled_rews

    def save(self, dir_location):
        """
            A method that stores every variable into disk.
        """
        with open(dir_location+'/obs.npy', 'wb') as f:
            np.save(f, self.obs)
        with open(dir_location + '/n_obs.npy', 'wb') as f:
            np.save(f, self.next_obs)
        with open(dir_location+'/actions.npy', 'wb') as f:
            np.save(f, self.actions)
        with open(dir_location+'/dones.npy', 'wb') as f:
            np.save(f, self.dones)
        with open(dir_location+'/eps_lengths.npy', 'wb') as f:
            np.save(f, self.eps_lengths)
        with open(dir_location+'/num_episodes.npy', 'wb') as f:
            np.save(f, np.asarray([self.num_episodes]))
        with open(dir_location + '/size.npy', 'wb') as f:
            np.save(f, np.asarray([self.size]))
        with open(dir_location+'/pointer.npy', 'wb') as f:
            np.save(f, np.asarray([self.pointer]))
        with open(dir_location + '/rewards.npy', 'wb') as f:
            np.save(f, self.rewards)

    def load(self, dir_location):
        """
            A method that loads experiences stored within a disk.
        """
        self.obs = np.load(dir_location+"/obs.npy")
        self.next_obs = np.load(dir_location+"n_obs.npy")
        self.actions = np.load(dir_location+"/actions.npy")
        self.eps_lengths = np.load(dir_location+"/eps_lengths.npy")
        self.num_episodes = np.load(dir_location + "/num_episodes.npy")[0]
        self.pointer = np.load(dir_location + "/pointer.npy")[0]
        self.dones = np.load(dir_location + "/dones.npy")
        self.rewards = np.load(dir_location + "/rewards.npy")
        self.size = np.load(dir_location+"/size.npy")


class CrossPlayExperienceReplay(object):
    """
        Class that encapsulates the experience replay used for UDRL.
    """
    def __init__(self, ob_shape, act_shape, max_transitions=1000000):
        """
            Constructor of the experience replay class.
            :param ob_shape: Observation shape.
            :param act_shape: Action space dimensionality.
            :param max_episodes: Maximum number of stored episodes.
            :param max_eps_length: Maximum length of each episode.
        """

        real_ob_shape = [ob_shape[0]]
        real_ob_shape.extend(list(ob_shape))

        self.size = 0
        self.pointer = 0
        self.max_transitions = max_transitions
        self.ob_shape = real_ob_shape

        obs_shape, acts_shape = [max_transitions], [max_transitions]
        obs_shape.extend(list(real_ob_shape))
        acts_shape.extend(list(act_shape))

        self.obs = np.zeros(obs_shape)
        self.actions = np.zeros(acts_shape)
        self.next_obs = np.zeros(obs_shape)
        self.rewards = np.zeros([max_transitions, 1])
        self.dones = np.zeros([max_transitions, 1])


    def get_size(self):
        """
            A method that returns the number of episodes stored in the experience replay.
            :return: Number of episodes stored in the replay.
        """
        return self.size

    def add_experience(self, obs, act, rew, n_obs, done):
        """
            A method to add an experience into the replay buffer.
            :param obs: Stored obses.
            :param act: Stored acts.
            :param rew: Stored reward.
            :param n_obs: Stored next observation.
            :param done: Stored done flag.
        """

        self.obs[self.pointer] = obs
        self.actions[self.pointer] = act
        self.rewards[self.pointer,0] = rew
        self.next_obs[self.pointer] = n_obs
        self.dones[self.pointer,0] = done

        self.size = min(self.size + 1, self.max_transitions)
        self.pointer = (self.pointer + 1) % self.max_transitions

    def sample(self, num_samples):
        """
            A method to sample (obs, acts) from replay buffer.
            :param num_samples: Number of sampled (obs, acts) tuples.
            :return: A numpy array of sampled obses and sampled acts.
        """
        sampled_trx_idx = np.random.randint(self.size, size=num_samples)

        sampled_obs = np.concatenate([np.expand_dims(self.obs[trx_id], axis=0) for trx_id in sampled_trx_idx], axis=0)
        sampled_acts = np.concatenate([np.expand_dims(self.actions[trx_id], axis=0) for trx_id in sampled_trx_idx], axis=0)
        sampled_rews = np.concatenate([self.rewards[trx_id] for trx_id in sampled_trx_idx], axis=0)
        sampled_n_obses = np.concatenate([np.expand_dims(self.next_obs[trx_id], axis=0) for trx_id in sampled_trx_idx], axis=0)
        sampled_dones = np.concatenate([self.dones[trx_id] for trx_id in sampled_trx_idx], axis=0)

        return sampled_obs, sampled_acts, sampled_rews, sampled_dones, sampled_n_obses

    def sample_all(self):
        """
            A method to return everything stored in the buffer.
            :return: Everything contained in buffer.
        """
        sampled_obs = np.concatenate([np.expand_dims(self.obs[trx_id], axis=0) for trx_id in range(self.size)], axis=0)
        sampled_acts = np.concatenate([np.expand_dims(self.actions[trx_id], axis=0) for trx_id in range(self.size)],
                                      axis=0)
        sampled_rews = np.concatenate([self.rewards[trx_id] for trx_id in range(self.size)], axis=0)
        sampled_n_obses = np.concatenate([np.expand_dims(self.next_obs[trx_id], axis=0) for trx_id in range(self.size)],
                                         axis=0)
        sampled_dones = np.concatenate([self.dones[trx_id] for trx_id in range(self.size)], axis=0)

        return sampled_obs, sampled_acts, sampled_rews, sampled_dones, sampled_n_obses

    def save(self, dir_location):
        """
            A method that stores every variable into disk.
        """
        with open(dir_location+'/obs.npy', 'wb') as f:
            np.save(f, self.obs)
        with open(dir_location+'/actions.npy', 'wb') as f:
            np.save(f, self.actions)
        with open(dir_location+'/rews.npy', 'wb') as f:
            np.save(f, self.rewards)
        with open(dir_location+'/next_obs.npy', 'wb') as f:
            np.save(f, self.next_obs)
        with open(dir_location+'/dones.npy', 'wb') as f:
            np.save(f, self.dones)
        with open(dir_location+'/size.npy', 'wb') as f:
            np.save(f, np.asarray([self.size]))
        with open(dir_location+'/pointer.npy', 'wb') as f:
            np.save(f, np.asarray([self.pointer]))

    def load(self, dir_location):
        """
            A method that loads experiences stored within a disk.
        """
        self.obs = np.load(dir_location+"/obs.npy")
        self.actions = np.load(dir_location+"/actions.npy")
        self.rews = np.load(dir_location + "/rews.npy")
        self.next_obs = np.load(dir_location + "/next_obs.npy")
        self.dones = np.load(dir_location + "/dones.npy")

        self.size = np.load(dir_location + "/size.npy")[0]
        self.pointer = np.load(dir_location + "/pointer.npy")[0]
