import numpy as np
import torch.distributions as dist
import torch
import torch.nn.functional as F

class PLASTIC_Policy_Agent(object):
    def __init__(self, trained_agent_br_pair, config, device):
        self.trained_agent_br_pair = trained_agent_br_pair
        self.config = config
        self.num_populations = self.config.populations["num_populations"]
        self.type_belief_prior = None
        self.likelihood_eta = self.config.plastic["eta"]
        self.device = device

    def load_agent_model(self, int_id):
        self.trained_agent_br_pair.load_model(int_id)

    def get_teammate_representation(self, obses, prev_acts):
        obs_tensor = torch.tensor(obses).double().to(self.device)
        batch_size, num_agents, num_features = obs_tensor.size()

        repeated_obs_tensor = torch.repeat_interleave(obs_tensor, self.num_populations, dim = 0)
        id_input = torch.eye(self.num_populations).unsqueeze(1).repeat(batch_size, num_agents, 1)
        in_population_agent_id = torch.eye(num_agents).repeat(batch_size*self.num_populations, 1, 1)

        obs_w_commands = torch.cat([repeated_obs_tensor, id_input, in_population_agent_id], dim=-1)
        act_logits = self.trained_agent_br_pair.separate_act_select(obs_w_commands)[:, 1, :]

        particle_dist = dist.Categorical(logits=act_logits)
        prev_acts_log_likelihood = particle_dist.log_prob(
            torch.repeat_interleave(torch.tensor(prev_acts), self.num_populations).long()
        ).view(batch_size, self.num_populations)

        prev_acts_log_likelihood = 1.0 - self.likelihood_eta * (1.0 - torch.exp(prev_acts_log_likelihood))

        self.type_belief_prior = (self.type_belief_prior * prev_acts_log_likelihood)/(
            torch.sum(self.type_belief_prior * prev_acts_log_likelihood, dim=-1, keepdim=True).repeat(1, self.num_populations)
        )

    def decide_acts(self, obses):
        obs_tensor = torch.tensor(obses).double().to(self.device)
        batch_size, num_agents, num_features = obs_tensor.size()

        repeated_obs_tensor = torch.repeat_interleave(obs_tensor, self.num_populations, dim=0)
        id_input = torch.eye(self.num_populations).unsqueeze(1).repeat(batch_size, num_agents, 1)
        in_population_agent_id = torch.eye(num_agents).repeat(batch_size * self.num_populations, 1, 1)

        obs_w_commands = torch.cat([repeated_obs_tensor, id_input, in_population_agent_id], dim=-1)
        act_logits = self.trained_agent_br_pair.separate_act_select(obs_w_commands)[:, 0, :]
        particle_dist = dist.Categorical(logits=act_logits)
        chosen_acts = particle_dist.sample().view(batch_size, self.num_populations)

        if self.type_belief_prior == None:
            # Select uniform prior
            self.type_belief_prior = torch.ones(batch_size, self.num_populations).to(self.device).double() * (
                    1.0/self.num_populations
            )

        selected_type = []
        for idx in range(batch_size):
            max_prob = None
            max_id_prob = []
            for id_pop in range(self.num_populations):
                if max_prob == None or max_prob < self.type_belief_prior[idx][id_pop]:
                    max_prob = self.type_belief_prior[idx][id_pop]
                    max_id_prob = [id_pop]
                elif max_prob == self.type_belief_prior[idx][id_pop]:
                    max_id_prob.append(id_pop)

            selected_type.append(np.random.choice(max_id_prob))

        selected_type = torch.tensor(selected_type).long().to(self.device).unsqueeze(-1)

        final_chosen_acts = chosen_acts.gather(-1, selected_type).view(-1).detach().numpy().tolist()
        return final_chosen_acts
