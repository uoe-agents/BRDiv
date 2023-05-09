from Network import fc_network
from torch import optim, nn
from torch.autograd import Variable
import numpy as np
import torch.distributions as dist
import torch
#import pyro
import torch.nn.functional as F

class Agents(object):
    """
        A class that encapsulates the joint policy model controlling each agent population.
    """
    def __init__(self, obs_size, agent_o_size, num_agents, num_populations, configs, act_sizes, device, logger, mode="train"):
        """
            :param obs_size: The length of the entire observation inputted to the model
            :param agent_o_size: The length of u (graph-level features) for the model
            :param num_agents: Number of agents in env.
            :param configs: The dictionary config containing hyperparameters of the model
            :param act_sizes: Per-agent action space dimension
            :param device: Device to store model
            :param loss_writer: Tensorboard writer to log results
            :param model_grad_writer: Tensorboard writer to log model gradient magnitude
        """
        self.obs_size = obs_size
        self.agent_o_size = agent_o_size
        self.config = configs
        self.act_size = act_sizes
        self.device = device
        self.total_updates = 0
        self.next_log_update = 0
        self.total_updates_xp = 0
        self.num_agents = num_agents
        self.parameter_sharing = self.config.model.get("parameter_sharing", True)
        self.with_any_play = self.config.any_play.get("with_any_play", False)
        self.separate_model_in_populations = self.config.model.get("separate_model_in_populations", True)

        self.mode = mode
        self.id_length = self.config.populations["num_populations"]
        self.num_populations = self.config.populations["num_populations"]
        self.effective_crit_obs_size = obs_size
        if not self.parameter_sharing:
            self.effective_crit_obs_size += self.num_agents

        if self.separate_model_in_populations:
            if not self.parameter_sharing:
                actor_dims = [self.obs_size-self.num_populations, *self.config.model.actor_dims, self.act_size]
            else:
                actor_dims = [
                    self.obs_size-self.num_populations+self.num_agents, *self.config.model.actor_dims, self.act_size
                ]
        else:
            if not self.parameter_sharing:
                actor_dims = [self.obs_size, *self.config.model.actor_dims, self.act_size]
            else:
                actor_dims = [
                    self.obs_size+self.num_agents, *self.config.model.actor_dims, self.act_size
                ]

        critic_dims = [self.num_agents * (self.obs_size + self.num_agents), *self.config.model.critic_dims, 1]

        if self.with_any_play:
            classifier_dims = [
                self.num_agents * (self.obs_size - self.num_populations), *self.config.any_play.classifier_dims,
                self.num_populations
            ]
        init_ortho = self.config.model.get("init_ortho", True)

        if self.separate_model_in_populations:
            if not self.parameter_sharing:
                self.joint_policy = [
                    fc_network(actor_dims, init_ortho).double().to(self.device)
                    for _ in range(2*self.num_populations)
                ]
            else:
                self.joint_policy = [
                    fc_network(actor_dims, init_ortho).double().to(self.device)
                    for _ in range(self.num_populations)
                ]
        else:
            if not self.parameter_sharing:
                self.joint_policy = [
                    fc_network(actor_dims, init_ortho).double().to(self.device),
                    fc_network(actor_dims, init_ortho).double().to(self.device)
                ]
            else:
                self.joint_policy = fc_network(actor_dims, init_ortho).double().to(self.device)

        self.joint_action_value_function = \
            fc_network(critic_dims, init_ortho).double().to(self.device)
        self.target_joint_action_value_function = \
            fc_network(critic_dims, init_ortho).double().to(self.device)

        if self.with_any_play:
            self.pop_classifier = fc_network(classifier_dims, init_ortho).double().to(self.device)

        self.hard_copy()

        # Initialize optimizer
        params_list = None
        if not self.with_any_play:
            params_list = [
                *self.joint_action_value_function.parameters(),
                *(param for actor in self.joint_policy for param in actor.parameters())
               ]
        else:
            params_list = params_list = [
                *self.joint_action_value_function.parameters(),
                *(param for actor in self.joint_policy for param in actor.parameters()),
                *self.pop_classifier.parameters(),
               ]

        self.optimizer = optim.Adam(
            params_list,
            lr=self.config.train["lr"]
        )

        self.diversity_loss = self.config.loss_weights["diversity_loss"]
        self.logger = logger

    def to_one_hot_population_id(self, indices):
        act_indices = np.asarray(indices).astype(int)
        one_hot_ids = np.eye(self.num_populations)[act_indices]

        return one_hot_ids

    def to_one_hot(self, actions):
        """
            A method that changes agents' actions into a one-hot encoding format.
            :param actions: Agents' actions in form of an integer.
            :return: Agents' actions in a one-hot encoding form.
        """
        act_indices = np.asarray(actions).astype(int)
        one_hot_acts = np.eye(self.act_size)[act_indices]
        return one_hot_acts

    def separate_act_select(self, input):
        additional_input_length = self.num_agents
        per_id_input = None
        if self.parameter_sharing:
            per_id_input = [
                torch.cat([
                    input[input[:, :, -(self.num_populations + additional_input_length) + id] == 1][:, :-(self.id_length+additional_input_length)],
                    input[input[:, :, -(self.num_populations + additional_input_length) + id] == 1][:, -additional_input_length:],
                ], dim=-1)
                for id in range(self.num_populations)
            ]
        else:
            per_id_input = [None] * (2 * self.num_populations)
            for pop_id in range(self.num_populations):
                for a_id in range(self.num_agents):
                    per_id_input[2*pop_id+a_id] = input[
                                                    torch.logical_and(
                                                        input[:, :, -(self.num_populations + additional_input_length) + pop_id] == 1,
                                                        input[:, :, -self.num_agents + a_id] == 1
                                                    )
                                                ][:, :-(self.id_length+additional_input_length)]

        per_id_input_filtered = [(idx,inp) for idx, inp in enumerate(per_id_input) if not inp.nelement() == 0]
        executed_models = [policy for idx, policy in enumerate(self.joint_policy) if
                           not per_id_input[idx].nelement() == 0]

        futures = [
            torch.jit.fork(model, per_id_input_filtered[i][1]) for i, model
            in enumerate(executed_models)
        ]

        results = [torch.jit.wait(fut) for fut in futures]
        logits = torch.zeros([input.size()[0], input.size()[1], self.act_size]).double().to(self.device)

        id = 0
        for idx, _ in per_id_input_filtered:
            if self.parameter_sharing:
                logits[input[:, :, -(self.num_populations + additional_input_length) + idx] == 1] = results[id]
            else:
                population_id = idx // 2
                agent_id = idx % 2
                logits[
                    torch.logical_and(
                        input[:, :, -(self.num_populations + additional_input_length) + population_id] == 1,
                        input[:, :, -self.num_agents + agent_id] == 1
                    )
                ] = results[id]
            id += 1

        return logits

    def decide_acts(self, obs_w_commands, with_log_probs=False, eval=False):
        """
            A method to decide the actions of agents given obs & target returns.
            :param obs_w_commands: A numpy array that has the obs concatenated with the target returns.
            :return: Sampled actions under the specific obs.
        """
        obs_w_commands = torch.tensor(obs_w_commands).to(self.device)
        batch_size, num_agents = obs_w_commands.size()[0], obs_w_commands.size()[1]

        if self.parameter_sharing or self.separate_model_in_populations:
            in_population_agent_id = torch.eye(num_agents).repeat(batch_size, 1, 1)
            obs_w_commands = torch.cat([obs_w_commands, in_population_agent_id], dim=-1)

        if self.separate_model_in_populations:
            act_logits = self.separate_act_select(obs_w_commands)
        else:
            if self.parameter_sharing:
                act_logits = self.joint_policy(obs_w_commands)
            else:
                futures = [
                    torch.jit.fork(model, obs_w_commands[:,i:i+1,:]) for i, model
                    in enumerate(self.joint_policy)
                ]

                results = [torch.jit.wait(fut) for fut in futures]
                act_logits = torch.cat(results, dim=1)

        if not eval:
            particle_dist = dist.OneHotCategorical(logits=act_logits)
            original_acts = particle_dist.sample()
            acts = original_acts.argmax(dim=-1)
        else:
            particle_dist = dist.Categorical(logits=act_logits)
            original_acts = torch.argmax(act_logits, dim=-1)
            acts = original_acts

        acts_list = acts.tolist()

        if with_log_probs:
            return acts_list, torch.exp(particle_dist.log_prob(original_acts))
        return acts_list

    def compute_pop_class_loss(self, obs_batch):

        real_input = obs_batch[:,:,:,:-self.num_populations].reshape(obs_batch.size()[0] * obs_batch.size()[1], -1)
        output_classes = obs_batch[:,:,:1,-self.num_populations:].reshape(obs_batch.size()[0] * obs_batch.size()[1], -1)

        logits = self.pop_classifier(real_input)
        pop_categorical_dist = dist.OneHotCategorical(logits=logits)

        return -pop_categorical_dist.log_prob(output_classes).mean()

    def compute_additional_rewards(self, obs):
        obs_tensor = torch.tensor(obs).to(self.device)
        obs_only = obs_tensor[:,:,:-self.num_populations].reshape(obs_tensor.size()[0],-1)
        pop_only = obs_tensor[:,:1,-self.num_populations:].reshape(obs_tensor.size()[0],-1)

        pop_logits = self.pop_classifier(obs_only)
        pop_log_likelihood = dist.OneHotCategorical(logits=pop_logits).log_prob(pop_only)

        return pop_log_likelihood

    def compute_jsd_loss_v2(self, obs_batch, acts_batch):
        comparator_prob = None
        batch_size, num_steps, num_agents = obs_batch.size()[0], obs_batch.size()[1], obs_batch.size()[2]

        action_probs_per_population = []
        agent_real_ids = obs_batch[:, :, :, self.obs_size - self.num_populations: self.obs_size].argmax(dim=-1)

        for idx in range(self.num_populations):
            original_states = obs_batch[:, :, :, :self.obs_size - self.num_populations]
            original_states = original_states.view(batch_size*num_steps, num_agents, -1)

            pop_annot = torch.zeros_like(
                obs_batch.view(
                    batch_size*num_steps, num_agents, -1
                )[:, :, self.obs_size - self.num_populations:self.obs_size]).double().to(self.device)
            pop_annot[:, :, idx] = 1.0

            if self.parameter_sharing or self.separate_model_in_populations:
                comparator_input = torch.cat([original_states, pop_annot, torch.eye(self.num_agents).repeat(original_states.size()[0],1,1)], dim=-1)
            else:
                comparator_input = torch.cat([original_states, pop_annot], dim=-1)

            if self.separate_model_in_populations:
                comparator_act_logits = self.separate_act_select(comparator_input)
            else:
                if self.parameter_sharing:
                    comparator_act_logits = self.joint_policy(comparator_input)
                else:
                    futures = [
                        torch.jit.fork(model, comparator_input[:, i:i + 1, :]) for i, model
                        in enumerate(self.joint_policy)
                    ]

                    results = [torch.jit.wait(fut) for fut in futures]
                    comparator_act_logits = torch.cat(results, dim=1)

            comparator_act_logits = comparator_act_logits.view(batch_size, num_steps, num_agents, -1)

            action_logits = dist.OneHotCategorical(logits=comparator_act_logits).log_prob(acts_batch)
            action_probs_per_population.append(action_logits.unsqueeze(-1))

            if comparator_prob is None:
                comparator_prob = torch.exp(action_logits)
            else:
                comparator_prob = comparator_prob + torch.exp(action_logits)

        action_logits_per_population = torch.cat(action_probs_per_population, dim=-1)

        temp_pi= torch.gather(action_logits_per_population, -1, agent_real_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        log_pi_i = temp_pi.sum(dim=-1)

        temp_pi_hat = action_logits_per_population.sum(dim=-2).sum(dim=-2)
        log_pi_hat = torch.log(torch.exp(temp_pi_hat).mean(dim=-1))

        summed_term_list = []
        separate_delta_list = []
        per_pop_delta = []
        for t in range(num_steps):
            multiplier = self.config.train["gamma_act_jsd"]**(torch.abs(t-torch.tensor(list(range(num_steps))).to(self.device)))
            multiplier = multiplier.unsqueeze(0).repeat(batch_size,1)
            delta_hat_var = action_logits_per_population.sum(dim=-2)
            separate_deltas = (temp_pi * multiplier).sum(dim=-1)
            log_average_only_delta = (delta_hat_var * multiplier.unsqueeze(-1).repeat(1, 1, self.num_populations)).sum(dim=-2)
            per_pop_delta.append(log_average_only_delta.unsqueeze(-2))
            average_only_delta = torch.log(torch.exp(log_average_only_delta).mean(dim=-1))

            separate_delta_list.append(separate_deltas.unsqueeze(-1))
            summed_term_list.append(average_only_delta.unsqueeze(-1))

        stacked_summed_term_list = torch.cat(summed_term_list, dim=-1)
        stacked_separate_delta_list = torch.cat(separate_delta_list, dim=-1)
        stacked_pop_delta = torch.cat(per_pop_delta, dim=-2)
        repeated_stacked_summed_term_list = stacked_summed_term_list.unsqueeze(-1).repeat(1, 1, self.num_populations)

        calculated_logs = repeated_stacked_summed_term_list - stacked_pop_delta
        calc_logs_mean_ovt = calculated_logs.mean(dim=-2)

        is_ratio = torch.exp(temp_pi_hat - log_pi_i.unsqueeze(-1).repeat(1, self.num_populations))
        jsd_loss = (is_ratio.detach() * calc_logs_mean_ovt).mean(dim=-1).mean()

        return jsd_loss

    def compute_jsd_loss(self, obs_batch, acts_batch):
        comparator_prob = None
        batch_size, num_steps, num_agents = obs_batch.size()[0], obs_batch.size()[1], obs_batch.size()[2]

        action_probs_per_population = []
        agent_real_ids = obs_batch[:, :, :, self.obs_size - self.num_populations: self.obs_size].argmax(dim=-1)

        for idx in range(self.num_populations):
            original_states = obs_batch[:, :, :, :self.obs_size - self.num_populations]
            original_states = original_states.view(batch_size*num_steps, num_agents, -1)

            pop_annot = torch.zeros_like(
                obs_batch.view(
                    batch_size*num_steps, num_agents, -1
                )[:, :, self.obs_size - self.num_populations:self.obs_size]).double().to(self.device)
            pop_annot[:, :, idx] = 1.0

            if self.parameter_sharing or self.separate_model_in_populations:
                comparator_input = torch.cat([original_states, pop_annot, torch.eye(self.num_agents).repeat(original_states.size()[0],1,1)], dim=-1)
            else:
                comparator_input = torch.cat([original_states, pop_annot], dim=-1)

            if self.separate_model_in_populations:
                comparator_act_logits = self.separate_act_select(comparator_input)
            else:
                comparator_act_logits = self.joint_policy(comparator_input)

            comparator_act_logits = comparator_act_logits.view(batch_size, num_steps, num_agents, -1)

            action_logits = dist.OneHotCategorical(logits=comparator_act_logits).log_prob(acts_batch)
            action_probs_per_population.append(action_logits.unsqueeze(-1))

            if comparator_prob is None:
                comparator_prob = torch.exp(action_logits)
            else:
                comparator_prob = comparator_prob + torch.exp(action_logits)

        action_logits_per_population = torch.cat(action_probs_per_population, dim=-1)

        temp_pi= torch.gather(action_logits_per_population, -1, agent_real_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        log_pi_i = temp_pi.sum(dim=-1)

        temp_pi_hat = action_logits_per_population.sum(dim=-2).sum(dim=-2)
        log_pi_hat = torch.log(torch.exp(temp_pi_hat).mean(dim=-1))

        summed_term_list = []
        separate_delta_list = []
        for t in range(num_steps):
            multiplier = self.config.train["gamma_act_jsd"]**(torch.abs(t-torch.tensor(list(range(num_steps))).to(self.device)))
            multiplier = multiplier.unsqueeze(0).repeat(batch_size,1)

            delta_hat_var = action_logits_per_population.sum(dim=-2)
            separate_deltas = (temp_pi * multiplier).sum(dim=-1)
            log_average_only_delta = (delta_hat_var * multiplier.unsqueeze(-1).repeat(1, 1, self.num_populations)).sum(dim=-2)
            average_only_delta = torch.log(torch.exp(log_average_only_delta).mean(dim=-1))

            separate_delta_list.append(separate_deltas.unsqueeze(-1))
            summed_term_list.append(average_only_delta.unsqueeze(-1))

        stacked_summed_term_list = torch.cat(summed_term_list, dim=-1)
        stacked_separate_delta_list = torch.cat(separate_delta_list, dim=-1)

        pi_hat_per_pi_i = torch.exp(log_pi_hat - log_pi_i)
        term1 = pi_hat_per_pi_i.unsqueeze(-1).repeat(1, num_steps) * torch.exp(stacked_separate_delta_list)
        final_term1 = term1.detach() * stacked_separate_delta_list

        term2_mult = torch.exp(stacked_summed_term_list) - (stacked_separate_delta_list/self.num_populations)
        final_term2 = term2_mult.detach() * log_pi_i.unsqueeze(-1).repeat(1, num_steps)

        jsd_loss = (final_term1 + final_term2).mean(dim=-1).mean()

        return jsd_loss

    def onehot_from_logits(self, logits):
        argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).double()

        return argmax_acs

    def compute_sp_critic_loss(
            self, obs_batch, n_obs_batch,
            acts_batch, sp_rew_batch, sp_done_batch,
            input_graph
    ):
        batch_size = obs_batch.size()[0]
        obs_length = obs_batch.size()[1]
        obs_only_length = self.effective_crit_obs_size - self.num_populations
        if not self.parameter_sharing:
            obs_only_length -= self.num_agents

        predicted_values = []
        target_values = []
        baseline_xp_matrics = []
        opt_xp_matrics = []
        action_likelihood = []
        action_entropy = []

        # Compute added stuff related to index
        xp_agent_indices = [
            (i, j) for i in range(self.num_populations) for j in range(self.num_populations)
        ]

        agent1_array = np.tile(np.asarray([x[0] for x in xp_agent_indices]), [batch_size])
        agent2_array = np.tile(np.asarray([x[1] for x in xp_agent_indices]), [batch_size])

        one_hot_ids = np.expand_dims(self.to_one_hot_population_id(agent1_array), axis=1)
        one_hot_ids2 = np.expand_dims(self.to_one_hot_population_id(agent2_array), axis=1)

        one_hot_ids, one_hot_ids2 = torch.tensor(one_hot_ids).to(self.device), torch.tensor(one_hot_ids2).to(
            self.device)
        one_hot_id_all = torch.cat([one_hot_ids, one_hot_ids2], dim=1)

        target_value = None
        target_diversity_value = None
        for idx in reversed(range(obs_length)):

            obs_idx = obs_batch[:,idx,:,:]
            acts_idx = acts_batch[:, idx, :, :]
            n_obs_idx = n_obs_batch[:,idx,:,:]

            if not self.parameter_sharing:
                obs_idx = torch.cat([obs_batch[:,idx,:,:], torch.eye(self.num_agents).repeat(batch_size,1,1)], dim=-1)
                n_obs_idx = torch.cat([n_obs_batch[:,idx,:,:], torch.eye(self.num_agents).repeat(batch_size, 1, 1)], dim=-1)

            sp_v_values = self.joint_action_value_function(obs_idx.view(obs_idx.size(0), 1, -1).squeeze(1))

            sp_rl_rew = sp_rew_batch[:, idx]
            sp_rl_done = sp_done_batch[:, idx]

            if idx == obs_length-1:
                target_value = self.target_joint_action_value_function(n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1))
                target_diversity_value = self.joint_action_value_function(n_obs_idx.view(n_obs_idx.size(0), 1, -1).squeeze(1))

            target_value = (
                    sp_rl_rew.view(-1, 1) + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_value)
            ).detach()

            predicted_values.append(sp_v_values)
            target_values.append(target_value)

            obs_only = obs_idx[:, :, :obs_only_length]
            r_obs_only = obs_only.repeat([1, self.num_populations ** 2, 1]).view(
                -1, obs_only.size()[-2], obs_only.size()[-1]
            )

            xp_input = torch.cat([r_obs_only, one_hot_id_all], dim=-1)
            if not self.parameter_sharing:
                xp_input = torch.cat([xp_input, torch.eye(self.num_agents).repeat(xp_input.size()[0], 1, 1)], dim=-1)

            xp_value_input_graph = None

            baseline_matrix = self.joint_action_value_function(
                xp_input.view(xp_input.size(0), 1, -1).squeeze(1)
            ).view(batch_size, self.num_populations,
                   self.num_populations)

            target_diversity_value = (
                    sp_rl_rew.view(-1, 1) + (self.config.train["gamma"] * (1 - sp_rl_done.view(-1, 1)) * target_diversity_value)
            ).detach()

            offset = self.num_populations
            if not self.parameter_sharing:
                offset += self.num_agents

            if -offset + self.num_populations == 0:
                replacement_index =obs_idx[:,:,-offset:].argmax(dim=-1)
            else:
                replacement_index = obs_idx[:, :, -offset:-offset+self.num_populations].argmax(dim=-1)

            idx1, idx2 = replacement_index[:, 0], replacement_index[:, 1]

            rearranged_bm = baseline_matrix.detach().clone().view(-1, 1)
            replaced_indices = torch.tensor(
                [
                    i * (self.num_populations ** 2) for i in range(baseline_matrix.size()[0])
                ]).to(self.device) + idx1 * self.num_populations + idx2

            rearranged_bm[replaced_indices] = target_diversity_value
            bm_real = rearranged_bm.view(batch_size, self.num_populations,
                                         self.num_populations)

            if self.separate_model_in_populations:
                action_logits = self.separate_act_select(obs_idx)
            else:
                if self.parameter_sharing:
                    action_logits = self.joint_policy(
                        obs_idx
                    )
                else:
                    obs_idx = obs_idx[:,:,:-self.num_agents]
                    futures = [
                        torch.jit.fork(model, obs_idx[:,i:i+1,:]) for i, model
                        in enumerate(self.joint_policy)
                    ]

                    results = [torch.jit.wait(fut) for fut in futures]
                    action_logits = torch.cat(results, dim=1)


            #print("Action logits : ", action_logits[0])
            action_distribution = dist.OneHotCategorical(logits=action_logits)
            action_likelihood.append(action_distribution.log_prob(acts_idx))
            action_entropy.append(action_distribution.entropy())

            #+ self.config.loss_weights["entropy_regularizer_loss"] * selected_action_reg

            opt_xp_matrics.append(bm_real)
            baseline_xp_matrics.append(baseline_matrix)

        predicted_values = torch.cat(predicted_values, dim=0)
        all_target_values = torch.cat(target_values, dim=0)
        all_baseline_matrices = torch.cat(baseline_xp_matrics, dim=0)
        all_opt_matrices = torch.cat(opt_xp_matrics, dim=0)
        action_entropies = torch.cat(action_entropy, dim=0)

        sp_critic_loss = (0.5 * ((predicted_values - all_target_values) ** 2)).mean()

        baseline_diversity_values, _, _ = self.diversity_loss_computation(all_baseline_matrices)
        opt_diversity_values, opt_sp_component, opt_div_component = self.diversity_loss_computation(all_opt_matrices)
        action_log_likelihoods = torch.cat(action_likelihood, dim=0).sum(dim=-1)

        entropy_loss = self.config.loss_weights["entropy_regularizer_loss"] * -action_entropies.sum(dim=-1).mean()

        pol_loss = -(action_log_likelihoods * -(opt_diversity_values-baseline_diversity_values)).mean()
        return sp_critic_loss, pol_loss, entropy_loss, opt_div_component.mean()

    def compute_xp_critic_loss(self, xp_obs, xp_acts, xp_rew, xp_dones, xp_n_obses):
        batch_size = xp_obs.shape[0]
        obs_length = xp_obs.shape[1]
        xp_num_agents = xp_obs.shape[2]
        obs_only_length = self.effective_crit_obs_size - self.num_populations
        if not self.parameter_sharing:
            obs_only_length -= self.num_agents

        predicted_values = []
        target_values = []
        action_likelihood = []
        opt_xp_matrics = []
        baseline_xp_matrics = []
        action_entropies = []

        xp_obs = torch.tensor(xp_obs).double().to(self.device)
        xp_n_obses = torch.tensor(xp_n_obses).double().to(self.device)
        xp_rew = torch.tensor(xp_rew).double().to(self.device)
        xp_dones = torch.tensor(xp_dones).double().to(self.device)
        xp_acts = torch.tensor(xp_acts).double().to(self.device)

        xp_targ_values = None

        # Compute added stuff related to index
        xp_agent_indices = [
            (i, j) for i in range(self.num_populations) for j in range(self.num_populations)
        ]

        agent1_array = np.tile(np.asarray([x[0] for x in xp_agent_indices]), [batch_size])
        agent2_array = np.tile(np.asarray([x[1] for x in xp_agent_indices]), [batch_size])

        one_hot_ids = np.expand_dims(self.to_one_hot_population_id(agent1_array), axis=1)
        one_hot_ids2 = np.expand_dims(self.to_one_hot_population_id(agent2_array), axis=1)

        one_hot_ids, one_hot_ids2 = torch.tensor(one_hot_ids).to(self.device), torch.tensor(one_hot_ids2).to(
            self.device)
        one_hot_id_all = torch.cat([one_hot_ids, one_hot_ids2], dim=1)

        for idx in reversed(range(obs_length)):
            xp_critic_state_input = []
            xp_obs_id = xp_obs[:, idx, :, :, :]
            xp_acts_idx = xp_acts[:, idx, :, :]
            for agent_id in range(xp_num_agents):
                xp_critic_state_input.append(torch.unsqueeze(xp_obs_id[:, agent_id, agent_id, :], dim=1))

            xp_critic_state_input = torch.cat(
                xp_critic_state_input, dim=1
            )

            batch_size = xp_critic_state_input.size()[0]
            if not self.parameter_sharing:
                xp_critic_state_input = torch.cat(
                    [xp_critic_state_input, torch.eye(self.num_agents).repeat([batch_size, 1, 1])],
                    dim=-1
                )

            input_graph = None
            xp_v_values = self.joint_action_value_function(
                xp_critic_state_input.view(xp_critic_state_input.size(0), 1, -1).squeeze(1)
            )

            if idx == obs_length - 1:
                xp_critic_n_state_input = []
                xp_n_obses_id = xp_n_obses[:, idx, :, :, :]
                for agent_id in range(xp_num_agents):
                    xp_critic_n_state_input.append(torch.unsqueeze(xp_n_obses_id[:, agent_id, agent_id, :], dim=1))

                xp_critic_n_state_input = torch.cat(
                    xp_critic_n_state_input, dim=1
                )

                if not self.parameter_sharing:
                    xp_critic_n_state_input = torch.cat(
                        [xp_critic_n_state_input, torch.eye(self.num_agents).repeat([batch_size, 1, 1])],
                        dim=-1
                    )

                input_graph = None
                xp_targ_values = self.target_joint_action_value_function(
                    xp_critic_n_state_input.view(xp_critic_n_state_input.size(0), 1, -1).squeeze(1)
                )

                xp_targ_diversity_value = self.joint_action_value_function(
                    xp_critic_n_state_input.view(xp_critic_n_state_input.size(0), 1, -1).squeeze(1)
                )

            xp_rl_rew = xp_rew[:, idx]
            xp_rl_done = xp_dones[:, idx]

            xp_targ_values = (
                xp_rl_rew.view(-1, 1) + (self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_values)
            ).detach()

            predicted_values.append(xp_v_values)
            target_values.append(xp_targ_values)

            xp_obs_only = xp_critic_state_input[:, :, :obs_only_length]
            r_xp_obs_only = xp_obs_only.repeat([1, self.num_populations ** 2, 1]).view(
                -1, xp_obs_only.size()[-2], xp_obs_only.size()[-1]
            )

            xp_input = torch.cat([r_xp_obs_only, one_hot_id_all], dim=-1)
            if not self.parameter_sharing:
                xp_input = torch.cat([xp_input, torch.eye(self.num_agents).repeat(xp_input.size()[0], 1, 1)], dim=-1)

            xp_value_input_graph = None

            baseline_matrix = self.joint_action_value_function(
                xp_input.view(xp_input.size(0), 1, -1).squeeze(1),
            ).view(batch_size, self.num_populations, self.num_populations)

            xp_targ_diversity_value = (
                    xp_rl_rew.view(-1, 1) + (
                        self.config.train["gamma"] * (1 - xp_rl_done.view(-1, 1)) * xp_targ_diversity_value)
            ).detach()

            offset = self.num_populations
            if not self.parameter_sharing:
                offset += self.num_agents

            if -offset + self.num_populations == 0:
                replacement_index = xp_critic_state_input[:, :, -offset:].argmax(dim=-1)
            else:
                replacement_index = xp_critic_state_input[:, :, -offset:-offset + self.num_populations].argmax(dim=-1)

            idx1, idx2 = replacement_index[:, 0], replacement_index[:, 1]
            rearranged_bm = baseline_matrix.detach().clone().view(-1, 1)
            replaced_indices = torch.tensor(
                [
                    i * (self.num_populations ** 2) for i in range(baseline_matrix.size()[0])
                ]
            ).to(self.device) + idx1 * self.num_populations + idx2

            rearranged_bm[replaced_indices] = xp_targ_diversity_value
            bm_real = rearranged_bm.view(batch_size, self.num_populations,
                                         self.num_populations)

            torch.set_printoptions(profile="full")
            xp_act_log_input = xp_obs_id.reshape(-1, xp_obs_id.size()[2],xp_obs_id.size()[3])

            if not self.parameter_sharing:
                xp_act_log_input = torch.cat(
                    [xp_act_log_input, torch.eye(self.num_agents).repeat(xp_act_log_input.size()[0], 1, 1)],
                    dim=-1
                )

            if self.separate_model_in_populations:
                action_logits = self.separate_act_select(xp_act_log_input)
            else:
                if self.parameter_sharing:
                    action_logits = self.joint_policy(
                        xp_act_log_input
                    )
                else:
                    xp_act_log_input = xp_act_log_input[:, :, :-self.num_agents]
                    futures = [
                        torch.jit.fork(model, xp_act_log_input[:,i:i+1,:]) for i, model
                        in enumerate(self.joint_policy)
                    ]

                    results = [torch.jit.wait(fut) for fut in futures]
                    action_logits = torch.cat(results, dim=1)

            action_logits = action_logits.view(
                xp_obs_id.size()[0], xp_obs_id.size()[1], xp_obs_id.size()[2], -1
            )

            final_selected_logits = []
            for a_id in range(xp_num_agents):
                final_selected_logits.append(action_logits[:, a_id, a_id:(a_id + 1), :])

            final_selected_logits = torch.cat(final_selected_logits, dim=1)
            #print("Action logits XP : ", final_selected_logits[0])
            action_distribution = dist.OneHotCategorical(logits=final_selected_logits)
            action_likelihood.append(action_distribution.log_prob(xp_acts_idx))

            action_entropies.append(action_distribution.entropy())
            opt_xp_matrics.append(bm_real)
            baseline_xp_matrics.append(baseline_matrix)

        predicted_values = torch.cat(predicted_values, dim=0)
        all_target_values = torch.cat(target_values, dim=0)
        all_baseline_matrices = torch.cat(baseline_xp_matrics, dim=0)
        all_opt_matrices = torch.cat(opt_xp_matrics, dim=0)

        xp_critic_loss = (0.5 * ((predicted_values - all_target_values) ** 2)).mean()

        baseline_diversity_values, _, _ = self.diversity_loss_computation(all_baseline_matrices)
        opt_diversity_values, opt_sp_component, opt_div_component = self.diversity_loss_computation(all_opt_matrices)
        action_log_likelihoods = torch.cat(action_likelihood, dim=0).sum(dim=-1)
        action_entropies = torch.cat(action_entropies, dim=0)

        entropy_loss = self.config.loss_weights["entropy_regularizer_loss"] * -action_entropies.sum(dim=-1).mean()

        xp_pol_loss = -(action_log_likelihoods * -(opt_diversity_values - baseline_diversity_values)).mean()
        return xp_critic_loss, xp_pol_loss, entropy_loss, opt_div_component.mean()

    def update(self, batches, xp_batches):
        """
            A method that updates the joint policy model following sampled self-play and cross-play experiences.
            :param batches: A batch of obses and acts sampled from self-play experience replay.
            :param xp_batches: A batch of experience from cross-play experience replay.
        """

        # Get obs and acts batch and prepare inputs to model.
        obs_batch, acts_batch = torch.tensor(batches[0]).to(self.device), torch.tensor(batches[1]).to(self.device)
        sp_n_obs_batch = torch.tensor(batches[2]).to(self.device)
        sp_done_batch = torch.tensor(batches[3]).double().to(self.device)
        rewards_batch = torch.tensor(batches[4]).double().to(self.device)
        batch_size, num_steps, num_agents = obs_batch.size()[0], obs_batch.size()[1], obs_batch.size()[2]

        # Prepare graph structure to GNNs
        input_graph = None

        self.optimizer.zero_grad()
        # Compute SP Critic Loss
        sp_critic_loss, sp_pol_loss, sp_action_entropies, sp_pred_div = self.compute_sp_critic_loss(
            obs_batch, sp_n_obs_batch, acts_batch, rewards_batch, sp_done_batch, input_graph
        )

        #total_critic_loss = sp_critic_loss * self.config.loss_weights["sp_val_loss_weight"]

        # Get XP data and preprocess it for matrix computation
        total_xp_critic_loss, total_xp_actor_loss, total_xp_entropy_loss, xp_pred_div = 0, 0, 0, 0
        if self.config.loss_weights["xp_loss_weights"] != 0:
            xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses = xp_batches

            total_xp_critic_loss, total_xp_actor_loss, total_xp_entropy_loss, xp_pred_div = self.compute_xp_critic_loss(
                xp_obs, xp_acts, xp_rews, xp_dones, xp_n_obses
            )

        act_diff_log_mean = self.compute_jsd_loss_v2(obs_batch, acts_batch)
        if self.with_any_play:
            total_classifier_loss = self.config.any_play["any_play_classifier_loss_weight"] * self.compute_pop_class_loss(obs_batch)

        total_critic_loss = sp_critic_loss * self.config.loss_weights["sp_val_loss_weight"] + total_xp_critic_loss * self.config.loss_weights["xp_val_loss_weight"]
        total_actor_loss = sp_pol_loss + total_xp_actor_loss + sp_action_entropies + total_xp_entropy_loss + (act_diff_log_mean*self.config.loss_weights["jsd_weight"])

        # Write losses to logs
        if self.total_updates >= self.next_log_update == 0:
            self.next_log_update += self.logger.train_log_period
            train_step = self.total_updates * self.logger.steps_per_update
            self.logger.log_item("Train/sp/actor_loss", sp_pol_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/sp/critic_loss", sp_critic_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/sp/entropy", sp_action_entropies,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/sp/pred_div", sp_pred_div,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/jsd_loss", act_diff_log_mean*self.config.loss_weights["jsd_weight"],
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/xp/actor_loss", total_xp_actor_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/xp/critic_loss", total_xp_critic_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/xp/entropy", total_xp_entropy_loss,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.log_item("Train/xp/pred_div", xp_pred_div,
                                 train_step=train_step, updates=self.total_updates)
            self.logger.commit()

        # Backpropagate critic loss
        if not self.with_any_play:
            (total_critic_loss+total_actor_loss).backward()
        else:
            (total_critic_loss + total_actor_loss + total_classifier_loss).backward()

        # Clip grads if necessary
        if self.config.train['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(self.joint_action_value_function.parameters(), self.config.train['max_grad_norm'])

            if self.separate_model_in_populations:
                for model in self.joint_policy:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])
            else:
                if self.parameter_sharing:
                    nn.utils.clip_grad_norm_(self.joint_policy.parameters(), self.config.train['max_grad_norm'])
                else:
                    for model in self.joint_policy:
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.train['max_grad_norm'])

        # Log grad magnitudes if specified by config.
        if self.config.logger["log_grad"]:

            if self.separate_model_in_populations:
                for idx, model in enumerate(self.joint_policy):
                    for name, param in model.named_parameters():
                        if not param.grad is None:
                            self.logger.log_item(
                                f"Train/grad/actor_{idx}_{name}",
                                torch.abs(param.grad).mean(),
                                train_step=self.total_updates
                                )
            else:
                for name, param in self.joint_policy.named_parameters():
                    if not param.grad is None:
                        self.logger.log_item(
                            f"Train/grad/actor_{name}",
                            torch.abs(param.grad).mean(),
                            train_step=self.total_updates
                            )
            for name, param in self.joint_action_value_function.named_parameters():
                if not param.grad is None:
                    self.logger.log_item(
                        f"Train/grad/critic_{name}",
                        torch.abs(param.grad).mean(),
                        train_step=self.total_updates
                        )

        self.optimizer.step()
        self.soft_copy(self.config.train["target_update_rate"])

        self.total_updates += 1

    def _rank3_trace(self, x):
        return torch.einsum('ijj->i', x)

    def _rank3_diag(self, x):
        eye = torch.eye(x.size(1)).type_as(x)
        out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
        return out

    def diversity_loss_computation(self, xp_matrix, return_weighted=True):
        num_populations = xp_matrix.size()[-1]
        cross_play_matrix_diagonal = torch.diagonal(xp_matrix, dim1=-2, dim2=-1)

        div_loss = None

        if self.diversity_loss == "determinantal":
            adjacency_matrix = ((xp_matrix + torch.permute(xp_matrix, [0, 2, 1])) / 2.0)
            # adjacency_matrix = xp_matrix
            vals_idx1 = adjacency_matrix.repeat([1, 1, num_populations]).view(adjacency_matrix.size()[0], -1,
                                                                              adjacency_matrix.size()[-1])
            vals_idx2 = adjacency_matrix.repeat([1, num_populations, 1])

            kernel_matrix = torch.exp(
                -torch.sum((vals_idx1 - vals_idx2) ** 2, dim=-1) / (2 * (self.config.loss_weights["scale_length"] ** 2))
            ).view(adjacency_matrix.size()[0], num_populations, num_populations)
            #print("K0 Matrix: ", kernel_matrix[0])

            div_loss = -torch.linalg.det(kernel_matrix)

        elif self.diversity_loss == "spectral":
            adjacency_matrix = ((xp_matrix + torch.permute(xp_matrix, [0, 2, 1])) / 2.0)
            # adjacency_matrix = xp_matrix
            vals_idx1 = adjacency_matrix.repeat([1, 1, num_populations]).view(adjacency_matrix.size()[0], -1,
                                                                              adjacency_matrix.size()[-1])
            vals_idx2 = adjacency_matrix.repeat([1, num_populations, 1])

            kernel_matrix = torch.exp(
                -torch.sum((vals_idx1 - vals_idx2) ** 2, dim=-1) / (2 * (self.config.loss_weights["scale_length"] ** 2))
            ).view(adjacency_matrix.size()[0], num_populations, num_populations)

            diag_cross_play = torch.diagonal(kernel_matrix, dim1=-2, dim2=-1)

            symmetric_xp_matrix = kernel_matrix + torch.diag_embed(diag_cross_play)
            degrees_matrix = torch.diag_embed(torch.sum(symmetric_xp_matrix, dim=-1))
            laplacian_matrix = degrees_matrix - symmetric_xp_matrix

            #inverted_degree = torch.diag_embed(torch.sum(symmetric_xp_matrix, dim=-1)**-0.5)
            #normalized_laplacian = torch.bmm(inverted_degree, torch.bmm(laplacian_matrix, inverted_degree))

            # Get real component of graph
            graph_spectrum = torch.sort(torch.view_as_real(torch.linalg.eigvals(laplacian_matrix))[:, :, 0], dim=-1)[0]
            #graph_spectrum_diff = 2 * torch.ones_like(graph_spectrum)
            diff_spectrum = graph_spectrum[:, 1:] - graph_spectrum[:, :-1]

            #graph_spectrum_diff[:, :-1] = diff_spectrum

            multiplier_matrix = torch.tile(
                torch.tensor([idx + 0.0 for idx in range(num_populations-1)]).unsqueeze(0),
                 (kernel_matrix.size()[0], 1)
            ).to(self.device)

            #print("Spectrum Softmax : ", F.softmax(diff_spectrum, dim=-1)[0])
            #print("Multiplier matrix: ", multiplier_matrix)
            #div_loss = -torch.sum(F.softmax(graph_spectrum_diff, dim=-1) * multiplier_matrix, dim=-1).mean()
            div_loss = -torch.sum(F.softmax(diff_spectrum, dim=-1) * multiplier_matrix, dim=-1)

        elif self.diversity_loss == "values":
            diag_cross_play = torch.diag_embed(cross_play_matrix_diagonal)
            ones_like_diag = torch.ones_like(diag_cross_play)
            diag_ones_bmm = torch.bmm(diag_cross_play, ones_like_diag)
            # print(diag_ones_bmm[0])
            # exit()
            #print("xp mat : ",xp_matrix[0])

            maximised_matrix = (diag_ones_bmm - xp_matrix).sum(dim=-1).sum(dim=-1)
            div_loss1 = -(maximised_matrix/(
                     self.num_populations * (self.num_populations-1)
            ))

            maximised_matrix2 = (diag_ones_bmm - torch.permute(xp_matrix, (0,2, 1))).sum(dim=-1).sum(dim=-1)
            div_loss2 = -(maximised_matrix2/(
                     self.num_populations * (self.num_populations-1)
            ))

            div_loss = 0.5*div_loss1 + 0.5*div_loss2

        # print("XP 0 : ",xp_matrix[0])
        diagonal_udrl_loss = -self._rank3_trace(xp_matrix)/num_populations
        final_loss = self.config.loss_weights["xp_loss_weights"]*div_loss + self.config.loss_weights["sp_rew_weight"]*diagonal_udrl_loss

        #print("L Div : ", self.config.loss_weights["xp_loss_weights"]*div_loss, " L SP : ", self.config.loss_weights["sp_rew_weight"]*diagonal_udrl_loss)

        if return_weighted:
            return final_loss.detach(), self.config.loss_weights["sp_rew_weight"]*diagonal_udrl_loss, self.config.loss_weights["xp_loss_weights"]*div_loss
        else:
            return final_loss.detach(), diagonal_udrl_loss, div_loss

    def hard_copy(self):
        for target_param, param in zip(self.target_joint_action_value_function.parameters(), self.joint_action_value_function.parameters()):
            target_param.data.copy_(param.data)

    def soft_copy(self, tau=0.001):
        for target_param, param in zip(self.target_joint_action_value_function.parameters(), self.joint_action_value_function.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def save_model(self, int_id, save_model=False):
        """
            A method to save model parameters.
            :param int_id: Integer indicating ID of the checkpoint.
        """
        if not save_model:
            return

        if self.separate_model_in_populations:
            for id, model in enumerate(self.joint_policy):
                torch.save(model.state_dict(),
                           f"models/model_{id}_{int_id}.pt")
        else:
            if self.parameter_sharing:
                torch.save(self.joint_policy.state_dict(),
                       f"models/model_{int_id}.pt")
            else:
                for id, model in enumerate(self.joint_policy):
                    torch.save(model.state_dict(),
                               f"models/model_{id}_{int_id}.pt")


        torch.save(self.joint_action_value_function.state_dict(),
                   f"models/model_{int_id}-action-value.pt")

        torch.save(self.target_joint_action_value_function.state_dict(),
                   f"models/model_{int_id}-target-action-value.pt")

        torch.save(self.optimizer.state_dict(),
                   f"models/model_{int_id}-optim.pt")

    def load_model(self, int_id, overridden_model_dir=None):
        """
        """

        if self.mode == "train":
            model_dir = self.config['load_dir']

            if self.separate_model_in_populations:
                for id, model in enumerate(self.joint_policy):
                    model.load_state_dict(
                        torch.load(f"{model_dir}/models/model_{id}_{int_id}.pt")
                    )
            else:
                if self.parameter_sharing:
                    self.joint_policy.load_state_dict(
                        torch.load(f"{model_dir}/models/model_{int_id}.pt")
                    )
                else:
                    for id, model in enumerate(self.joint_policy):
                        model.load_state_dict(
                            torch.load(f"{model_dir}/models/model_{id}_{int_id}.pt")
                        )

            self.joint_action_value_function.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-action-value.pt")
            )

            self.target_joint_action_value_function.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-target-action-value.pt")
            )

            self.optimizer.load_state_dict(
                torch.load(f"{model_dir}/models/model_{int_id}-optim.pt")
            )

        else:
            model_dir = self.config.env['model_load_dir']
            if not overridden_model_dir is None:
                model_dir = overridden_model_dir

            if self.separate_model_in_populations:
                for id, model in enumerate(self.joint_policy):
                    model.load_state_dict(
                        torch.load(f"{model_dir}/model_{id}_{int_id}.pt", map_location=self.device)
                    )
            else:
                if self.parameter_sharing:
                    self.joint_policy.load_state_dict(
                        torch.load(f"{model_dir}/model_{int_id}.pt", map_location=self.device)
                    )
                else:
                    for id, model in enumerate(self.joint_policy):
                        model.load_state_dict(
                            torch.load(f"{model_dir}/models/model_{id}_{int_id}.pt")
                        )


class GST(object):
    """
        Gapped Straight-Through Estimator
        With help from: https://github.com/chijames/GST/blob/267ab3aa202d7a0cfd5b5861bd3dcad87faefd9f/model/basic.py
    """

    def __init__(self, logits, temperature=1.0, gap=1.0):
        self.logits = logits
        self.temperature = temperature
        self.gap = gap

    def replace_gradient(self, value, surrogate):
        """Returns `value` but backpropagates gradients through `surrogate`."""
        return surrogate + (value - surrogate).detach()

    @torch.no_grad()
    def _calculate_movements(self, logits, DD):
        max_logit = logits.max(dim=-1, keepdim=True)[0]
        selected_logit = torch.gather(logits, dim=-1, index=DD.argmax(dim=-1, keepdim=True))
        m1 = (max_logit - selected_logit) * DD
        m2 = (logits + self.gap - max_logit).clamp(min=0.0) * (1 - DD)
        return m1, m2

    def log_prob(self, sample):
        return dist.OneHotCategorical(logits=self.logits).log_prob(sample)

    def __call__(self, logits, need_gradients=True):
        DD = dist.OneHotCategorical(logits=logits).sample()
        if need_gradients:
            m1, m2 = self._calculate_movements(logits, DD)
            surrogate = F.softmax((logits + m1 - m2) / self.temperature, dim=-1)
            return self.replace_gradient(DD, surrogate)
        else:
            return DD

    def sample(self):
        return self.__call__(self.logits, need_gradients=False)

    def rsample(self):
        return self.__call__(self.logits, need_gradients=True)
