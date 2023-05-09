from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F

# File containing implementation of networks used in this research

def fc_network(layer_dims, init_ortho=True):
    """
    Builds a fully-connected NN with ReLU activation functions.

    """
    if init_ortho: 
        init = init_orthogonal
    else:
        init = lambda m: m

    network = nn.Sequential(
                *chain(
                    *((init(nn.Linear(layer_dims[i], layer_dims[i+1])),
                       nn.ReLU())
                      for i in range(len(layer_dims)-1))
                    ),
                )
    del network[-1]  # remove the final ReLU layer
    return network

def init_orthogonal(m):
    nn.init.orthogonal_(m.weight)
    if hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)
    return m

class AgentNetworkFC(nn.Module):
    def __init__(self, obs_dim, obs_u_dim, mid_dim1, mid_dim2, gnn_hdim1, gnn_hdim2, gnn_out_dim, num_acts, device):
        super(AgentNetworkFC, self).__init__()
        self.obs_dim = obs_dim
        self.obs_u_dim = obs_u_dim
        self.mid_dim1 = mid_dim1
        self.mid_dim2 = mid_dim2
        self.gnn_hdim1 = gnn_hdim1
        self.gnn_hdim2 = gnn_hdim2
        self.gnn_out_dim = gnn_out_dim
        self.num_acts = num_acts
        self.device = device

        self.linear_obs_rep = nn.Sequential(
            nn.Linear(obs_dim + obs_u_dim, mid_dim1),
            nn.ReLU(),
            nn.Linear(mid_dim1, mid_dim2),
            nn.ReLU(),
            nn.Linear(mid_dim2, mid_dim2)
        ).to(self.device)

        self.gnn = nn.Sequential(
            nn.Linear(mid_dim2, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2),
            nn.ReLU(),
            nn.Linear(gnn_hdim2, gnn_out_dim)
        )

        self.act_logits = nn.Linear(gnn_out_dim, num_acts).to(self.device)
        self.v_compute = nn.Linear(gnn_out_dim, 1).to(self.device)


    def forward(self, input):
        n_inp = self.linear_obs_rep(input.to(self.device))
        n_out = self.gnn(n_inp)
        act_logits = self.act_logits(n_out)

        return act_logits


class EncoderLSTM(nn.Module):
    def __init__(self, obs_dim, obs_u_dim, mid_dim1, rep_size, device):
        super(EncoderLSTM, self).__init__()
        self.obs_dim = obs_dim
        self.obs_u_dim = obs_u_dim
        self.mid_dim1 = mid_dim1
        self.rep_size = rep_size
        self.device = device

        self.linear_obs_rep = nn.Sequential(
            nn.Linear(obs_dim + obs_u_dim, mid_dim1),
            nn.ReLU(),
            nn.Linear(mid_dim1, mid_dim1),
            nn.ReLU(),
            nn.Linear(mid_dim1, mid_dim1)
        ).to(self.device)

        self.rep_gen = nn.LSTM(mid_dim1, rep_size, batch_first=False)

    def forward(self, input, input_c1, input_c2):
        n_out = self.linear_obs_rep(input.to(self.device))
        n_out, input_c1, input_c2 = n_out.unsqueeze(0), input_c1.unsqueeze(0), input_c2.unsqueeze(0)
        ag_reps, new_c = self.rep_gen(n_out, (input_c1, input_c2))

        return ag_reps.squeeze(0), new_c[0].squeeze(0), new_c[1].squeeze(0)


class Decoder(nn.Module):
    def __init__(self, input_size, mid_dim2, gnn_hdim1, gnn_hdim2, gnn_out_dim, num_acts, obs_size, device):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.mid_dim2 = mid_dim2
        self.gnn_hdim1 = gnn_hdim1
        self.gnn_hdim2 = gnn_hdim2
        self.obs_size = obs_size
        self.device = device

        self.gnn_act = nn.Sequential(
            nn.Linear(input_size, mid_dim2),
            nn.ReLU(),
            nn.Linear(mid_dim2, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2)
        ).to(self.device)

        self.gnn_obs = nn.Sequential(
            nn.Linear(input_size, mid_dim2),
            nn.ReLU(),
            nn.Linear(mid_dim2, gnn_hdim1),
            nn.ReLU(),
            nn.Linear(gnn_hdim1, gnn_hdim2)
        ).to(self.device)

        self.act_logits = nn.Sequential(
            nn.Linear(gnn_hdim2, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, num_acts)
        ).to(self.device)

        self.obs_recon = nn.Sequential(
            nn.Linear(gnn_hdim2, gnn_out_dim),
            nn.ReLU(),
            nn.Linear(gnn_out_dim, obs_size)
        ).to(self.device)

        self.device = device

    def forward(self, input):
        n_out_act, n_out_obs = self.gnn_act(input.to(self.device)), self.gnn_obs(input.to(self.device))
        act_logits, obs_recon = self.act_logits(n_out_act), self.obs_recon(n_out_obs)
        return act_logits, obs_recon


class AgentNetworkSeparate(nn.Module):
    def __init__(self, obs_dim, obs_u_dim, mid_dim1, mid_dim2, gnn_hdim1, gnn_hdim2, gnn_out_dim, num_acts, device, id_length):
        super(AgentNetworkSeparate, self).__init__()
        self.obs_dim = obs_dim
        self.obs_u_dim = obs_u_dim
        self.mid_dim1 = mid_dim1
        self.mid_dim2 = mid_dim2
        self.gnn_hdim1 = gnn_hdim1
        self.gnn_hdim2 = gnn_hdim2
        self.gnn_out_dim = gnn_out_dim
        self.num_acts = num_acts
        self.device = device

        self.id_length = id_length

        self.linear_layer1 = nn.ModuleList([nn.Linear(obs_dim + obs_u_dim - self.id_length, mid_dim1) for _ in range(self.id_length)])
        self.linear_layer2 = nn.ModuleList([nn.Linear(mid_dim1, mid_dim2) for _ in range(self.id_length)])
        self.linear_layer3 = nn.ModuleList([nn.Linear(mid_dim2, mid_dim2) for _ in range(self.id_length)])
        self.linear_layer4 = nn.ModuleList([nn.Linear(mid_dim2, gnn_hdim1) for _ in range(self.id_length)])
        self.linear_layer5 = nn.ModuleList([nn.Linear(gnn_hdim1, gnn_hdim2) for _ in range(self.id_length)])
        self.linear_layer6 = nn.ModuleList([nn.Linear(gnn_hdim2, gnn_out_dim) for _ in range(self.id_length)])
        self.act_logit_layer = nn.ModuleList([nn.Linear(gnn_out_dim, num_acts) for _ in range(self.id_length)])

    def forward(self, input):
        agent_ids = torch.argmax(input[:, :, -self.id_length:], dim=-1)
        batch_size, num_agents = input.size()[0], input.size()[1]

        # This until line 130 corresponds to linear_obs_rep
        # Process input to layer 1
        state_only_input = torch.unsqueeze(input[:, :, :-self.id_length], -2).view(batch_size * num_agents, 1, -1)
        w1, b1 = self.gather_weights_and_biases(self.linear_layer1, agent_ids)
        state_only_input = torch.unsqueeze(input[:, :, :-self.id_length], -2).view(batch_size * num_agents, 1, -1)

        out = torch.bmm(state_only_input, w1.permute(0,2,1)) + b1.view(batch_size*num_agents,1,-1)
        out = F.relu(out)

        w2, b2 = self.gather_weights_and_biases(self.linear_layer2, agent_ids)
        out = torch.bmm(out, w2.permute(0, 2, 1)) + b2.view(batch_size * num_agents, 1, -1)
        out = F.relu(out)

        w3, b3 = self.gather_weights_and_biases(self.linear_layer3, agent_ids)
        out = torch.bmm(out, w3.permute(0, 2, 1)) + b3.view(batch_size * num_agents, 1, -1)

        w4, b4 = self.gather_weights_and_biases(self.linear_layer4, agent_ids)
        out = torch.bmm(out, w4.permute(0, 2, 1)) + b4.view(batch_size * num_agents, 1, -1)
        out = F.relu(out)

        w5, b5 = self.gather_weights_and_biases(self.linear_layer5, agent_ids)
        out = torch.bmm(out, w5.permute(0, 2, 1)) + b5.view(batch_size * num_agents, 1, -1)
        out = F.relu(out)

        w6, b6 = self.gather_weights_and_biases(self.linear_layer6, agent_ids)
        out = torch.bmm(out, w6.permute(0, 2, 1)) + b6.view(batch_size * num_agents, 1, -1)

        w7, b7 = self.gather_weights_and_biases(self.act_logit_layer, agent_ids)
        out = torch.bmm(out, w7.permute(0, 2, 1)) + b7.view(batch_size * num_agents, 1, -1)

        act_logits = torch.squeeze(out, 1).view(batch_size, num_agents, -1)
        return act_logits


    def gather_weights_and_biases(self, param_list, agent_ids):
        weight_list = torch.stack([a.weight for a in param_list])
        bias_list = torch.stack([a.bias for a in param_list])

        agent_idsw, agent_idsb = agent_ids.view(-1,1,1), agent_ids.view(-1,1)
        gathered_weights = torch.gather(
            weight_list, 0, agent_idsw.repeat(1, weight_list.size()[1], weight_list.size()[2])
        )
        gathered_biasess = torch.gather(
            bias_list, 0, agent_idsb.repeat(1, weight_list.size()[1])
        )
        return gathered_weights, gathered_biasess


class AgentNetworkWHypernet(nn.Module):
    def __init__(self, obs_dim, obs_u_dim, mid_dim1, mid_dim2, gnn_hdim1, gnn_hdim2, gnn_out_dim, num_acts, device, id_length, hypernet_embed):
        super(AgentNetworkWHypernet, self).__init__()
        self.obs_dim = obs_dim
        self.obs_u_dim = obs_u_dim
        self.mid_dim1 = mid_dim1
        self.mid_dim2 = mid_dim2
        self.gnn_hdim1 = gnn_hdim1
        self.gnn_hdim2 = gnn_hdim2
        self.gnn_out_dim = gnn_out_dim
        self.num_acts = num_acts
        self.device = device

        self.id_length = id_length
        self.hypernet_embed = hypernet_embed

        self.layer1_hypernet = nn.Sequential(
            nn.Linear(self.id_length, 10*hypernet_embed),
            nn.ReLU()
        )

        self.hyper_net1_w = nn.Linear(hypernet_embed, (obs_dim + obs_u_dim - self.id_length) * mid_dim1)
        self.hyper_net1_b = nn.Linear(hypernet_embed, mid_dim1)
        self.hyper_net2_w = nn.Linear(hypernet_embed, mid_dim1 * mid_dim2)
        self.hyper_net2_b = nn.Linear(hypernet_embed, mid_dim2)
        self.hyper_net3_w = nn.Linear(hypernet_embed, mid_dim2 * mid_dim2)
        self.hyper_net3_b = nn.Linear(hypernet_embed, mid_dim2)
        self.hyper_net4_w = nn.Linear(hypernet_embed, mid_dim2 * gnn_hdim1)
        self.hyper_net4_b = nn.Linear(hypernet_embed, gnn_hdim1)
        self.hyper_net5_w = nn.Linear(hypernet_embed, gnn_hdim1 * num_acts)
        self.hyper_net5_b = nn.Linear(hypernet_embed, num_acts)

    def forward(self, input):

        hypernet_l1 = self.layer1_hypernet(input[:,:,-self.id_length:])
        batch_size, num_agents = input.size()[0], input.size()[1]

        # This until line 130 corresponds to linear_obs_rep
        # Process input to layer 1
        state_only_input = torch.unsqueeze(input[:,:,:-self.id_length], -2).view(batch_size*num_agents,1,-1)
        hypernet1_outw = self.hyper_net1_w(hypernet_l1[:,:,:self.hypernet_embed]).view(batch_size*num_agents, -1, self.mid_dim1)
        hypernet1_outb = self.hyper_net1_b(hypernet_l1[:,:,self.hypernet_embed:2*self.hypernet_embed]).view(batch_size*num_agents, 1, -1)

        out = torch.bmm(state_only_input, hypernet1_outw) + hypernet1_outb
        out = F.relu(out)

        hypernet2_outw = self.hyper_net2_w(hypernet_l1[:, :,2*self.hypernet_embed:3*self.hypernet_embed]).view(batch_size * num_agents, -1, self.mid_dim2)
        hypernet2_outb = self.hyper_net2_b(hypernet_l1[:, :,3*self.hypernet_embed:4*self.hypernet_embed]).view(batch_size * num_agents, 1, -1)
        out = F.relu(torch.bmm(out, hypernet2_outw) + hypernet2_outb)

        hypernet3_outw = self.hyper_net3_w(hypernet_l1[:, :, 4 * self.hypernet_embed:5 * self.hypernet_embed]).view(
            batch_size * num_agents, -1, self.mid_dim2)
        hypernet3_outb = self.hyper_net3_b(hypernet_l1[:, :, 5 * self.hypernet_embed:6 * self.hypernet_embed]).view(
            batch_size * num_agents, 1, -1)
        out = F.relu(torch.bmm(out, hypernet3_outw) + hypernet3_outb)

        # This until corresponds to L148 corresponds to self.gnn
        hypernet4_outw = self.hyper_net4_w(hypernet_l1[:, :, 6 * self.hypernet_embed:7 * self.hypernet_embed]).view(
            batch_size * num_agents, -1, self.gnn_hdim1)
        hypernet4_outb = self.hyper_net4_b(hypernet_l1[:, :, 7 * self.hypernet_embed:8 * self.hypernet_embed]).view(
            batch_size * num_agents, 1, -1)
        out = torch.bmm(out, hypernet4_outw) + hypernet4_outb

        hypernet5_outw = self.hyper_net5_w(hypernet_l1[:, :, 8 * self.hypernet_embed:9 * self.hypernet_embed]).view(
            batch_size * num_agents, -1, self.num_acts)
        hypernet5_outb = self.hyper_net5_b(hypernet_l1[:, :, 9 * self.hypernet_embed:10 * self.hypernet_embed]).view(
            batch_size * num_agents, 1, -1)
        out = F.relu(torch.bmm(out, hypernet5_outw) + hypernet5_outb)

        act_logits = torch.squeeze(out, 1).view(batch_size, num_agents, -1)

        return act_logits
