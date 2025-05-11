import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.utils import img_to_patch
from patch_selector.utils.utils import MaskedCategorical



class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims,
            fc1_dims=400, fc2_dims=400, chkpt_dir='./', params_str='', seed=0):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'patch_selector_reinforce' + params_str + f"_seed{seed}")
        
        self.actor = nn.Sequential(
                nn.Linear(*input_dims + n_actions, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_mask):
        state = torch.cat((state, action_mask), dim=1)
        
        logits = self.actor(state)
        
        return logits

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class REINFORCEAgentEnsemble:
    def __init__(self, n_actions, input_dims, gamma=0.99, ensemble_type="min-entropy", num_models=5, seeds=[0, 1, 2, 3, 4], chkpt_dir='./'):
        self.n_actions = n_actions
        
        params_str = ""
        self.num_models = num_models
        self.actors = [ActorNetwork(n_actions, input_dims, chkpt_dir=chkpt_dir, params_str=params_str, seed=seed) for seed in seeds]
        self.ensemble_type = ensemble_type
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    def save_models(self):
        self.actor.save_checkpoint()

    def load_models(self):
        for i in range(self.num_models):
            self.actors[i].load_checkpoint()

    def choose_action(self, observation, action_mask, determenistic=False):
        logits_avg = 0 
        probs_avg = 0
        entropy = torch.zeros((observation[0].size(0), self.num_models)).to(self.device)
        probs_arr = torch.zeros((observation[0].size(0), self.num_models, self.n_actions)).to(self.device)
        
        for i in range(self.num_models):
            logits = self.actors[i](observation[i], action_mask)
            logits_avg += logits
            p = F.softmax(logits, dim=-1)
            probs_avg += p
            
            if self.ensemble_type == 'min-entropy':
                p = (p * action_mask) / (p * action_mask).sum(dim=-1, keepdim=True)
                probs_arr[:, i, :] = p
                entropy[:, i] =  - (p * torch.log(p)).sum(dim=1) 

            
        logits_avg /= self.num_models
        probs_avg /= self.num_models
        
        if self.ensemble_type == 'min-entropy':
            pi_idx = entropy.min(dim=-1)[1]
            probs_avg = torch.stack([probs_arr[i, pi_i, :] for i, pi_i in enumerate(pi_idx)], dim=0)
        
        #print(probs_avg)
        dists = MaskedCategorical(probs=probs_avg, mask=action_mask.to(torch.bool))
        
        if determenistic:
            actions = dists.mode
        else:    
            actions = dists.sample()

        probs = dists.log_prob(actions)
                
        return actions, probs
    