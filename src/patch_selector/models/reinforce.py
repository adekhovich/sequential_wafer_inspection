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

class REINFORCEMemory:
    def __init__(self):
        self.y_preds = []
        self.states = []
        self.action_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size=1):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        return self.y_preds,\
               self.states,\
               self.action_masks,\
               self.actions,\
               self.probs,\
               self.rewards,\
               self.dones,\
               batches
        

    def store_memory(self, y_pred, state, action_mask, action, probs, reward, done):
        self.y_preds.append(y_pred.detach().cpu())
        self.states.append(state.cpu())
        self.action_masks.append(action_mask.cpu())
        self.actions.append(action.cpu())
        self.probs.append(probs.cpu())
        self.rewards.append(reward.cpu())
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.action_masks = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, num_actions, input_dims, lr,
            fc1_dims=400, fc2_dims=400, chkpt_dir='./', params_str=''):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'patch_selector_reinforce' + params_str)
        
        self.actor = nn.Sequential(
                nn.Linear(*input_dims + num_actions, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, num_actions),
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_mask):
        state = torch.cat((state, action_mask), dim=1)
        
        logits = self.actor(state)
        dists = MaskedCategorical(logits=logits, mask=action_mask.to(torch.bool))
        
        return dists

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class REINFORCEAgent:
    def __init__(self, num_actions, input_dims, gamma=0.99, lr=0.0003, batch_size=64, num_iters=10, seed=0, chkpt_dir='./'):
        self.gamma = gamma
        self.num_iters = num_iters
        self.lr = lr
        self.num_actions = num_actions
        self.batch_size = batch_size
        
        params_str = f"_seed{seed}"
        self.actor = ActorNetwork(num_actions, input_dims, lr, chkpt_dir=chkpt_dir, params_str=params_str)
        self.memory = REINFORCEMemory()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
    def remember(self, y_pred, state, action_mask, action, probs, reward, done):
        self.memory.store_memory(y_pred, state, action_mask, action, probs, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()

    def choose_action(self, observation, action_mask, determenistic=False):
        state = observation
        dists = self.actor(state, action_mask)
        if determenistic:
            actions = dists.mode
        else:    
            actions = dists.sample()
        
        probs = dists.log_prob(actions)
                
        return actions, probs
    
    
    def learn(self):
        self.actor.train()
        
        y_pred_arr, state_arr, action_mask_arr, action_arr, prob_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
     
        returns  = torch.zeros((len(reward_arr), reward_arr[0].size(0)), dtype=torch.float).to(self.actor.device)
        reward_arr = torch.stack(reward_arr, dim=0).to(self.actor.device)
        
        for t in range(len(reward_arr)):
            discount = 1
            a_t = 0
            r_t = 0 
            for k in range(t, len(reward_arr)):
                r_t += discount * reward_arr[k].float()
                discount *= self.gamma
                
            returns[t] = r_t    
            
        num_timesteps = min(len(state_arr), self.num_actions-1)
        bs = self.batch_size
        
        for _ in range(1):
            total_loss = 0
            batch_inds = np.arange(num_timesteps)
            np.random.shuffle(batch_inds)

            for ind, batch in enumerate(batch_inds):
                if ind % bs == 0:
                    self.actor.optimizer.zero_grad()

                y_preds = y_pred_arr[batch].to(self.actor.device)            
                states = state_arr[batch].to(self.actor.device)
                action_masks = action_mask_arr[batch].to(self.actor.device)
                actions = action_arr[batch].to(self.actor.device)

                dists = self.actor(states, action_masks)
                log_probs = dists.log_prob(actions)
                loss = (- returns[batch] * log_probs).mean()

                loss.backward()
                total_loss += loss.item()

                if ((ind+1) % bs == 0) or (ind == num_timesteps-1):
                #if (ind+1 % bs == 0) or (ind == num_timesteps-1):
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                    self.actor.optimizer.step()
                    
        return total_loss
            

