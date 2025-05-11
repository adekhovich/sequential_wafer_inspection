import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from utils.utils import img_to_patch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WaferMapGrid(gym.Env):
    def __init__(self, classifier, train_loader, test_loader, confidnet=None, start_policy=0):
        super(WaferMapGrid, self).__init__()
        
        self.classifier = classifier
        self.confidnet = confidnet
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.patches = None
        self.hidden = None
        self.patch_size = classifier.patch_size
        
        self.hidden_confid = None
        self.batch_dones = [0]
        
        self.curr_time = 0
        self.start_policy = start_policy
        
        self.action_space = spaces.Discrete(classifier.seq_len)
        self.action_mask = torch.ones(classifier.seq_len).to(device)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(classifier.hidden_dim, ), dtype=np.float64)
        
        
    def reward_func(self, inputs, mode='train'):
        if self.confidnet == None:
            #reward = - F.cross_entropy(inputs, self.targets, reduction='none')
            #reward = (2 * (inputs.argmax(dim=-1) == self.targets) - 1)
            reward = torch.tensor([F.softmax(inputs, dim=-1)[i, self.targets[i]] for i in range(inputs.size(0))]).to(device)
            
            if mode == 'train':
                confid = reward.clone()
                reward = torch.zeros_like(confid).to(device)
                reward[(confid >= 0.5)] = 1
        else:
            if mode == 'train':
                confid, hidden_confid  = self.confidnet.make_step(inputs, hidden=self.hidden_confid)
                confid = torch.squeeze(confid, dim=1)

                reward = torch.zeros_like(confid).to(device)
                reward[(confid >= 0.5)] = 1
            else:
                confid, hidden_confid  = self.confidnet.make_step(inputs, hidden=self.hidden_confid)
                reward = torch.squeeze(confid, dim=1)
                        
            return reward, hidden_confid
            
        return reward  
    
    def step(self, actions, compute_reward=True, mode='train'):
        inputs = torch.stack([self.patches[i, action, :] for i, action in enumerate(actions)], dim=0).to(device)   # batch x patch_num x h * w
        
        for i, action in enumerate(actions):
            self.action_mask[i, action] = 0
        
        y_pred, feat, hidden, input_patch = self.classifier.make_step(inputs, patch_num=actions, hidden=self.hidden)
        
        if compute_reward:
            if self.confidnet == None:
                reward = self.reward_func(y_pred, mode=mode)
            else:
                reward, hidden_confid = self.reward_func(feat, mode=mode)
                self.hidden_confid = hidden_confid
        else:
            reward = None
               
        self.hidden = hidden
        self.curr_time += 1
        done = self.curr_time == self.classifier.seq_len 
        
        #print(reward)
        #print(a)
        
        return y_pred, feat, reward, done, None
    
    
    def reset(self, patches=None, y=None, seed=0):        
        self.patches = patches
        self.targets = y
                
        batch_size = self.patches.size(0)
        seq_len = self.patches.size(1)
        
        self.hidden = [self.classifier.init_hidden(batch_size, self.classifier.hidden_dim)] * self.classifier.num_layers
        if self.confidnet != None:
            self.hidden_confid = [self.confidnet.init_hidden(batch_size, self.confidnet.hidden_dim)] * self.confidnet.num_layers
        
        self.batch_dones = torch.zeros(batch_size).to(device)
        
        self.action_mask = torch.ones((batch_size, seq_len)).to(device)
        
        self.curr_time = 0
        '''
        patches_explore = torch.tensor( np.random.choice(np.arange(seq_len), size=(batch_size, 1)) ).to(device)
        for action_num in range(patches_explore.size(1)):
            actions = patches_explore[:, action_num]
    
            inputs = torch.stack([patches[i, action, :] for i, action in enumerate(actions)], dim=0).to(device)
           
            for i, action in enumerate(actions):
                self.action_mask[i, action] = 0
            
            y_pred, hidden, encoded_input  = self.classifier.make_step(inputs, patch_num=actions, hidden=self.hidden)

            self.hidden = hidden
            self.curr_time += 1
        '''    
        
        return None, self.hidden[-1]
        
    
    def render(self):
        pass
