import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from utils.utils import img_to_patch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class  WaferMapGridEnsemble(gym.Env):
    def __init__(self, classifier, train_loader, test_loader, confidnet=None, start_policy=0):
        super(WaferMapGridEnsemble, self).__init__()
        
        self.classifier = classifier
        self.confidnet = confidnet
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.patches = None
        self.hidden = []
        self.patch_size = classifier[0].patch_size
        
        self.hidden_confid = []
        self.batch_dones = [0]
        self.num_models = len(classifier)
        self.curr_time = 0
        self.start_policy = start_policy
        
        self.action_space = spaces.Discrete(classifier[0].seq_len)
        self.action_mask = torch.ones(classifier[0].seq_len).to(device)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                             shape=(classifier[0].hidden_dim, ), dtype=np.float64)
        
        
    def reward_func(self, inputs, mode='train', i=0):
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
                confid, hidden_confid  = self.confidnet[i].make_step(inputs, hidden=self.hidden_confid[i])
                confid = torch.squeeze(confid, dim=1)

                reward = torch.zeros_like(confid).to(device)
                reward[(confid >= 0.5)] = 1
            else:
                confid, hidden_confid  = self.confidnet[i].make_step(inputs, hidden=self.hidden_confid[i])
                reward = torch.squeeze(confid, dim=1)
                        
            return reward, hidden_confid
            
        return reward  
    
    def step(self, actions, compute_reward=True, mode='train'):
        batch_size = self.patches.size(0)
        inputs = torch.stack([self.patches[i, action, :] for i, action in enumerate(actions)], dim=0).to(device)   # batch x patch_num x h * w
        
        for i, action in enumerate(actions):
            self.action_mask[i, action] = 0
            
        probs_avg = 0
        reward_ens = torch.zeros((self.num_models, batch_size)).to(device)
        reward_avg = 0
        feats  = []
        
        for i in range(self.num_models):
            y_pred, feat, hidden, input_patch = self.classifier[i].make_step(inputs, patch_num=actions, hidden=self.hidden[i])

            if compute_reward:
                if self.confidnet == None:
                    reward = self.reward_func(y_pred, mode=mode)
                else:
                    reward, hidden_confid = self.reward_func(feat, mode=mode, i=i)
                    self.hidden_confid[i] = hidden_confid
            else:
                reward = None

            self.hidden[i] = hidden
            reward_avg += reward
            reward_ens[i] = reward
            probs_avg += F.softmax(y_pred, dim=-1)
            feats.append(feat)
            
        reward_avg /= self.num_models
        probs_avg /= self.num_models
        
        reward_avg = torch.mean(reward_ens, dim=0)
        #reward_avg = torch.median(reward_ens, dim=0)[0]
        self.curr_time += 1
        done = self.curr_time == self.classifier[0].seq_len 
      
        
        return probs_avg, feats, reward_avg, done, None
    
    
    def reset(self, patches=None, y=None, seed=0):        
        self.patches = patches
        self.targets = y
                
        batch_size = self.patches.size(0)
        seq_len = self.patches.size(1)
        
        self.hidden = [ [self.classifier[i].init_hidden(batch_size, self.classifier[i].hidden_dim)] * self.classifier[i].num_layers for i in range(self.num_models)]
        if self.confidnet != None:
            self.hidden_confid = [ [self.confidnet[i].init_hidden(batch_size, self.confidnet[i].hidden_dim)] * self.confidnet[i].num_layers for i in range(self.num_models)]
        
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
        
        return None, [self.hidden[i][-1] for i in range(self.num_models)]
        
    
    def render(self):
        pass
