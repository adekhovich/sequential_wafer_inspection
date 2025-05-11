#import gym
import numpy as np
import torch

from utils.utils import img_to_patch
from patch_selector.utils.eval import eval_reinforce

from datetime import datetime


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_one_iter_reinforce(agent, env):
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    
    train_loader = env.train_loader
    score = 0
    total_loss = 0
    total_reward = 0
    for i, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        patches = img_to_patch(X, patch_size=env.patch_size)
        done = False
        _, observation = env.reset(patches, y)
        
        t_step = 0
        total_loss_batch = 0
        total_reward_batch = 0
        
        discount = 1.0
        while not done: 
            with torch.no_grad():
                agent.actor.eval()
                    
                action_mask = env.action_mask.clone()
                action, prob = agent.choose_action(observation, action_mask)
                y_pred, observation_, reward, done, info = env.step(action)

                if env.curr_time > env.start_policy:
                    total_reward += discount * reward.sum().float()
                    agent.remember(y_pred, observation, action_mask, action, prob, reward, done) 
                    discount *= agent.gamma

                observation = observation_
         
        total_loss_batch = agent.learn()
        agent.memory.clear_memory()            
        print(f"Batch {i+1}/{len(train_loader)} | Last loss: {total_loss_batch:.4f}")
        total_loss += total_loss_batch * X.size(0)
        
    total_loss /= len(train_loader.dataset)
    total_reward /= len(train_loader.dataset)
    
    return agent, total_loss, total_reward


def training_loop_reinforce(agent, env, num_iters=100, ensemble=False):
    best_score = -999999
    best_loss = 1e+10
    best_acc = 0
    
    train_loader = env.train_loader
    
    for epoch in range(0, num_iters):
        agent, train_loss, train_score = train_one_iter_reinforce(agent, env)
        
        test_acc, _, test_score, _, _ = eval_reinforce(agent, env, train=False)
        
        if train_score > best_score:
            best_score = train_score
            agent.save_models()
            

        print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  #f'Valid loss: {valid_loss:.4f}\t'
                  #f'Train accuracy: {train_acc:.2f}\t'
                  f'Train score: {train_score:.2f}\t'
                  f'Valid accuracy: {test_acc:.2f}\t'
                  f'Valid score: {test_score:.2f}'
             )
        
    return agent