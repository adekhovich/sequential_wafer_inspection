import os
import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F

from utils.utils import img_to_patch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def eval_reinforce(agent, env, train=False, confidnet=None, cnn_model=None, offset=0, ensemble=False, supcon=False):
    if ensemble:
        for i in range(len(agent.actors)):
            agent.actors[i].eval()
    else:
        agent.actor.eval()
    
    if cnn_model != None:
        cnn_model.eval()
    
    if train:
        data_loader = env.train_loader
    else:
        data_loader = env.test_loader
    
    correct_preds = 0
    total_reward = 0
    gamma = agent.gamma
    
    if ensemble:
        num_classes = env.classifier[0].num_classes
    else:
        num_classes = env.classifier.num_classes
    
    patch_order = []
    stopping = []
    max_confids = []
    
    correct_full = 0
    correct_notfull = 0
    
    preds = []
    labels = []
    
    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        
        patches = img_to_patch(X, patch_size=env.patch_size)
        
        batch_size = patches.size(0)
        seq_len = patches.size(1)
        
        _, observation = env.reset(patches, y)
        discount = 1.0
        patch_order_batch = []
        
        if confidnet != None:
            if ensemble:
                hidden_dim = confidnet[0].hidden_dim
            else:
                hidden_dim = confidnet.hidden_dim
                
            filled = torch.zeros(batch_size).to(device)
            y_pred = torch.zeros((batch_size, num_classes)).to(device)
            stopping_batch = torch.zeros(batch_size).to(device)
            
            max_confid = torch.zeros(batch_size).to(device)
        
        for t in range(0, seq_len):
            actions_mask = env.action_mask.clone()
            action, prob = agent.choose_action(observation, actions_mask, determenistic=True)

            if confidnet != None and t >= offset:
                outputs, observation_, reward, done, info = env.step(action, mode='inference')
                confidence = reward.to(device)
                
                y_pred[(confidence >= 0.5) * (filled == 0)] = outputs[(confidence >= 0.5) * (filled == 0)]
                stopping_batch[(confidence >= 0.5) * (filled == 0)] = t + 1
                
                filled[(confidence >= 0.5)] = 1
                max_confid = torch.maximum(confidence, max_confid)
            else:
                y_pred, observation_, reward, done, info = env.step(action)
        
            
            if env.curr_time >= env.start_policy:
                total_reward += (discount * reward).sum()
                discount *= gamma
            
            observation = observation_
            patch_order_batch.append(action)
            
        if confidnet != None:  
            max_confids.append(max_confid)
            
            if cnn_model == None:
                y_pred[filled == 0] = outputs[filled == 0] 
            else:
                if (filled == 0).sum() > 0:
                    if not supcon:
                        y_pred[filled == 0] = cnn_model(X[filled == 0])
                    else:
                        y_pred[filled == 0], _ = cnn_model(X[filled == 0])
            
            correct_full += (y_pred[filled == 0].argmax(dim=-1) == y[filled == 0]).sum()
            correct_notfull += (y_pred[filled == 1].argmax(dim=-1) == y[filled == 1]).sum()
                
            stopping_batch[filled == 0] = seq_len
            #print(stopping_batch)
            stopping.append(stopping_batch)
            
        patch_order_batch = torch.stack(patch_order_batch, dim=1).detach().cpu()
        patch_order.append(patch_order_batch)
        correct_preds += (y_pred.argmax(dim=-1) == y).sum()
        preds.append(y_pred.argmax(dim=-1))
        labels.append(y)
        print(f"Evaluation: batch {batch_idx}/{len(data_loader)}")        
    
    patch_order = torch.cat(patch_order, dim=0)
    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    labels = torch.cat(labels, dim=0).detach().cpu().numpy()
    
    acc = 100 * correct_preds / len(data_loader.dataset)
    f1 = f1_score(labels, preds, average='macro')
    score = total_reward / len(data_loader.dataset)
    
    if confidnet != None:
        #max_confids = torch.cat(max_confids, dim=0)
       # print("Very unsure: ", max_confids.min(), (max_confids < 1/num_classes).sum().item())
        
        stopping = torch.cat(stopping, dim=0)
        #print(f"Full scan correct: {(100 * correct_full / (stopping == seq_len).sum()):.2f}, NOT Full scan correct: {(100 * correct_notfull / (stopping < seq_len).sum()):.2f}")
        return acc, f1, score, patch_order, stopping
    
    return acc, f1, score, patch_order, None


def predict_reinforce(agent, env, train=False, confidnet=None, cnn_model=None, offset=0, ensemble=False):
    if ensemble:
        for i in range(len(agent.actors)):
            agent.actors[i].eval()
    else:
        agent.actor.eval()
    
    if cnn_model != None:
        cnn_model.eval()
    
    if train:
        data_loader = env.train_loader
    else:
        data_loader = env.test_loader
    
    correct_preds = 0
    total_reward = 0
    gamma = agent.gamma
    
    if ensemble:
        num_classes = env.classifier[0].num_classes
    else:
        num_classes = env.classifier.num_classes
   
    y_preds = []
    
    correct_full = 0
    correct_notfull = 0
    
    for batch_idx, (X, y) in enumerate(data_loader):
        X = X.to(device)
        y = y.to(device)
        
        patches = img_to_patch(X, patch_size=env.patch_size)
        
        batch_size = patches.size(0)
        seq_len = patches.size(1)
        
        _, observation = env.reset(patches, y)
        discount = 1.0
        patch_order_batch = []
        agent.memory.clear_memory()
        
        if confidnet != None:
            if ensemble:
                hidden_dim = confidnet[0].hidden_dim
            else:
                hidden_dim = confidnet.hidden_dim
                
            filled = torch.zeros(batch_size).to(device)
            y_pred = torch.zeros((batch_size, num_classes)).to(device)
            
            max_confid = torch.zeros(batch_size).to(device)
        
        for t in range(0, seq_len):
            actions_mask = env.action_mask.clone()
            action, prob = agent.choose_action(observation, actions_mask, determenistic=True)

            if confidnet != None and t >= offset:
                outputs, observation_, reward, done, info = env.step(action, mode='inference')
                confidence = reward.to(device)
                
                y_pred[(confidence >= 0.5) * (filled == 0)] = outputs[(confidence >= 0.5) * (filled == 0)]
                
                filled[(confidence >= 0.5)] = 1
            else:
                y_pred, observation_, reward, done, info = env.step(action)
        
            if env.curr_time >= env.start_policy:
                total_reward += (discount * reward).sum()
                discount *= gamma 
            
            observation = observation_
            
        if confidnet != None:
            
            if cnn_model == None:
                y_pred[filled == 0] = outputs[filled == 0] 
            else:
                if (filled == 0).sum() > 0:
                    y_pred[filled == 0] = cnn_model(X[filled == 0])
            
            correct_full += (y_pred[filled == 0].argmax(dim=-1) == y[filled == 0]).sum()
            correct_notfull += (y_pred[filled == 1].argmax(dim=-1) == y[filled == 1]).sum()
            
        y_preds.append(y_pred)
        
    y_preds = torch.cat(y_preds, dim=0)
    
    
    return y_preds


