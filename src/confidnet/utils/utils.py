import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


import os
import random
from datetime import datetime

from sklearn.metrics import f1_score



def validate_rnnconfidnet(confidnet, rnn, valid_loader, criterion, device, patch_order=None, multioutput=False):
    '''
    Function for the validation step of the training loop
    '''
   
    confidnet.eval()
    rnn.eval()
    running_loss = 0
       
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)
        
        # Forward pass and record loss
        if patch_order != None:
            y_preds, hidden = rnn(X, patch_order=patch_order, multioutput=multioutput)
        else:    
            y_preds, hidden = rnn(X, multioutput=multioutput)
        
        if multioutput: 
            hidden = torch.stack(hidden, dim=0)
            y_preds = torch.stack(y_preds, dim=0)

            confidence, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
            confidence = torch.stack(confidence, dim=0)
            
            loss_conf = criterion(confidence, y_preds, y_true)   
            loss = loss_conf.mean()
        else:
            confidence = confidnet(hidden[-1])
            loss = criterion(confidence, y_preds, y_true)
            
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return confidnet, epoch_loss


def accuracy_rnnconfidnet(rnn, confidnet, data_loader, device, cnn_model=None, patch_order=None, prediction_index=-1, offset1=0, offset2=256, supcon=False):
    
    if cnn_model != None:
        cnn_model.eval()
        
    correct_preds = 0 
    n = 0
    idx = []   
    best_heads = []
    num_scans = []
    
    all_preds = []
    all_labels = []
    all_confid = []
        
    with torch.no_grad():
        rnn.eval()
        confidnet.eval()
        for i, (X, y_true) in enumerate(data_loader):

            X = X.to(device)
            y_true = y_true.to(device)
           
            if prediction_index == 'confidnet':
                y_preds, hidden = rnn(X, patch_order=patch_order, multioutput=True)
                y_preds = torch.stack(y_preds, dim=1)
                
                hidden = torch.stack(hidden, dim=0)
                confidence, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
                    
                confidence = torch.stack(confidence, dim=0)
                confidence = confidence.transpose(0, 1).squeeze(-1)
                
                head_mask = (confidence[:, offset1:offset2] >= 0.5).sum(dim=-1)
                best_head = (1*(confidence[:, offset1:offset2] >= 0.5)).argmax(dim=-1) + offset1
                best_head[head_mask == 0] = confidence[head_mask == 0, offset1:offset2].argmax(dim=-1) + offset1
                
                idx.append(best_head.clone())
                best_head[head_mask == 0] = offset2 - 1
                num_scans.append(best_head.clone() + 1)                
                
                y_preds = torch.stack([y_preds[i, idx[-1][i], :] for i in range(len(idx[-1]))], dim=0)
                
            else:
                y_preds, _ = rnn(X, patch_order=patch_order, prediction_index=prediction_index, repeat=repeat)
           
            if cnn_model != None:
                if supcon:
                    y_preds[head_mask == 0], _ = cnn_model(X[head_mask == 0])
                else:
                    y_preds[head_mask == 0] = cnn_model(X[head_mask == 0])
                
            n += y_true.size(0)
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()
            
            all_labels.append(y_true)
            all_preds.append(y_preds)
            
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0).argmax(dim=-1)
    
    f1 = f1_score(all_labels.detach().cpu().numpy(), all_preds.detach().cpu().numpy(), average='macro')        
            
    if prediction_index == 'confidnet':
        idx = torch.cat(idx, dim=0)
        num_scans = torch.cat(num_scans, dim=0)
        return (correct_preds/n).item(), f1.item(),  idx, num_scans
    
    torch.cuda.empty_cache()
    return (correct_preds/n).item(), f1.item()
