import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score


import os
import random
import math
import seaborn as sns
from datetime import datetime


TYPE_TO_CLASS = {
    'Loc' : 0, 
    'Edge-Loc' : 1, 
    'Center' : 2, 
    'Edge-Ring' : 3, 
    'Scratch' : 4,
    'Random' : 5, 
    'Near-full' : 6, 
    'Donut' : 7,
    'none' : 8,
}



def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    
    
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    
    
def img_to_patch(x, patch_size=4, denoising_filter=None, flatten_channels=True, visualize=False):
    """
    Source: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
    
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)              # [B, H'*W', C, p_H, p_W]
    
    if flatten_channels:
        x = x.flatten(2, 4)          # [B, H'*W', C*p_H*p_W]
        
    if visualize:
        #img_patches = img_to_patch(x, patch_size=4, flatten_channels=False)

        fig, ax = plt.subplots(x.shape[0], 1, figsize=(14,3))
        fig.suptitle("Images as input sequences of patches")
        for i in range(x.shape[0]):
            img_grid = torchvision.utils.make_grid(x[i], nrow=64, normalize=True, pad_value=0.9)
            img_grid = img_grid.permute(1, 2, 0)
            ax[i].imshow(img_grid)
            ax[i].axis('off')
        plt.show()
        plt.close()
        
    return x  

def patch_to_img(x, patch_size=4):
    B, S, L = x.shape
    
    x = x.unflatten(dim=-1, sizes=(1, patch_size, patch_size) )
    size = int(math.sqrt(S))
    x = x.unflatten(dim=1, sizes=(size, size))
    x = x.permute(0, 3, 1, 4, 2, 5)
    B, C, H_, _, W_, _  = x.shape       
    x = x.reshape(B, C, H_ * patch_size, W_ * patch_size)
    
    return x


def compute_entropy(logits):
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, -1)
    
    return entropy


def validate(model, valid_loader, criterion, device, patch_order=None, multioutput=False):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    if multioutput:
        criterion = nn.CrossEntropyLoss(reduction='none')
       
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)
        
        # Forward pass and record loss
        if patch_order != None:
            y_preds, _ = model(X, patch_order=patch_order, multioutput=multioutput)
        else:    
            y_preds, _ = model(X, multioutput=multioutput)
        
        if multioutput: 
            loss  = []
            for i in range(len(y_preds)):
                loss_head = criterion(y_preds[i], y_true)
                loss.append(loss_head)
            
            loss = torch.stack(loss, dim=0)
            loss = loss.min(dim=0)[0].mean()
        else:
            loss = criterion(y_preds, y_true) 
            
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def validate_cnn(model, valid_loader, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
       
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)
        
        # Forward pass and record loss
        y_preds = model(X)
        loss = criterion(y_preds, y_true) 
            
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def rnn_ensemble(model, x, patch_order, seeds=[0, 1, 2, 3, 4], multioutput=False, prediction_index=-1):
    
    logits = 0
    hidden = 0
    for seed in seeds:
        PATH = f"models/gru/gru_ln_8heads_{model.num_layers}hidden{model.hidden_dim}_seed{seed}.pth"
        model.load_state_dict(torch.load(PATH))
        
        logits_seed, hidden_seed = model(x, patch_order=patch_order, multioutput=multioutput, prediction_index=prediction_index)
        if multioutput:
            logits_seed = torch.cat(logits_seed, dim=0)
        
        logits += logits_seed
        
    logits /= len(seeds)  
        
    return logits


def accuracy(model, data_loader, device, patch_order=None, prediction_index=-1, offset1=0, offset2=256, ensemble=False):
    correct_preds = 0 
    n = 0
    idx = []   
        
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)
             
            if prediction_index in ['best-e', 'best-max', 'best-5', 'mean-diff']:
                y_preds, _ = model(X, patch_order=patch_order, multioutput=True)
                y_preds = torch.stack(y_preds, dim=1)
                    
                if prediction_index == 'best-e' or prediction_index == 'best-5':
                    probs = F.softmax(y_preds, -1) + 1e-8
                    entropy = - probs * torch.log(probs)
                    entropy = torch.sum(entropy, -1)                
                    idx.append( torch.argmin(entropy[:, offset1: offset2], dim=1)  + offset1 )
                elif prediction_index == 'best-max':                     
                    y_max = y_preds[:, offset1: offset2].max(dim=-1)[0]
                    idx.append( torch.argmax(y_max, dim=1) + offset1 )
                elif prediction_index == 'mean-diff': 
                    probs = y_preds #F.softmax(y_preds, -1)
                    p_max = torch.max(probs, -1)[0].unsqueeze(2)
                    delta_p = (p_max - probs)**2
                    diff = delta_p.sum(dim=-1) / (delta_p.size(-1) - 1)
                    idx.append( torch.argmax(diff[:, offset1: offset2], dim=1)  + offset1 )
        
                 
                if prediction_index in ['best-e', 'best-max', 'mean-diff']:  
                    y_preds = torch.stack([y_preds[i, idx[-1][i], :] for i in range(len(idx[-1]))], dim=0)
                elif prediction_index == 'best-5':
                    y_preds_copy = torch.zeros(y_preds.size(0), y_preds.size(2)).to(device)
                    for i in range(len(idx[-1])):
                        for j in range(idx[-1][i], min(idx[-1][i].item() + 5, model.seq_len), 1):
                            y_preds_copy[i] += y_preds[i, j]
                            
                        y_preds_copy[i] /= (min(idx[-1][i].item() + 5, model.seq_len) - idx[-1][i] )
                        
                    y_preds = y_preds_copy
            else:
                if ensemble:
                    y_preds = rnn_ensemble(model, X, patch_order=patch_order)
                else:    
                    y_preds, _ = model(X, patch_order=patch_order, prediction_index=prediction_index)
           
            n += y_true.size(0)
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()
            
    if prediction_index in ['best-e', 'best-max', 'best-5', 'mean-diff']:
        idx = torch.cat(idx, dim=0)
        return (correct_preds/n).item(), idx
    torch.cuda.empty_cache()
    return (correct_preds/n).item()


def predict(model, data_loader, device, patch_order=None, prediction_index=-1, offset1=0, offset2=256, repeat=0, ensemble=False):
    correct_preds = 0 
    n = 0
    idx = []
    
    predictions = []
        
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)
             
            if prediction_index in ['best-e', 'best-max', 'best-5', 'mean-diff']:
                y_preds, _ = model(X, patch_order=patch_order, multioutput=True)
                y_preds = torch.stack(y_preds, dim=1)
                    
                if prediction_index == 'best-e' or prediction_index == 'best-5':
                    probs = F.softmax(y_preds, -1) + 1e-8
                    entropy = - probs * torch.log(probs)
                    entropy = torch.sum(entropy, -1)                
                    idx.append( torch.argmin(entropy[:, offset1: offset2], dim=1)  + offset1 )
                elif prediction_index == 'best-max':                     
                    y_max = y_preds[:, offset1: offset2].max(dim=-1)[0]
                    idx.append( torch.argmax(y_max, dim=1) + offset1 )
                elif prediction_index == 'mean-diff': 
                    probs = y_preds #F.softmax(y_preds, -1)
                    p_max = torch.max(probs, -1)[0].unsqueeze(2)
                    #print(p_max.size(), p_max.expand(-1, -1, probs.size(-1)).size())
                    delta_p = (p_max - probs)**2
                    diff = delta_p.sum(dim=-1) / (delta_p.size(-1) - 1)
                    idx.append( torch.argmax(diff[:, offset1: offset2], dim=1)  + offset1 )
        
                 
                if prediction_index in ['best-e', 'best-max', 'mean-diff']:  
                    y_preds = torch.stack([y_preds[i, idx[-1][i], :] for i in range(len(idx[-1]))], dim=0)
                elif prediction_index == 'best-5':
                    y_preds_copy = torch.zeros(y_preds.size(0), y_preds.size(2)).to(device)
                    for i in range(len(idx[-1])):
                        for j in range(idx[-1][i], min(idx[-1][i].item() + 5, model.seq_len), 1):
                            y_preds_copy[i] += y_preds[i, j]
                            
                        y_preds_copy[i] /= (min(idx[-1][i].item() + 5, model.seq_len) - idx[-1][i] )
                        
                    y_preds = y_preds_copy
            else:
                if ensemble:
                    y_preds = rnn_ensemble(model, X, patch_order=patch_order)
                else:    
                    y_preds, _ = model(X, patch_order=patch_order, prediction_index=prediction_index, repeat=repeat)
           
            n += y_true.size(0)
            #correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()
            predictions.append(y_preds.argmax(dim=1))
            
    predictions = torch.cat(predictions, dim=0)
            
    if prediction_index in ['best-e', 'best-max', 'best-5', 'mean-diff']:
        idx = torch.cat(idx, dim=0)
        return predictions, idx
    torch.cuda.empty_cache()
    return predictions


def accuracy_cnn(model, data_loader, device, supcon=False):
    correct_preds = 0 
    n = 0
    idx = []   
        
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)
            n += y_true.size(0)
            
            if supcon:
                y_preds, _ = model(X)
            else:
                y_preds = model(X)
                            
            correct_preds += (y_preds.argmax(dim=1) == y_true).float().sum()
  
    torch.cuda.empty_cache()
    return (correct_preds/n).item()


def f1score(model, data_loader, device, patch_order=None, prediction_index=-1):
    f1 = 0 
    n = 0
    
    y_preds = []
    labels = []
       
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            
            labels.append( y_true.detach().numpy() )
                        
            if patch_order != None:
                y_preds.append( model(X, patch_order=patch_order, prediction_index=prediction_index)[0].argmax(dim=1).detach().cpu().numpy() )
            else:    
                y_preds.append( model(X, prediction_index=prediction_index)[0].argmax(dim=1).detach().cpu().numpy() )
            
            n += y_true.size(0)
            
        labels = np.concatenate(labels) 
        y_preds = np.concatenate(y_preds) 
        f1 = f1_score(labels, y_preds, average='macro')

    return f1

def f1score_cnn(model, data_loader, device, supcon=False):
    f1 = 0 
    n = 0
    
    y_preds = []
    labels = []
       
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            labels.append( y_true.detach().cpu().numpy() )   
            
            if supcon:
                y_hat, _ = model(X)
            else:
                y_hat = model(X)
                
            y_hat = y_hat.argmax(dim=1).detach().cpu().numpy()
                
            y_preds.append( y_hat )
            n += y_true.size(0)
            
        labels = np.concatenate(labels) 
        y_preds = np.concatenate(y_preds) 
        f1 = f1_score(labels, y_preds, average='macro')

    return f1


def choose_criterion(criterion_name="CE", weights=None):
    if criterion_name == 'CE':
        
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        
    return criterion   


def display_confusion_matrix(model, data_loader, device, num_classes=8, save_fname=None):
    plt.figure(figsize = (9, 7))
    
    y_pred = []
    y_true = []
    
    # num_classes = num_classes
    with torch.no_grad():
        model.eval()
        for X, y in data_loader:

            X = X.to(device)
            y = y.to(device)

            y_pred.append(model(X).argmax(dim=1)[0] )
            y_true.append(y)

        y_true =  torch.cat(y_true, dim=0).detach().cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).detach().cpu().numpy()
        
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2f}%".format(value) for value in cm_normalized.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_percentages, group_counts)]
    
    labels = np.asarray(labels).reshape(num_classes, num_classes)
    group_percentages = np.asarray(group_percentages).reshape(num_classes, num_classes)
    
    heatmap = sns.heatmap(cm_normalized, annot=labels, fmt='', cmap='Blues') #, colorbar_kw={'label': 'Percenatge'})
    heatmap.set_xlabel('Predicted label', fontsize=12)
    heatmap.set_ylabel('True label', fontsize=12)
    heatmap.set_title('Confusion matrix', fontsize=12)
    
    heatmap.set_xticklabels(list(TYPE_TO_CLASS.keys())[:num_classes], fontsize=10, rotation=30)
    heatmap.set_yticklabels(list(TYPE_TO_CLASS.keys())[:num_classes], fontsize=10, rotation=30)
    
    if save_fname != None:
        fig = heatmap.get_figure()
        fig.savefig(save_fname)
    


def choose_optimizer(model, optimizer_name, lr=1e-3, wd=1e-5):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        
    return optimizer


