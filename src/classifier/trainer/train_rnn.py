import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import random

import torchvision
#from torchvision.transforms import v2

from utils.utils import accuracy, validate, img_to_patch

    
def train(model, train_loader, criterion, optimizer, num_heads, device):
    model.train()
    running_loss = 0
       
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
           
        # Forward pass
        # -----------------   Multiple Heads loss -----------------
        if num_heads != -1:
            train_heads = list(random.sample(range(0, model.seq_len), num_heads))   
                       
            y_hat, _ = model(X, prediction_index=train_heads) 
            loss = torch.zeros(num_heads, dtype=torch.float64)
            for i in range(len(train_heads)):
                loss[i] = criterion(y_hat[i], y_true)
               
            loss = loss.sum() / num_heads
        else:
            y_hat, _ = model(X, prediction_index=-1)
            loss = criterion(y_hat, y_true)
        
        # -----------------   -------------------- -----------------
        
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



def training_loop_rnn(model, criterion, optimizer, scheduler, train_loader, valid_loader, 
                      epochs, num_heads, device, file_name='model.pth', print_every=1):
 
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    
    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(model, train_loader, criterion, optimizer, num_heads, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(model, valid_loader, criterion, device, multioutput=True)
            
            if scheduler != None:
                scheduler.step()
       
        valid_acc = accuracy(model, valid_loader, device=device)    
        
        if (valid_acc > best_acc):
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc
        
        if epoch % print_every == (print_every - 1):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  #f'Valid loss: {valid_loss:.4f}\t'
                  #f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    
    return model, (train_losses, valid_losses)