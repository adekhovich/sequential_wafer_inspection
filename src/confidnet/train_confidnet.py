import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import random

from utils.utils import *
from confidnet.utils.utils import *


def train_rnn(confidnet, rnn, train_loader, criterion, optimizer, device):
    confidnet.train()
    freeze_parameters(rnn)
    #rnn.eval()
    
    running_loss = 0
           
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
                
        # Forward pass
        
        # -----------------   Multiple Heads loss -----------------
        
        
        y_hat, hidden = rnn(X, multioutput=True)
        hidden = torch.stack(hidden, dim=0)
        
        y_hat = torch.stack(y_hat, dim=0)
        confidence, _ = confidnet(hidden.transpose(0, 1), multioutput=True)
        confidence = torch.stack(confidence, dim=0)
        
        loss_conf = criterion(confidence, y_hat, y_true)
        loss = loss_conf.mean()
        
      
        # -----------------   -------------------- -----------------
        
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(confidnet.parameters(), max_norm=0.5)
        optimizer.step()   
        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return confidnet, optimizer, epoch_loss



def training_loop_confidnet(confidnet, rnn, criterion, optimizer, scheduler, train_loader, valid_loader,
                            epochs, device, file_name='model.pth', print_every=1):
 
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
    
    #rnn.eval()
    freeze_parameters(rnn)

    
    gen_model = None
    
    for epoch in range(0, epochs):
        # training
        confidnet, optimizer, train_loss = train_rnn(confidnet, rnn, train_loader, criterion, optimizer,device)
        
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            confidnet, valid_loss = validate_rnnconfidnet(confidnet, rnn, valid_loader, criterion, device, multioutput=True)
            valid_losses.append(valid_loss)
            
            if scheduler != None:
                scheduler.step()
       
        
        if (valid_loss < best_loss):
            torch.save(confidnet.state_dict(), file_name)
            best_loss = valid_loss
    
        
        if epoch % print_every == (print_every - 1):
                            
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t')

    #plot_losses(train_losses, valid_losses)
    
    return confidnet, (train_losses, valid_losses)