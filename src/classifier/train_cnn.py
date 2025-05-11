import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.utils import accuracy_cnn, validate_cnn

    
def train( model, train_loader, criterion, optimizer, device, supcon_loss=None):
   
    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        if supcon_loss == None:
            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_hat = model(X) 
            loss = criterion(y_hat, y_true)
        else:
            X1 = X[0]
            X2 = X[1]
            X = torch.cat([X1, X2], dim=0).to(device)
            y_true = y_true.to(device)
            c = 0.1
            y_hat, projected = model(X)     # ???????????????
                          
            bsz = X1.size(0)
            h1, h2 = torch.split(projected, [bsz, bsz], dim=0)
            h = torch.cat([h1.unsqueeze(1), h2.unsqueeze(1)], dim=1) 
                
            y_true_cat = torch.cat([y_true, y_true], dim=0)
            loss = criterion(y_hat, y_true_cat) + c * supcon_loss(features=h, labels=y_true)
           
            
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss



def training_loop(model, criterion, optimizer, scheduler, train_loader, valid_loader, 
                  epochs, device, file_name='model.pth', supcon_loss=None, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    best_acc = 0
    train_losses = []
    valid_losses = []
 
    # Train model
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
    #                                                       factor=0.33, patience=3, verbose=1)
    
    supcon = supcon_loss != None

    for epoch in range(0, epochs):
        # training
        model, optimizer, train_loss = train(model, train_loader, criterion, optimizer, 
                                             device, supcon_loss=supcon_loss)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            if not supcon:
                model, valid_loss = validate_cnn(model, valid_loader, criterion, device)
                valid_losses.append(valid_loss)
            
            if scheduler != None:
                scheduler.step()

        
        if not supcon:
            train_acc = accuracy_cnn(model, train_loader, device=device)
            
        valid_acc = accuracy_cnn(model, valid_loader, device=device, supcon=supcon)    

        if (valid_acc > best_acc):
            torch.save(model.state_dict(), file_name)
            best_acc = valid_acc

        if epoch % print_every == (print_every - 1):    
            if not supcon:
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                      f'Epoch: {epoch}\t'
                      f'Train loss: {train_loss:.4f}\t'
                      f'Valid loss: {valid_loss:.4f}\t'
                      f'Train accuracy: {100 * train_acc:.2f}\t'
                      f'Valid accuracy: {100 * valid_acc:.2f}')
            else:
                print(f'{datetime.now().time().replace(microsecond=0)} --- '
                      f'Epoch: {epoch}\t'
                      f'Train loss: {train_loss:.4f}\t'
                      f'Valid accuracy: {100 * valid_acc:.2f}')

    #plot_losses(train_losses, valid_losses)
    
    return model, (train_losses, valid_losses)