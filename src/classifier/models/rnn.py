import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import img_to_patch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()



class RNN(nn.Module):

    def __init__(self, input_dim=16, hidden_dim=128, num_classes=8, seq_len=64, num_layers=2, patch_size=4, embedding_dim=64,
             dropout_prob=0.0, cell_type="GRU", batch_first=True, layer_norm=True, permute_patches=False):
        super(RNN, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.layer_norm = layer_norm
        
        self.dropout_prob = dropout_prob
        self.permute_patches = permute_patches

        layers = []
        ln_layers = []
        if self.cell_type == "GRU" or self.cell_type == "gru":
            cell = nn.GRU
        elif self.cell_type == "RNN" or self.cell_type == "rnn":
            cell = nn.RNN
        elif self.cell_type == "LSTM" or self.cell_type == "lstm":
            cell = nn.LSTM
        else:
            raise NotImplementedError
                
        if permute_patches:
            self.projection = nn.Linear(in_features=input_dim, out_features=embedding_dim)                        
            self.pos_embedding = nn.Embedding(num_embeddings=seq_len, embedding_dim=embedding_dim) 
            input_dim = embedding_dim

        for i in range(num_layers):
            if i == 0:
                c = cell(input_dim, hidden_dim, dropout=dropout_prob)
            else:
                c = cell(hidden_dim, hidden_dim, dropout=dropout_prob)
            
            if layer_norm:
                ln = nn.LayerNorm(hidden_dim)
            else:
                ln = None
            
            layers.append(c)
            ln_layers.append(ln)
            
        self.cells = nn.Sequential(*layers)
        self.ln_layers = nn.Sequential(*ln_layers)
        
        self.fc = nn.Linear(hidden_dim, num_classes)

    def positional_encoding(self, X, patch_order):
        X = self.projection(X)
        positions_emb = self.pos_embedding(patch_order)
        X = X + positions_emb
            
        return X
    

    def forward(self, inputs, hidden=None, multioutput=False, patch_order=None, prediction_index=-1):
        inputs = img_to_patch(inputs, patch_size=self.patch_size, flatten_channels=True)
                
        if self.permute_patches:
            if patch_order == None:
                patch_order = torch.stack([torch.randperm(inputs.size(1)) for _ in range(inputs.size(0))]).to(device)
                
            else:
                if patch_order.dim() == 1:
                    patch_order = patch_order.expand(inputs.size(0), -1).to(device)
            
            inputs = inputs.gather(dim=1, index=patch_order.unsqueeze(-1).expand(inputs.shape))  # patch permutation  
            
            inputs = self.positional_encoding(inputs, patch_order)
        
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        
        outputs = []
        for i, (cell, layer_norm) in enumerate(zip(self.cells, self.ln_layers)):
            if hidden is None:
                inputs, _ = self.rnn_layer(cell, inputs, i=i)
            else:
                inputs, hidden[i] = self.rnn_layer(cell, inputs, hidden[i], i=i)

            if self.layer_norm:
                inputs = layer_norm(inputs)
    
            outputs.append(inputs)
             
        if self.batch_first:
            output_rnn = inputs.transpose(0, 1)
            
            if multioutput:
                fc_output = [self.fc(output_rnn[:, i, :]) for i in range(self.seq_len)]
                output_rnn = [output_rnn[:, i, :] for i in range(self.seq_len)]
            elif type(prediction_index) is list:
                fc_output = [self.fc(output_rnn[:, i, :]) for i in prediction_index]
                output_rnn = [output_rnn[:, i, :] for i in prediction_index]
            else:    
                fc_output = self.fc(output_rnn[:, prediction_index, :])
                output_rnn = output_rnn[:, prediction_index, :]
        
        return fc_output, output_rnn

    def rnn_layer(self, cell, inputs, hidden=None, i=0):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        if hidden is None:
            outputs, hidden = self._apply_cell(inputs, cell, batch_size, hidden_size, i=i)
        else:
            outputs, hidden = self._apply_cell(inputs, cell, batch_size, hidden_size, hidden=hidden, i=i)

        return outputs, hidden

    def _apply_cell(self, inputs, cell, batch_size, hidden_size, hidden=None, i=0):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size, hidden_size).unsqueeze(0)
        
        outputs, hidden = cell(inputs, hidden)
      
        return outputs, hidden


    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        #if use_cuda:
        hidden = hidden.to(device) #cuda()
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory.to(device) #cuda()
            return (hidden, memory)
        else:
            return hidden
