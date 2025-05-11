import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import img_to_patch
from patch_selector.models.cells import MyGRUCell


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()



class Confidnet(nn.Module):

    def __init__(self, input_dim=256, hidden_dim=128, seq_len=64, num_layers=2, 
                 dropout_prob=0.0, cell_type="GRU", layer_norm=False, batch_first=True):
        super(Confidnet, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.layer_norm = layer_norm
        
        self.dropout_prob = dropout_prob

        layers = []
        ln_layers = []
        if self.cell_type == "GRU" or self.cell_type == "gru":
            cell = MyGRUCell #nn.GRU, GRUCell
        elif self.cell_type == "RNN" or self.cell_type == "rnn":
            cell = nn.RNN
        elif self.cell_type == "LSTM" or self.cell_type == "lstm":
            cell = nn.LSTM
        else:
            raise NotImplementedError
            
        self.projection = nn.Linear(input_dim, hidden_dim)
        input_dim = hidden_dim            

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
        
        self.fc = nn.Linear(hidden_dim, 1)    
    
    
    def make_step(self, inputs, hidden=None):
        new_hidden = []
        inputs = F.relu(self.projection(inputs))
        for l, (cell, layer_norm) in enumerate(zip(self.cells, self.ln_layers)):
            if  l == 0:
                output = self.rnn_layer(cell, inputs, hidden[l], l=l)
            else:
                output = self.rnn_layer(cell, output, hidden[l], l=l)
                
            new_hidden.append(output.clone())
                
            if self.layer_norm:
                output = layer_norm(output)
                
        output = torch.sigmoid(self.fc(output))       
                        
        return output, new_hidden
    


    def rnn_layer(self, cell, inputs, hidden=None, l=0, t=0):
        n_steps = len(inputs)
        batch_size = inputs.size(0)
        hidden_size = cell.hidden_size

        outputs = self._apply_cell(inputs, cell, batch_size, hidden_size, hidden=hidden, l=l, t=t)

        return outputs

    def _apply_cell(self, inputs, cell, batch_size, hidden_size, hidden=None, l=0, t=0):
       
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size, hidden_size).unsqueeze(0)
              
        
        outputs = cell(inputs, hidden, t=t)

        return outputs


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
        
        
    def load_weights(self, file_name):
        file = torch.load(file_name)
        
        sd = self.state_dict()
        
        for state in list(file):
            if 'weight_ih_l0' in state or 'bias_ih_l0' in state:
                if 'weight' in state:
                    param_name = state.replace('weight_ih_l0', 'x2h.weight')
                else:
                     param_name = state.replace('bias_ih_l0', 'x2h.bias')
            elif 'weight_hh_l0' in state or 'bias_hh_l0' in state:
                if 'weight' in state:
                    param_name = state.replace('weight_hh_l0', 'h2h.weight')
                else:
                     param_name = state.replace('bias_hh_l0', 'h2h.bias')
            else: 
                param_name = state
                
            sd[param_name] = file[state]
            
        self.load_state_dict(sd)    
