import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# !!!!  https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

class MyGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0, bias=True):
        super(MyGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias
        self.x2h = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        #self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
            
    def apply_timestep(self, x, hidden):
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        #gate_x = gate_x.squeeze()
        #gate_h = gate_h.squeeze()
        
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + (reset_gate * h_n))
        
        hy = new_gate + input_gate * (hidden - new_gate)
        
        return hy
        
    
    def forward(self, x, hidden, t=0):
        #cell_out = []
        
        hidden = self.apply_timestep(x, hidden) 
        #cell_outs.append(hidden)
        
        #cell_outs = torch.cat(cell_outs, dim=0)
        
        return hidden