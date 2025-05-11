import torch
import torch.nn as nn
import torch.nn.functional as F



class RNNSelfConfidMSELoss(nn.modules.loss._Loss):
    def __init__(self, num_classes=8, weight=2):
        super().__init__()
        
        self.num_classes = num_classes
        self.weight = weight
        self.device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, confidence, y_preds, target):
        probs = F.softmax(y_preds, dim=-1)
        weights = torch.ones((y_preds.size(0), y_preds.size(1))).type(torch.FloatTensor).to(self.device)
        weights[(probs.argmax(dim=-1) != target)] *= self.weight
        labels_hot = F.one_hot(target, self.num_classes).unsqueeze(0).repeat(y_preds.size(0), 1, 1).to(self.device)
        
        loss = weights * (confidence.squeeze(-1) - (probs * labels_hot).sum(dim=-1)) ** 2
        
        return torch.mean(loss, dim=1)

