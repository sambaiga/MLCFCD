import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_weights(pos_ratio, eps=15):
    """[calcultae postive and negative weights given by 
        w_p = 1/(N_p/N_T)]
        w_n = 1/(N_n/N_T) where N_p+N_n=N_T
    """
    #get posive weights
    w_p = 1/pos_ratio
    w_p = np.where(w_p>eps, eps, w_p)
    
    #get negative weights
    w_n = 1/(1 - pos_ratio)
    w_n = np.where(w_n>eps, eps, w_n)
    
    return {1:w_p, 0:w_n}

def get_batched_weights(targets:torch.Tensor, weights_dict:dict):
    """[create batched weighted with size Nb, C]
    
    Arguments:
        targets {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    weights = torch.zeros(targets.size(), requires_grad=False).to(targets.device)
    for col in range(targets.shape[1]):
        weights[:,col]= torch.DoubleTensor([weights_dict[targets[:,col][idx].item()][col]\
                for idx in range(targets.size(0))])
    return weights



class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, pos_ratio=None):
        """[weighted cross entropy loss with f2_beta loss]
        
        Keyword Arguments:
            pos_ratio {[type]} -- [description] (default: {None})
            epsilon {[type]} -- [description] (default: {1e-9})
            beta2 {int} -- [description] (default: {1})
        """
        
        self.pos_ratio =  pos_ratio
        if self.pos_ratio is not None:
            self.weights = get_weights(pos_ratio)
        super(WeightedCrossEntropyLoss, self).__init__()
        
    
    def forward(self, inputs, targets):
        batch_size, label_len = targets.size()
        inputs = inputs.view(batch_size, -1, label_len)
        
        if self.pos_ratio is not None:
            weights=get_batched_weights(targets, self.weights)
            nll_los = F.nll_loss(F.log_softmax(inputs, 1), targets, reduction="none")
            nll_los = torch.mean(nll_los * weights)
        else:
            nll_los = F.nll_loss(F.log_softmax(inputs, 1), targets)
        
        
        return nll_los

 
class WeightedBinaryCrossEntropyLoss(nn.Module):

    def __init__(self, pos_ratio=None):
        """[weighted cross entropy loss with f2_beta loss]
        
        Keyword Arguments:
            pos_ratio {[type]} -- [description] (default: {None})
            epsilon {[type]} -- [description] (default: {1e-9})
            beta2 {int} -- [description] (default: {1})
        """
        
        self.pos_ratio =  pos_ratio
        if self.pos_ratio is not None:
            self.weights = get_weights(pos_ratio)
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        
    
    def forward(self, inputs, targets):
        
        
        if self.pos_ratio is not None:
            weights=get_batched_weights(targets, self.weights)
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), weight=weights)
            
        else:
            
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float())
        

        return bce_loss


        
