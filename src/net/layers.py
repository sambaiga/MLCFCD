import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def create_conv2(in_channels, out_channels, kernel_size, bias=True, dilation=1, stride=2):
    return nn.Conv2d(
        in_channels,out_channels, kernel_size, bias=bias, dilation=dilation, stride=stride)


def create_conv1(in_channels, out_channels, kernel_size, bias=True, dilation=1, stride=2):
    return nn.Conv1d(
        in_channels,out_channels, kernel_size, bias=bias, dilation=dilation, stride=stride)


class MLPLayer(nn.Module):
    def __init__(self, in_size, hidden_arch=[128, 512, 1024], output_size=1, activation=nn.ReLU(True),
                 batch_norm=True):
        
        super(MLPLayer, self).__init__()
        self.in_size = in_size
        self.output_size = output_size
        layer_sizes = [in_size] + [x for x in hidden_arch]
        self.layers = []
        
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)            
            if batch_norm and i!=0:# if used as discriminator, then there is no batch norm in the first layer
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], output_size)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)
        self.layers.append(activation)
        self.add_module("activation"+str(i+1), activation)
        self.init_weights()
        self.mlp_network =  nn.Sequential(*self.layers)
        
    def forward(self, z):
        return self.mlp_network(z)
        
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight.data)
                    init.normal_(layer.bias.data)
            except: pass
                

    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None,stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -np.inf)

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
