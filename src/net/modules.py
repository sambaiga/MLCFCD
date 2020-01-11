import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import create_conv2, MLPLayer,  ScaledDotProductAttention
from .utils import  initilize_final_layer


class ConvEncoder(nn.Module):
    def __init__(self, in_size=1, out_channels=[16, 32, 64, 128], 
                 kernel_size=[5,5,3,3],  bias=True, bn=True, activation=nn.ReLU(True),
                 strides=[2,2,2,2]):
        super(ConvEncoder, self).__init__()
        self.layers = []
        
        for i in range(len(out_channels)):
            if i==0:
                layer=create_conv2(in_size, out_channels[i], kernel_size[i], bias=True, dilation=1, stride=strides[i])
            else:
                layer=create_conv2(out_channels[i-1], out_channels[i], kernel_size[i], bias=True, dilation=1, stride=strides[i])
            
            self.layers.append(layer)
            self.add_module("cnn_layer_"+str(i+1), layer)  
            if bn: 
                bn=nn.BatchNorm2d(out_channels[i], eps=1e-4, momentum=0.95)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)
            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)
        
        self.init_weights()
        self.cnn_network =  nn.Sequential(*self.layers)
        
    
    def init_weights(self):
        for layer in self.layers:
            try:
                if isinstance(layer, nn.Conv2d):
                    init.kaiming_normal_(layer.weight.data)
                    init.normal_(layer.bias.data)
                    
                if isinstance(layer, nn.BatchNorm2d):
                    init.normal_(layer.weight.data, mean=1, std=0.02)
                    init.constant_(layer.bias.data, 0)
            except: pass
            
    def forward(self, x):
        out=F.adaptive_avg_pool2d(self.cnn_network(x), 1)
        return out.view(x.size(0), -1)


class MultilabelVI(nn.Module):
    def __init__(self, in_size=None,  d_model=128,  
                out_size=12, dropout=0.1, pi=None):
        super(MultilabelVI, self).__init__()
        self.enc_cnn = ConvEncoder(in_size=in_size,bn=False)
        self.enc_imax = MLPLayer(in_size=in_size, hidden_arch=[16, 32, 64], output_size=128, batch_norm=False)
        self.dec_out  = MLPLayer(in_size=d_model*2, hidden_arch=[512, 1024], output_size=128, batch_norm=False)
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(d_model, out_size)
        

        initilize_final_layer(self.fc_out, pi)
        

    def init_hidden(self, bsz):
        pass
            
    def forward(self, x, i_max):
        
        enc_out_vi   = self.dropout(self.enc_cnn(x))
        enc_out_imax = self.dropout(self.enc_imax(i_max))
        
        comb_out    = torch.cat([enc_out_imax, enc_out_vi], 1)
        dec_out     = self.dropout(self.dec_out(comb_out))
        out         = self.fc_out(dec_out)
        return out




