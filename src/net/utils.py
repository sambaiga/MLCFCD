import csv
import collections
import numpy as np
import torch
import torch.nn as nn

def get_post_neg_weight(labels):
    
    num_postive=[]
    num_negative=[]
    
    total_labels = labels.shape[0]
    for col in range(labels.shape[1]):
        counter=collections.Counter(labels[:,col])
        P      = sum(counter.values())
        num_negative.append(float(counter[0]))
        num_postive.append(float(counter[1]))
        
    post_ratio = np.array(num_postive)/P
    neg_ratio  = np.array(num_negative)/P

    return post_ratio


class CSVLogger():
    def __init__(self, filename, fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        """
        writer = csv.writer(self.csv_file)
        
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])
        """

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def initilize_final_layer(fc_out, pi):
    nn.init.xavier_normal_(fc_out.weight.data)
    if pi is not None and fc_out.bias.size(0)==len(pi):
        pi = torch.tensor(pi).float()
        for k in range(fc_out.bias.size(0)):
            nn.init.constant_(fc_out.bias[k], -torch.log((1-pi[k])/pi[k]))
    elif pi is not None and fc_out.bias.size(0)==len(pi)*2:
        pi = torch.tensor(pi).float()
        for k in range(fc_out.bias.view(len(pi), 2).size(0)):
            nn.init.constant_(fc_out.bias.view(len(pi), 2)[k], -torch.log((1-pi[k])/pi[k]))
            
    else:
        fc_out.bias.data.fill_(0)