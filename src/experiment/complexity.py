import numpy as np
import time
import math
import torch
from utils.utils import generate_input_feature, create_paa, compute_active_non_active_features, generate_input_feature
from utils.data_generator import get_data, get_loaders
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from net.loss_functions import WeightedCrossEntropyLoss
from experiment.multilabel import set_seed
from net.learner import Learner
from net.modules import MultilabelConv2D
from net.utils import CSVLogger
from net.metrics import  fit_metrics, compute_metrics
from net.baseline import LearnBaseline
import pandas as pd
from utils.visual_functions import *
import os

#https://www.thekerneltrip.com/machine/learning/computational-complexity-learning-algorithms/
class ComplexityEvaluator:

    def __init__(self, n_samples=[0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7, 0.8, 0.9], dataset="plaid", feature="decomposed_distance"):
        self._n_samples = n_samples
        
        set_seed()
        #load data
        current, voltage, labels, I_max = get_data(data_type=dataset)
        self.input_feature = generate_input_feature(current, voltage, feature, width=50,  p=2)
        self.feature = feature
        self.in_size = self.input_feature.size(1) 
        self.checkpoint_path = "../checkpoints"
        self.results_path ="../results/"
        self.optim_params ={"opt_params": {"lr": 1e-3, "betas":(0.9, 0.98), "eps":1e-9},
             "opt_name":"adam",
             "sheduler_name":"ReduceLROnPlateau",
             "softmax":True
            }
        self.logs_path = "../logs"
        #encode labels
        mlb = MultiLabelBinarizer()
        mlb.fit(labels)
        self.y = mlb.transform(labels)
        classes=list(np.unique(np.hstack(labels)))
        self.num_class=len(classes)

        self.metric_fn = fit_metrics()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        

    def _time_samples(self, model="CNN"):
        rows_list = []
        for n_samples in self._n_samples:
            X_train, X_test, y_train, y_test = train_test_split(self.input_feature,
                                                                 self.y, train_size=n_samples, 
                                                                 random_state=42,
                                                                 shuffle=True)
            result = {"N": n_samples}
            if model=="MLKNNbaseline":
                saved_model_path   = f'{self.checkpoint_path}/{model}_{str(n_samples)}_baseline_checkpoint.pkl'
                learner_baseline = LearnBaseline(model_name=model)
                img_tra=X_train.reshape(len(X_train), -1)
                img_test=X_test.reshape(len(X_test), -1)
            
                print(f"Train:{round(len(X_train)*100/len(self.y), 2)}, Test:{round(len(X_test)*100/len(self.y), 2)}")
                start_time = time.time()
                learner_baseline.fit(Xtrain=img_tra,ytrain=y_train, checkpoint_path=saved_model_path)
                result["train_time"] = time.time() - start_time
                
                start_time = time.time()
                pred = learner_baseline.get_prediction(img_test, saved_model_path)
                result["pred_time"] = time.time() - start_time
                ground_t = y_test
                metrics = compute_metrics(ground_t, pred, all_metrics=True, verbose=False)
                result["maF1"]=metrics['maF1']    
                rows_list.append(result)
                os.remove(saved_model_path)
            if model == "CNN":
                saved_model_path   = f'{self.checkpoint_path}/{model}_{str(n_samples)}_cnn_checkpoint.pt'
                loaders = get_loaders(X_train, X_test, X_train, X_test, y_train, y_test, batch_size=16)
                opt_params = self.optim_params['opt_params']
                opt_name   =  self.optim_params['opt_name']
                opt_scheduler = self.optim_params['sheduler_name']
                singlelabel=True if self.optim_params['softmax'] else False
                num_class =  self.num_class*2  
                net =  MultilabelConv2D(in_size=self.in_size, d_model=128, out_size=num_class,  dropout=0.25)
                net = net.to(self.device) 
                criterion = WeightedCrossEntropyLoss(pos_ratio=None)
                saved_model_path   = f'{self.checkpoint_path}/CNN_{str(n_samples)}_complexity_checkpoint.pt'
                csv_logger = CSVLogger(filename=f'{self.logs_path}/{model}_{str(n_samples)}_cnn_log.csv',
                            fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])

                learner = Learner(net, criterion, opt_name, opt_params, self.device, self.metric_fn,
                    saved_model_path, 100, f"{model}_{str(n_samples)}_cnn", opt_scheduler,single_label=singlelabel,
                    patience=100, rnn=False, grad_clip=30)
            
                start_epoch, best_score = learner.load_saved_model()
                print(f"Train:{round(len(X_train)*100/len(self.y), 2)}, Test:{round(len(X_test)*100/len(self.y), 2)}")
                start_time = time.time()
                learner.fit(loaders['train'], loaders['val'], csv_logger, 16,
                best_score, start_epoch, anneal_factor = 0.1,
                anneal_against_train_loss= False,  anneal_with_restarts = False,
                anneling_patience = 20)
                result["train_time"] = time.time() - start_time
                _, _ = learner.load_saved_model()
                img = X_test.float().to(self.device)
                learner.model.eval()
                start_time = time.time() 
                pred = learner.model(img)
                result["pred_time"] = time.time() - start_time
                pred = learner.get_prediction(pred).long().numpy()
                os.remove(saved_model_path )
               
                metrics = compute_metrics(y_test, pred, all_metrics=True, verbose=False)
                result["maF1"]=metrics['maF1']    
                rows_list.append(result)

        return rows_list

    def Run(self, model="CNN"):
        
        if model != "CNN":
            model = "MLKNNbaseline"
          
        results = self._time_samples(model)
        df=pd.DataFrame(results)
        np.save(f"{self.results_path}{model}_{self.feature}_complexity_results.npy", df)
        return df
        
        #data = data.applymap(math.log)
        #linear_model = LinearRegression(fit_intercept=True)
        #linear_model.fit(data[["N", "P"]], data[["Time"]])
        #return linear_model.coef_
        
if __name__ == "__main__":
    complexity = ComplexityEvaluator()  
    results = {}
    for i,  model in enumerate(["CNN"]):
        results[model] = complexity.Run(model) 
    """    
    fig = figure(columns=1)
    plt.plot(results["CNN"]["N"], np.log(results["CNN"]["train_time"]), label="TRAIN CNN", color=colors[1], marker='v',markersize=3)
    plt.plot(results["CNN"]["N"], np.log(results["CNN"]["pred_time"]), label="PRED CNN", color=colors[1], linestyle='--', marker='v',markersize=3)
    plt.plot(results["MLkNN"]["N"], np.log(results["MLkNN"]["train_time"]), label="TRAIN MLkNN", color=colors[0], marker='v',markersize=3)
    plt.plot(results["MLkNN"]["N"], np.log(results["MLkNN"]["pred_time"]), label="PRED MLkNN", color=colors[0], linestyle='--', marker='v',markersize=3)
    plt.legend()
    plt.xlabel("Data sample")
    plt.ylabel("Train Time")
    savefig("train_time", format=".pdf")
    fig = figure(columns=2)
    plt.plot(results["CNN"]["N"].values,results["CNN"]["maF1"]*100, marker='v',markersize=3, label="CNN", color=colors[1])
    plt.plot(results["MLkNN"]["N"].values,results["MLkNN"]["maF1"]*100, marker='v',markersize=3, label="MLkNN", color=colors[0])
    plt.xlabel("Data sample")
    plt.ylabel("$F_1$ macro score ($\%$)")
    savefig("score", format=".pdf")    
    """