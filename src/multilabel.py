import os
import sys
import random
import warnings
import pandas as pd
import numpy as np
import torch
from net.modules import MultilabelVI, MultilabelBinaryVI, MultilabelMaxI, MultilabelVItrajectory, MultilabelVIDecomposeI
from net.learner import Learner
from net.loss_functions import WeightedCrossEntropyLoss, WeightedBinaryCrossEntropyLoss
from net.utils import CSVLogger, get_post_neg_weight
from net.metrics import  fit_metrics, compute_metrics
from net.baseline import LearnBaseline
from utils.data_generator import get_data, get_loaders
from utils.utils import generate_input_feature, create_paa, compute_active_non_active_features, generate_input_feature
from collections import OrderedDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
warnings.filterwarnings("ignore")


seed = 4783957
print("set seed")
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)


class Multilabel():

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """
        self.MODEL_NAME = params.get('model_name',"CNNModel")
        self.logs_path =params.get('log_path',"../logs")
        self.checkpoint_path =params.get('checkpoint_path',"../checkpoints")
        self.results_path = params.get('results_path',"../results/")
        self.models = OrderedDict()
        self.n_epochs = params.get('n_epochs', 200)
        self.feature = params.get('feature', 'vi')
        self.batch_size = params.get('batch_size',4)
        self.dataset = params.get('dataset',"plaid")
        self.optim_params  = params.get('optim_params',{})
        self.dropout = params.get('dropout', 0.1)
        self.weight_loss = params.get('weight_loss',False)
        

    def partial_fit(self):

        

        #load data
        
        current, voltage, labels, I_max = get_data(data_type=self.dataset)
        
        if self.feature =='decompose-current-vi':
                I_max  = compute_active_non_active_features(current, voltage, emb_size=50)  
                input_feature = generate_input_feature(current, voltage, 'vi', 50, True)     
        else:    
            input_feature = generate_input_feature(current, voltage, self.feature, width=50,  p=2)
        in_size = input_feature.size(1) if self.dataset=="plaid" else 3
        

        #encode labels
        mlb = MultiLabelBinarizer()
        mlb.fit(labels)
        y = mlb.transform(labels)
        classes=list(np.unique(np.hstack(labels)))
        self.num_class=len(classes)

        
        

    

        metric_fn = fit_metrics()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        

        # perform multilabel stratified cross-validation to ensure equal distribuitions of states
        y_pred_total = []
        y_test_total = []
        computed_metrics ={}
        #mskf = MultilabelStratifiedShuffleSplit(n_splits=4, test_size=0.20, random_state=42)
        mskf = MultilabelStratifiedKFold(n_splits=10, random_state=42)
        k = 1
        self.arch = f'{self.MODEL_NAME}_{self.dataset}_{self.feature}'
        for train_index, test_index in  mskf.split(input_feature, y):
        
        
            Xtrain, Xtest = input_feature[train_index], input_feature[test_index]
            ytrain, ytest = y[train_index], y[test_index]
            Itrain, Itest = I_max[train_index], I_max[test_index]
            

            if self.MODEL_NAME=="MLKNNbaseline" or self.MODEL_NAME=="BRkNNbaseline" or self.MODEL_NAME=="BRSVCbaseline":
                
                self.saved_model_path   = f'{self.checkpoint_path}/{self.arch}_fold_{str(k)}_baseline_checkpoint.pkl'
                self.learner_baseline = LearnBaseline(model_name=self.MODEL_NAME)
                img_tra=Xtrain.reshape(len(Xtrain), -1)
                img_test=Xtest.reshape(len(Xtest), -1)
                if self.feature in [ "decompose_current_vi", "vi_imax", "wrg_imax", "distance_imax", "decompose_current_rms"]: 
                    img_tra  = np.concatenate([Xtrain.reshape(len(Xtrain), -1), Itrain.reshape(len(Itrain), -1)], 1)
                    img_test = np.concatenate([Xtest.reshape(len(Xtest), -1), Itest.reshape(len(Itest), -1)], 1)
                
                        
                if self.feature == "i-max":
                    img_tra=Itrain
                    img_test=Itest
                    
                
                if os.path.isfile(self.saved_model_path):
                    print("=> The baseline model has been trained and saved in '{}'".format(self.saved_model_path))
                else:
                    print(f"...Multilabel partial_fit running for  {self.arch} on {self.dataset} data.....")
                    #fit model
                    self.learner_baseline.fit(img_tra,img_test, ytrain, ytest, self.saved_model_path)
                    
                    #get prediction
                pred = self.learner_baseline.get_prediction(img_test, self.saved_model_path)
                ground_t = ytest
            else:
                pos_ratio = get_post_neg_weight(ytrain)
                loaders = get_loaders(Xtrain, Xtest, Itrain, Itest, ytrain, ytest, batch_size=self.batch_size)
                
                self.arch = f"{self.MODEL_NAME}_{self.dataset}_{self.feature}"
                opt_params = self.optim_params['opt_params']
                opt_name   =  self.optim_params['opt_name']
                opt_scheduler = self.optim_params['sheduler_name']
                singlelabel=True if self.optim_params['softmax'] else False
                if singlelabel:
                    self.arch  = f"{self.arch}_softmax"
                    num_class =  self.num_class*2
                else: 
                    self.arch  = f"{self.arch}_sigmoid"
                    num_class =  self.num_class
                
              
                 
                if self.feature in [ "vi_imax", "wrg_imax", "distance_imax"]:
                        model =  MultilabelVI(in_size=in_size, d_model=128, out_size=num_class, dropout=0.25)    
                        
                if self.feature in [ "vi", "wrg", "distance", "decomposed_distance", "decomposed_wrg", "decomposed_vi",  "decomposed_distance_rms"]:
                    model =  MultilabelBinaryVI(in_size=in_size, d_model=128, out_size=num_class,  dropout=0.25) 
                     
                if self.feature == "i-max":
                    model =  MultilabelMaxI(in_size=in_size, d_model=128, out_size=num_class,  dropout=0.25) 
               
                if self.feature in ["decomposed_current", "current",  "decompose_current_rms"]:
                    model =  MultilabelVItrajectory(in_size=in_size, d_model=128, out_size=num_class,  dropout=0.25) 
                if self.feature in ["decomposed_current-vi", "decomposed_current_wrg", "decomposed_current_distance"]:
                    model =  MultilabelVIDecomposeI(in_size=in_size, d_model=128, out_size=num_class,  dropout=0.25)                           
                model = model.to(self.device)
                
                
                #define loss function and optimisation
                if self.weight_loss:
                    if singlelabel:
                        criterion = WeightedCrossEntropyLoss(pos_ratio=None)
                    else:
                        criterion = WeightedBinaryCrossEntropyLoss(pos_ratio=None)
                    self.arch  = f"{self.arch}_weighted_loss"
                else:
                    if singlelabel:
                        criterion = WeightedCrossEntropyLoss(pos_ratio=None)
                    else:
                        criterion = WeightedBinaryCrossEntropyLoss(pos_ratio=None)

                self.saved_model_path   = f"{self.checkpoint_path}/{self.arch}_fold_{str(k)}_checkpoint.pt"
                csv_logger = CSVLogger(filename=f'{self.logs_path}/{self.arch}_fold_{str(k)}_log.csv',
                            fieldnames=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])

        
                

                self.learner = Learner(model, criterion, opt_name, opt_params, self.device, metric_fn,
                    self.saved_model_path, self.n_epochs, self.arch, opt_scheduler,single_label=singlelabel,
                    patience=50, rnn=False, grad_clip=30)
            
                start_epoch, best_score = self.learner.load_saved_model()
                print(f"...Multilabel partial_fit running for  {self.arch} on {self.dataset} data.....")
        
                self.learner.fit(loaders['train'], loaders['val'], csv_logger, self.batch_size,
                best_score, start_epoch, anneal_factor = 0.1,
                anneal_against_train_loss= False,  anneal_with_restarts = False,
                anneling_patience = 20)

                #get prediction
                _, _ = self.learner.load_saved_model()
                pred, ground_t = self.learner.predict(loaders['val'], self.num_class, self.batch_size, self.results_path)
            
            results = compute_metrics(ground_t, pred, all_metrics=True, verbose=False)
            df=pd.DataFrame.from_dict(results, orient="index")
            computed_metrics[k] = df
            y_pred_total.append(pred)
            y_test_total.append(ground_t)
            print(f"............ {self.MODEL_NAME} ..............")
            print(df.to_string())
            k+=1
        np.save(f"{self.results_path}{self.arch}_pred.npy", np.vstack(y_pred_total))
        np.save(f"{self.results_path}{self.arch}_true.npy", np.vstack(y_test_total))
        np.save(f"{self.results_path}{self.arch}_results.npy", computed_metrics)


       


    def return_network(self, num_class, in_size):
        
        if self.MODEL_NAME == "CNNModel":
            net =  MultilabelVI(in_size=in_size, d_model=128, out_size=num_class)
        
        return net

def get_experiments(dataset, optim_params, feature):
    experiments = {'CNNModel': Multilabel({'n_epochs':500,'batch_size':16,
        'model_name':"CNNModel",
        'optim_params':optim_params,
        'feature':feature,
        "dataset":dataset}),
        'MLKNNbaseline': Multilabel({'n_epochs':1,'batch_size':4,
        'model_name':"MLKNNbaseline",
        'optim_params':optim_params,
        'feature':feature,
        "dataset":dataset}),
        'BRKNNbaseline': Multilabel({'n_epochs':1,'batch_size':4,
        'model_name':"BRkNNbaseline",
        'optim_params':optim_params, 
        'feature':feature,
        "dataset":dataset})
        }
    return experiments

if __name__ == "__main__":
    optim_params ={"opt_params": {"lr": 1e-3, "betas":(0.9, 0.98), "eps":1e-9},
             "opt_name":"adam",
             "sheduler_name":"ReduceLROnPlateau",
             "softmax":True
            }

    for feature in ["decomposed_current", "decomposed_distance", "vi", "current", "distance"]:
        for dataset in ["plaid"]:
            experiments = get_experiments(dataset, optim_params, feature)
            for model_name, clf in experiments.items():
                clf.partial_fit()