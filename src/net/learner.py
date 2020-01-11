from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import os
import sys


class Learner(object):

    def __init__(self, model=None, criterion=None, opt_name: str=None, opt_params: dict=None, device=None, metric=None,
                 saved_model_path: str=None, max_epoch: int=50, arch: str=None, scheduler: str=None,single_label=False,
                 patience: int =20, checkpoints: bool = True, score_mode: str="max",
                 min_delta: float=1e-4, rnn: bool=False, grad_clip=10.0):

        self.scheduler = scheduler
        self.device = device
        self.model = model
        self.criterion = criterion
        self.opt_params = opt_params
        self.opt_name = opt_name
        self.saved_model_path = saved_model_path
        self.checkpoint = checkpoints
        self.max_epoch = max_epoch
        self.arch = arch
        self.single_label = single_label
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.score_mode = score_mode
        self.num_bad_epochs = 0
        self.is_better = None
        self.rnn = rnn
        self.grad_clip = grad_clip
        self._init_is_better(score_mode, min_delta)
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
        self.set_optimizer()
        

    def set_optimizer(self):
        if self.opt_name == "sgd":
            self.opt = torch.optim.SGD(
                self.model.parameters(), **self.opt_params)
        elif self.opt_name == "adam":
            self.opt = torch.optim.Adam(
                self.model.parameters(), **self.opt_params)
        elif self.opt_name == "asgd":
            self.opt = torch.optim.ASGD(self.model.parameters(), **self.opt_params)
        elif self.opt_name == "adamw":
            self.opt = torch.optim.AdamW(self.model.parameters(), **self.opt_params)
        else:
            raise AssertionError("Select correct optimizer")

    def _init_is_better(self, score_mode, min_delta):
        if score_mode not in {'min', 'max'}:
            raise ValueError('mode ' + score_mode + ' is unknown!')

        elif score_mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta

        elif score_mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

    def early_stopping(self, metric, states):

        if self.best is None:
            self.best = metric

        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
            states['best_score'] = self.best

            # save the best model
            self.save_checkpoint(states)

        else:
            self.num_bad_epochs += 1

        if (self.num_bad_epochs >= self.patience) or np.isnan(metric):
            terminate_flag = True

        else:
            terminate_flag = False

        return terminate_flag


    def save_checkpoint(self, state):
        """
        Save best models
        arg:
           state: model states
           is_best: boolen flag to indicate whether the model is the best model or not
           saved_model_path: path to save the best model.
        """

        torch.save(state, self.saved_model_path)

    def get_prediction(self, pred):
        batch_size = pred.size(0)
        if self.single_label:
            pred = pred.view(batch_size, 2, -1)
            pred = torch.max(F.softmax(pred, 1), 1)[1]
        else:
            pred = pred.sigmoid()
            pred = torch.ge(pred, 0.5)
        return pred

    def load_saved_model(self):
        saved_model_path = self.saved_model_path

        if os.path.isfile(saved_model_path):
            #print("=> loading checkpoint '{}'".format(saved_model_path))
            checkpoint = torch.load(
                saved_model_path,  map_location=lambda storage, loc: storage)

            start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'])
            if self.opt_name == "adam-adaptive":
                self.opt.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{0}' (trained for {1} epochs with {2:.3f} best_score)".format(
                saved_model_path, checkpoint['epoch'], best_score))

        else:
            print("=> no checkpoint found at '{}'".format(saved_model_path))

            best_score = 0 if self.score_mode=="max" else 1e20
            start_epoch = 0
        return start_epoch, best_score


    def train(self,  loader, epoch,  batch_size):
        self.model.train()
        loss_avg = 0.
        score_count = 0.
        total = 0.
        total_statistics_list=[]
        progress_bar = tqdm(loader)
        if self.rnn:
            hidden = self.model.init_hidden(batch_size)
        else:
            self.model.init_hidden(batch_size)
        
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            
            if len(data)>2:
                inputs, i_max, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                i_max  = i_max.to(self.device)
            else:
                inputs,  labels=data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
            
            
            self.model.zero_grad()
            if self.rnn:
                out, hidden = self.model(inputs, i_max, hidden)
            else:
                out = self.model(inputs, i_max)
            
            if type(out) is tuple:
                loss = self.criterion(out,  labels)
                pred = self.get_prediction(out[-1])
            else:
                loss = self.criterion(out, labels)
                pred = self.get_prediction(out)

            
            
            new_statistics_list = self.metric.statistics(pred.data, labels.data)
            total_statistics_list = self.metric.update_statistics_list(total_statistics_list, new_statistics_list)
            mean_f1_score, f1_score_list = self.metric.calc_f1_score(total_statistics_list)
                
            #score = metric_fn(pred, labels)
            total += labels.size(0)
            score_count += mean_f1_score
            
            accuracy = score_count/ (i+1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            
            loss_avg +=loss.item()
        
            

            progress_bar.set_postfix(
                loss='%.3f' % (loss_avg / (i + 1)),
                score='%.3f' % accuracy)

        return loss_avg / (i + 1), accuracy


    def test(self, loader,  batch_size):
     
        self.model.eval()    
        
        running_loss = []
        total_statistics_list = []
        
        with torch.no_grad():
            if self.rnn:
                hidden = self.model.init_hidden(batch_size)
            else:
                self.model.init_hidden(batch_size)

            for i, data in enumerate(loader):
            
                
                if len(data)>2:
                    inputs, i_max, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    i_max  = i_max.to(self.device)
                else:
                    inputs,  labels=data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                
                
                if self.rnn:
                    out, hidden = self.model(inputs, i_max, hidden)
                else:
                    out = self.model(inputs, i_max)

                if type(out) is tuple:
                    loss = self.criterion(out,  labels)
                    pred = self.get_prediction(out[-1])
                else:
                    loss = self.criterion(out, labels)
                    pred = self.get_prediction(out)
                
                new_statistics_list = self.metric.statistics(pred.data, labels.data)
                total_statistics_list = self.metric.update_statistics_list(total_statistics_list, new_statistics_list)
                
                
            
                running_loss.append(loss.item())
                #running_acc.append(mean_f1_score)
                
            mean_f1_score, f1_score_list = self.metric.calc_f1_score(total_statistics_list)
        
        
        self.model.train()
        return np.mean(running_loss), mean_f1_score

    def set_opt_scheduler(self, anneal_factor, anneal_against_train_loss,  anneling_patience):
        anneal_mode = 'min' if anneal_against_train_loss else 'max'
        if self.scheduler=="ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=anneal_factor,
                                          patience=anneling_patience, mode=anneal_mode,
                                          verbose=True)

            
        return scheduler

    def fit(self, train_loader, test_loader, csv_logger, batch_size,
            best_score: float=0, start_epoch: int =0, anneal_factor: float = 0.1,
            anneal_against_train_loss: bool = False,  anneal_with_restarts: bool = True,
            anneling_patience: int = 5):
    
        self.best = best_score
        if self.scheduler is not None:
            scheduler = self.set_opt_scheduler(
                anneal_factor, anneal_against_train_loss,  anneling_patience)
            previous_learning_rate = self.opt_params["lr"]

        for epoch in range(start_epoch, self.max_epoch):
            loss_tra, score_tra    = self.train(train_loader, epoch,  batch_size)
            loss_test, score_test  = self.test(test_loader, batch_size)

            tqdm.write('test_loss: %.3f, test_score: %.4f' % (loss_test, score_test))
            row = {'epoch': str(epoch), 'train_loss': str(loss_tra), 'test_loss': str(loss_test),
                    'train_acc': str(score_tra), 'test_acc': str(score_test)}
            
            csv_logger.writerow(row)

            if self.scheduler is not None:
                # get current learning_rate
                for group in self.opt.param_groups:
                    learning_rate = group['lr']

                if learning_rate != previous_learning_rate and anneal_with_restarts and os.path.isfile(self.saved_model_path):
                    print('resetting to best model')
                    checkpoint = torch.load(
                        self.saved_model_path,  map_location=lambda storage, loc: storage)
                    self.model.load_state_dict(checkpoint['state_dict'])

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 1e-6:
                    print(
                        'learning rate {:6f} too small - quitting training!'.format(learning_rate))
                    break

                scheduler.step(score_test)

            
            # call early stopping
            if self.checkpoint and (epoch) >= 2:
                states = {
                    'epoch': epoch+1,
                    'arch': self.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.opt.state_dict() if self.opt_name != "adam-adaptive" else self.opt.optimizer.state_dict()
                }
                if self.early_stopping(score_test, states):
                    print("Early stopping with {:.3f} best score, the model did not improve after {} iterations".format(
                        self.best, self.num_bad_epochs))
                    break
            
        csv_logger.close()

    def predict(self, dataloader, num_class, batch_size, result_path):
    
       
        self.model.eval()
        num_elements = dataloader.len if hasattr(dataloader, 'len') else len(dataloader.dataset)
        batch_size   = dataloader.batchsize if hasattr(dataloader, 'len') else dataloader.batch_size
        num_batches = len(dataloader)
        ground_label = torch.zeros(num_elements, num_class)
        predictions = torch.zeros(num_elements, num_class)
        

        values = range(num_batches)
        if self.rnn:
            hidden = self.model.init_hidden(batch_size)
        else:
            self.model.init_hidden(batch_size)
       
        
        with tqdm(total=len(values), file=sys.stdout) as pbar:
        
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    start = i*batch_size
                    end = start + batch_size

                
                
                    if len(data)>2:
                        inputs, i_max, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        i_max  = i_max.to(self.device)
                    else:
                        inputs,  labels=data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    
                    
                    
                    if self.rnn:
                        out, hidden = self.model(inputs,i_max, hidden)
                    else:
                        out = self.model(inputs, i_max)

                    if type(out) is tuple:
                        pred = self.get_prediction(out[-1])
                    
                    else:
                        pred = self.get_prediction(out)

                    
                    predictions[start:end] = pred.data.long()
                    ground_label[start:end] = labels.long()
                    

                    del inputs
                    del labels
                    del  data
                    pbar.set_description('processed: %d' % (1 + i))
                    pbar.update(1)
                pbar.close()

                
        predictions=predictions.numpy().astype(np.int32)  
        ground_label=ground_label.numpy().astype(np.int32) 
        

        
        assert(num_elements==len(predictions))
        
    
        self.model.train()
        return predictions, ground_label
