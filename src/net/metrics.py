from sklearn import metrics
import math
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
from threading import Lock
from threading import Thread
from collections import OrderedDict

def subset_accuracy(true_targets, predictions, per_sample=False, axis=1):
    
    result = np.all(true_targets == predictions, axis=axis)

    if not per_sample:
        result = np.mean(result)

    return result

def compute_jaccard_score(true_targets, predictions, per_sample=False, average='macro'):
    if per_sample:
        jaccard = metrics.jaccard_score(true_targets, predictions, average=None)
    else:
        if average not in set(['samples', 'macro', 'weighted']):
            raise ValueError("Specify samples or macro")
        jaccard = metrics.jaccard_score(true_targets, predictions, average=average)
    return jaccard

def hamming_loss(true_targets, predictions, per_sample=False, axis=1):

    result = np.mean(np.logical_xor(true_targets, predictions),
                        axis=axis)

    if not per_sample:
        result = np.mean(result)

    return result


def compute_tp_fp_fn(true_targets, predictions, axis=1):
    # axis: axis for instance
    tp = np.sum(true_targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(true_targets) * predictions,
                   axis=axis).astype('float32')
    fn = np.sum(true_targets * np.logical_not(predictions),
                   axis=axis).astype('float32')

    return (tp, fp, fn)


def example_f1_score(true_targets, predictions, per_sample=False, axis=1):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    numerator = 2*tp
    denominator = (np.sum(true_targets,axis=axis).astype('float32') + np.sum(predictions,axis=axis).astype('float32'))

    zeros = np.where(denominator == 0)[0]

    denominator = np.delete(denominator,zeros)
    numerator = np.delete(numerator,zeros)

    example_f1 = numerator/denominator


    if per_sample:
        f1 = example_f1
    else:
        f1 = np.mean(example_f1)

    return f1

def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    elif average == 'macro':

        def safe_div(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
            return c[np.isfinite(c)]

        f1 = np.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1

def f1_score(true_targets, predictions, average='micro', axis=1):
    """
        average: str
            'micro' or 'macro'
        axis: 0 or 1
            label axis
    """
    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)

    return f1


def compute_aupr_thread(all_targets,all_predictions):
    
    aupr_array = []
    lock = Lock()

    def compute_aupr_(start,end,all_targets,all_predictions):
        for i in range(all_targets.shape[1]):
            try:
                precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
                auPR = metrics.auc(recall,precision)
                lock.acquire() 
                aupr_array.append(np.nan_to_num(auPR))
                lock.release()
            except Exception: 
                pass
                 
    t1 = Thread(target=compute_aupr_, args=(0,100,all_targets,all_predictions) )
    t2 = Thread(target=compute_aupr_, args=(100,200,all_targets,all_predictions) )
    t3 = Thread(target=compute_aupr_, args=(200,300,all_targets,all_predictions) )
    t4 = Thread(target=compute_aupr_, args=(300,400,all_targets,all_predictions) )
    t5 = Thread(target=compute_aupr_, args=(400,500,all_targets,all_predictions) )
    t6 = Thread(target=compute_aupr_, args=(500,600,all_targets,all_predictions) )
    t7 = Thread(target=compute_aupr_, args=(600,700,all_targets,all_predictions) )
    t8 = Thread(target=compute_aupr_, args=(700,800,all_targets,all_predictions) )
    t9 = Thread(target=compute_aupr_, args=(800,900,all_targets,all_predictions) )
    t10 = Thread(target=compute_aupr_, args=(900,919,all_targets,all_predictions) )
    t1.start();t2.start();t3.start();t4.start();t5.start();t6.start();t7.start();t8.start();t9.start();t10.start()
    t1.join();t2.join();t3.join();t4.join();t5.join();t6.join();t7.join();t8.join();t9.join();t10.join()
    

    aupr_array = np.array(aupr_array)

    mean_aupr = np.mean(aupr_array)
    median_aupr = np.median(aupr_array)
    return mean_aupr,median_aupr,aupr_array

def compute_fdr(all_targets,all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i],pos_label=1)
            fdr = 1- precision
            cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not math.isnan(fdr_at_cutoff):
                fdr_array.append(np.nan_to_num(fdr_at_cutoff))
        except: 
            pass
    
    fdr_array = np.array(fdr_array)
    mean_fdr = np.mean(fdr_array)
    median_fdr = np.median(fdr_array)
    var_fdr = np.var(fdr_array)
    return mean_fdr,median_fdr,var_fdr,fdr_array


def compute_aupr(all_targets,all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = metrics.precision_recall_curve(all_targets[:,i], all_predictions[:,i], pos_label=1)
            auPR = metrics.auc(recall,precision)
            if not math.isnan(auPR):
                aupr_array.append(np.nan_to_num(auPR))
        except: 
            pass
    
    aupr_array = np.array(aupr_array)
    mean_aupr = np.mean(aupr_array)
    median_aupr = np.median(aupr_array)
    var_aupr = np.var(aupr_array)
    return mean_aupr,median_aupr,var_aupr,aupr_array


def compute_auc(all_targets,all_predictions):
    auc_array = []
    lock = Lock()

    for i in range(all_targets.shape[1]):
        try:  
            auROC = metrics.roc_auc_score(all_targets[:,i], all_predictions[:,i])
            auc_array.append(auROC)
        except ValueError:
            pass
    
    auc_array = np.array(auc_array)
    mean_auc = np.mean(auc_array)
    median_auc = np.median(auc_array)
    var_auc = np.var(auc_array)
    return mean_auc,median_auc,var_auc,auc_array

def compute_mcc(all_targets,all_predictions):
    mcc_array = []
    for i in range(all_targets.shape[1]):
        try:
            mcc = metrics.matthews_corrcoef(all_targets[:,i], all_predictions[:,i])
            if not math.isnan(mcc):
                mcc_array.append(np.nan_to_num(mcc))
        except: 
            pass
    
    mcc_array = np.array(mcc_array)
    mean_mcc = np.mean(mcc_array)
    median_mcc = np.median(mcc_array)
    var_mcc = np.var(mcc_array)
    return mean_mcc,median_mcc,var_mcc,mcc_array


def compute_metrics(all_predictions,all_targets,all_metrics=True,verbose=False):
    
    if all_metrics:
        meanAUC,medianAUC,varAUC,allAUC = compute_auc(all_targets,all_predictions)
        meanAUPR,medianAUPR,varAUPR,allAUPR = compute_aupr(all_targets,all_predictions)
        meanFDR,medianFDR,varFDR,allFDR = compute_fdr(all_targets,all_predictions)
        meanMCC, medianMCC, varMCC, allMCC = compute_mcc(all_targets,all_predictions)
    else:
        meanAUC,medianAUC,varAUC,allAUC = 0,0,0,0
        meanAUPR,medianAUPR,varAUPR,allAUPR = 0,0,0,0
        meanFDR,medianFDR,varFDR,allFDR = 0,0,0,
        
    acc_ = list(subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions, axis=1, per_sample=True))
    exjacc_ = list(compute_jaccard_score(all_targets, all_predictions, per_sample=True))
    jacc = round(compute_jaccard_score(all_targets, all_predictions, average='macro'), 4)       
    acc = round(np.mean(acc_), 4)
    hl = round(np.mean(hl_), 4)
    exf1 = round(np.mean(exf1_), 4)
    

    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)
    mif1 = round(f1_score_from_stats(tp, fp, fn, average='micro'),4)
    maf1 = round(f1_score_from_stats(tp, fp, fn, average='macro'),4)
    
    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', mif1),
                        ('Label-based Macro F1', maf1)])

    
    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    miF1 = eval_ret['Label-based Micro F1']
    maF1 = eval_ret['Label-based Macro F1']
    if verbose:
        print('ACC:   '+str(ACC))
        print('HA:    '+str(HA))
        print('ebF1:  '+str(ebF1))
        print('miF1:  '+str(miF1))
        print('maF1:  '+str(maF1))
        
        
    if verbose:
        print('uAUC:  '+str(meanAUC))
        # print('mAUC:  '+str(medianAUC))
        print('uAUPR: '+str(meanAUPR))
        # print('mAUPR: '+str(medianAUPR))
        print('uFDR: '+str(meanFDR))
        # print('mFDR:  '+str(medianFDR))

    metrics_dict = {}
    metrics_dict['subACC'] = ACC
    metrics_dict['JACC'] = jacc
    #metrics_dict['ebJACC'] = exjacc_
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['miF1'] = miF1
    metrics_dict['maF1'] = maF1
    metrics_dict['meanAUC'] = meanAUC
    metrics_dict['medianAUC'] = medianAUC
    metrics_dict['meanAUPR'] = meanAUPR
    #metrics_dict['allAUC'] = allAUC
    metrics_dict['medianAUPR'] = medianAUPR
    #metrics_dict['allAUPR'] = allAUPR
    metrics_dict['meanFDR'] = meanFDR
    metrics_dict['medianFDR'] = medianFDR
    metrics_dict['meanMCC'] = meanMCC
    metrics_dict['medianMCC'] = medianMCC
    

    return metrics_dict



class fit_metrics():

    @staticmethod
    def statistics(pred, y):
        pred = pred.cpu().numpy()
        y   = y.cpu().numpy()
        statistics_list = []
        tp, fp, fn = compute_tp_fp_fn(y, pred, axis=0)
           
        statistics_list.append({'TP': tp, 'FP': fp,  'FN': fn})
        return statistics_list
    
    @staticmethod
    def calc_f1_score(statistics_list):
        f1_score_list = []

        for i in range(len(statistics_list)):
            tp = statistics_list[i]['TP']
            fp = statistics_list[i]['FP']
            fn = statistics_list[i]['FN']
            
            mif1 = f1_score_from_stats(tp, fp, fn, average='micro')
            maf1 = f1_score_from_stats(tp, fp, fn, average='macro')
            f1_score_list.append(maf1)
        mean_f1_score = sum(f1_score_list) / len(f1_score_list)

        return mean_f1_score, f1_score_list
    
    @staticmethod
    def update_statistics_list(old_list, new_list):
        if not old_list:
            return new_list

        assert len(old_list) == len(new_list)

        for i in range(len(old_list)):
            old_list[i]['TP'] += new_list[i]['TP']
            old_list[i]['FP'] += new_list[i]['FP']
            old_list[i]['FN'] += new_list[i]['FN']

        return old_list




