from skmultilearn.adapt import MLkNN, BRkNNaClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
import joblib
from net.metrics import compute_metrics, compute_tp_fp_fn
import scipy.sparse as sp
import numpy as np

def sparse2dense(sparse_matrix):
    """ convert a sparse matrix into a dense matrix of 0 or 1.
    """
    assert sp.issparse(sparse_matrix)

    return np.asarray(sparse_matrix.toarray())

class LearnBaseline():
    def __init__(self, model_name="MLKNNbaseline"):
        
        if model_name=="MLKNNbaseline":
            self.model = MLkNN()
        elif model_name =="BRkNNbaseline":
            self.model = BRkNNaClassifier()
        elif model_name =="BRSVCbaseline":
            self.model = BinaryRelevance(classifier = SVC(),require_dense = [False, True])
        else:
            if model_name not in set(["MLKNNbaseline", "BRkNNbaseline", "BRSVCbaseline"]):
                raise ValueError("Specify MLKNNbaseline, BRkNNbaseline, or BRSVCbaseline model name")
        self.model_name = model_name

      
    
    def fit(self, Xtrain, Xtest, ytrain, ytest, checkpoint_path):
       
        best_score=0
        print(f"Train {self.model_name}")
        clf = self.model
        clf.fit(Xtrain,ytrain)
        y_pred = clf.predict(Xtest)
        y_pred_tra = clf.predict(Xtrain)

        y_pred = sparse2dense(y_pred)
        y_pred_tra = sparse2dense(y_pred_tra)
        
        print(compute_tp_fp_fn(ytest, y_pred, axis=0))
        metrics_test = compute_metrics(ytest, y_pred, all_metrics=True, verbose=False)
        #metrics_tra = compute_metrics(ytrain, y_pred_tra, all_metrics=True, verbose=False)
        

        joblib.dump(clf, '{}'.format(checkpoint_path))
        """  
        print("[TRAIN]  {1:.4f} {2:.4f}" \
            .format(metrics_tra['maF1'], metrics_test['maF1']))
        """

        print('\n')
        print('**********************************')
        print('best subACC:  '+str(metrics_test['subACC']))
        print('best JACC:   '+str(metrics_test['JACC']))
        print('best HA:   '+str(metrics_test['HA']))
        print('best ebF1: '+str(metrics_test['ebF1']))
        print('best miF1: '+str(metrics_test['miF1']))
        print('best maF1: '+str(metrics_test['maF1']))
        print('best meanAUC:  '+str(metrics_test['meanAUC']))
        print('best meanAUPR: '+str(metrics_test['meanAUPR']))
        print('best meanFDR: '+str(metrics_test['meanFDR']))
        print('best meanMCC: '+str(metrics_test['meanMCC']))
        print('**********************************')
        

    def load_model(self, checkpoint_path):
        print(f"load saved  {self.model_name} model ")
        return joblib.load('{}'.format(checkpoint_path))

    def get_prediction(self, X_test, checkpoint_path):
        model = self.load_model(checkpoint_path)
        pred  =  model.predict(X_test)
        pred = sparse2dense(pred)
        return pred


   


