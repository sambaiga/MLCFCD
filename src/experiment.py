from .multilabel import Multilabel
from .complexity import ComplexityEvaluator

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


def run_experiment_one(dataset="plaid"):
    optim_params ={"opt_params": {"lr": 1e-3, "betas":(0.9, 0.98), "eps":1e-9},
             "opt_name":"adam",
             "sheduler_name":"ReduceLROnPlateau",
             "softmax":True
            }
    
    for feature in ["vi", "current", "decomposed_current",  "distance", "decomposed_distance"]:
        experiments = get_experiments(dataset, optim_params, feature)
        for model_name, clf in experiments.items():
            clf.partial_fit()
            
    complexity = ComplexityEvaluator()  
    results = {}
    for i,  model in enumerate(["CNN", "MLkNN"]):
        results[model] = complexity.Run(model)         