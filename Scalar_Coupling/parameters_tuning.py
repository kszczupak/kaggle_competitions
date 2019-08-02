import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from time import time
import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
from tqdm import tqdm
import json
import os


#GLOBAL HYPEROPT PARAMETERS
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**11 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM
EVAL_METRIC_LGBM_REG = 'mae' #LightGBM regression metric. Note that 'rmse' is more commonly used 
EVAL_METRIC_LGBM_CLASS = 'auc'#LightGBM classification metric

#OPTIONAL OUTPUT
BEST_SCORE = 0

def main():
    start_time = time()
    best_parameters = dict()
    dir_to_save_params = 'tuned_parametes'
    try:
        os.mkdir(dir_to_save_params)
    except OSError:
        pass

    print("Loading data")
    full_train = pd.read_pickle("train_with_features.zip")
    targets = pd.read_csv("train.csv", usecols=['scalar_coupling_constant'])
    # full_train = pd.read_pickle("../input/feature-generator-distance-cosine/train_set.zip")
    # targets = pd.read_csv("../input/champs-scalar-coupling/train.csv", usecols=['scalar_coupling_constant'])

    groups = sorted(full_train.type.unique())

    for idx, group in enumerate(groups):

        group_idx = get_indexes_for_group(full_train, group)
        group_train = full_train.iloc[group_idx].copy().drop('type', axis=1)
        group_targets = targets.iloc[group_idx]
        
        #####
        group_train = group_train.iloc[:1000]
        group_targets = group_targets.iloc[:1000]
        #####
        print("-" * 20)
        print(f"Tuning parameters for {group} ({idx}/{len(groups)})")
        print(f"# of training samples: {len(group_train)}")
        group_best_parameters = quick_hyperopt(
            group_train, 
            group_targets.scalar_coupling_constant.values,
            num_evals=120
            )

        best_parameters[group] = group_best_parameters

        with open(f'{dir_to_save_params}/{group}.json', 'w') as file:
            json.dump(group_best_parameters, file)
    
    print("Saving all best parameters")
    with open(f'{dir_to_save_params}/best_parameters.pickle', 'wb') as f:
        pickle.dump(best_parameters, f)

    print(f"Done, elapsed time: {time() - start_time}")


def quick_hyperopt(data, labels, num_evals, diagnostic=False):
    
    #==========
    #LightGBM
    #==========
    #clear space
    gc.collect()
    
    integer_params = ['max_depth',
                     'num_leaves',
                      'max_bin',
                     'min_data_in_leaf',
                     'min_data_in_bin']
    
    def objective(space_params):
        
        #cast integer params from float to int
        for param in integer_params:
            space_params[param] = int(space_params[param])
        
        #extract nested conditional parameters
        if space_params['boosting']['boosting'] == 'goss':
            top_rate = space_params['boosting'].get('top_rate')
            other_rate = space_params['boosting'].get('other_rate')
            #0 <= top_rate + other_rate <= 1
            top_rate = max(top_rate, 0)
            top_rate = min(top_rate, 0.5)
            other_rate = max(other_rate, 0)
            other_rate = min(other_rate, 0.5)
            space_params['top_rate'] = top_rate
            space_params['other_rate'] = other_rate
        
        subsample = space_params['boosting'].get('subsample', 1.0)
        space_params['boosting'] = space_params['boosting']['boosting']
        space_params['subsample'] = subsample
        
        #for classification, set stratified=True and metrics=EVAL_METRIC_LGBM_CLASS
        cv_results = lgb.cv(space_params, train, nfold = N_FOLDS, stratified=False,
                            early_stopping_rounds=100, metrics=EVAL_METRIC_LGBM_REG, 
                            seed=42, verbose_eval=False)
        
        best_loss = cv_results['l1-mean'][-1] #'l2-mean' for rmse
        #for classification, comment out the line above and uncomment the line below:
        #best_loss = 1 - cv_results['auc-mean'][-1]
        #if necessary, replace 'auc-mean' with '[your-preferred-metric]-mean'
        return{'loss':best_loss, 'status': STATUS_OK }
    
    train = lgb.Dataset(data, labels)
            
    #integer and string parameters, used with hp.choice()
    boosting_list = [{'boosting': 'gbdt',
                      'subsample': hp.uniform('subsample', 0.5, 1)},
                     {'boosting': 'goss',
                      'subsample': 1.0,
                     'top_rate': hp.uniform('top_rate', 0, 0.5),
                     'other_rate': hp.uniform('other_rate', 0, 0.5)}] #if including 'dart', make sure to set 'n_estimators'
    metric_list = ['MAE', 'RMSE'] 
    #for classification comment out the line above and uncomment the line below
    #metric_list = ['auc'] #modify as required for other classification metrics
    objective_list_reg = ['huber', 'fair']
    if min(labels) >= 0:
        # tweedie and gamma works with only non-negative labels
        objective_list_reg.append('tweedie')
        objective_list_reg.append('gamma')
        
    objective_list_class = ['binary', 'cross_entropy']
    #for classification set objective_list = objective_list_class
    objective_list = objective_list_reg

    space ={'boosting' : hp.choice('boosting', boosting_list),
            'num_leaves' : hp.quniform('num_leaves', 2, LGBM_MAX_LEAVES, 1),
            'max_depth': hp.quniform('max_depth', 2, LGBM_MAX_DEPTH, 1),
            'max_bin': hp.quniform('max_bin', 32, 255, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 256, 1),
            'min_data_in_bin': hp.quniform('min_data_in_bin', 1, 256, 1),
            'min_gain_to_split' : hp.quniform('min_gain_to_split', 0.1, 5, 0.01),
            'lambda_l1' : hp.uniform('lambda_l1', 0, 5),
            'lambda_l2' : hp.uniform('lambda_l2', 0, 5),
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'metric' : hp.choice('metric', metric_list),
            'objective' : hp.choice('objective', objective_list),
            'feature_fraction' : hp.quniform('feature_fraction', 0.5, 1, 0.01),
            'bagging_fraction' : hp.quniform('bagging_fraction', 0.5, 1, 0.01)
        }
    
    #optional: activate GPU for LightGBM
    #follow compilation steps here:
    #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
    #then uncomment lines below:
    #space['device'] = 'gpu'
    #space['gpu_platform_id'] = 0,
    #space['gpu_device_id'] =  0

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=num_evals, 
                trials=trials)
            
    #fmin() will return the index of values chosen from the lists/arrays in 'space'
    #to obtain actual values, index values are used to subset the original lists/arrays
    best['boosting'] = boosting_list[best['boosting']]['boosting']#nested dict, index twice
    best['metric'] = metric_list[best['metric']]
    best['objective'] = objective_list[best['objective']]
            
    #cast floats of integer params to int
    for param in integer_params:
        best[param] = int(best[param])
    
    print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
    if diagnostic:
        return(best, trials)
    else:
        return(best)


def get_indexes_for_group(data_set, group):
    return data_set[data_set.type == group].index


def group_mean_log_mae(y_true, y_pred, groups):
    """
    Metric used in this competition.
    """
    FLOOR = 1e-9
    maes = (y_true-y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, FLOOR))).mean()


if __name__ == '__main__':
    main()