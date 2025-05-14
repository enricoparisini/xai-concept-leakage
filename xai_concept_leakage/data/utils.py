import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def reseed(seed=87):
    os.environ['PYTHONHASHSEED'] = str(seed)
    #tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_correlations(list_metrics, no_trace = False):
    for metric in list_metrics:
        metric_copy = np.copy(metric)
        if no_trace:
            for i in range(metric.shape[0]):
                metric_copy[i,i] = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(metric_copy, interpolation='nearest')
        fig.colorbar(cax)
    plt.show()
    
def get_model_prm(checkpoint, prm_name):
    return [tensor for key, tensor in checkpoint['state_dict'].items() if prm_name in key][0]

def logits_from_probs(c, _EPS = 1e-30, sum_to_zero = False, log = False, sup = 5.):
    
    if log:
        if type(c) == torch.Tensor:
            out = torch.log(c + _EPS)
        else:
            out = np.log(c + _EPS)
    else:
        if type(c) == torch.Tensor:
            out = sup*torch.greater(c, 0.5).float() - sup*torch.less(c, 0.5).float()
        else:
            out = sup*(c > 0.5).float() - sup*(c < 0.5).float()
         
    if sum_to_zero:
        if type(out) == pd.core.frame.DataFrame:
            out = out.apply(lambda row: row - np.mean(row), axis = 1) 
        else:
            out = out - np.mean(out, axis = 1, keepdims=True)
    return out



##############################################





