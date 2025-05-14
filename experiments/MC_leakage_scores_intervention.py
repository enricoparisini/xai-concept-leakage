import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (sem, pearsonr, spearmanr)
from scipy.stats import t as student_t
from experiments.experiment_utils import extract_scores_from_results

##################################################################################################
### Assessing correlation of leakage and y_acc(k interventions):
##################################################################################################

def means_and_SEs_eval_and_leakage(results, eval_score = "random", leakage_score = 'CTL'):
    '''
    Extract the means and standard errors of the eval_score and leakage_score from the results dictionary.
    '''
    eval_scores_dict = extract_scores_from_results(results, score_labels = [eval_score])
    if eval_score in ["random", "coop"]:
        int_bool = True
        eval_scores_final_dict = { model_label: eval_scores[:, eval_scores.shape[-1]-1]  
                            for model_label, eval_scores in eval_scores_dict.items()}
    else: #if eval_score in ["c_accuracy", "y_accuracy", "ois", "ICL", "CTL"]:
        int_bool = False
        eval_scores_final_dict = eval_scores_dict
    leakage_scores_dict = extract_scores_from_results(results, score_labels = [leakage_score])
    
    eval_scores = list(eval_scores_final_dict.values())
    if int_bool:
        eval_means, eval_SE = 1 - np.mean(eval_scores, axis = -1), sem(eval_scores, axis = -1)
    else:
        eval_means, eval_SE = np.mean(eval_scores, axis = -1), sem(eval_scores, axis = -1)
    eval_dist = np.array([eval_means, eval_SE]).T

    leakage_scores = list(leakage_scores_dict.values())
    leakage_means, leakage_SE = np.mean(leakage_scores, axis = -1), sem(leakage_scores, axis = -1)
    leakage_dist = np.array([leakage_means, leakage_SE]).T
    return eval_means, leakage_means, eval_SE, leakage_SE 



def rubin_multiple_imputation_pearson(x_obs, y_obs, sx, sy, M = 10000,        
                                      rng_seed = None):      
    """
    Multiple-imputation test with null hypothesis of no correlation between x and y, 
    using Rubin's rule to combine  within-imputation and between-imputation variances.
    Return:
    - r_bar: mean of the M sample correlations
    - p_value: two-sided MI p-value
    - t_stat: pooled test statistic on the z-scale
    - dof: Barnard-Rubin adjusted degrees of freedom
    """
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    sx    = np.broadcast_to(sx, x_obs.shape).astype(float)
    sy    = np.broadcast_to(sy, y_obs.shape).astype(float)
    n   = x_obs.size
    rng = np.random.default_rng(rng_seed)

    # MC-sample M datasets: 
    r     = np.empty(M)
    for m in range(M):
        x_star = np.maximum(rng.normal(x_obs, sx), 0)
        y_star = np.maximum(rng.normal(y_obs, sy), 0)
        r[m]   = np.corrcoef(x_star, y_star)[0, 1]
    # Fisher z‑transform: 
    z      = np.arctanh(r)
    z_bar  = z.mean()

    # Within‑imputation variance is constant on the z‑scale
    U      = 1.0/(n - 3)
    # Between‑imputations variance
    B      = ((z - z_bar)**2).sum() / (M - 1)  
    # Total variance by Rubin pooling:
    T      = U + (1 + 1/M)*B                           

    # Wald‑type statistic and Barnard–Rubin dof
    t_stat = z_bar / np.sqrt(T)                        
    dof = (M - 1) * (1 + U/((1 + 1/M)*B))**2
    p_value = 2 * student_t.sf(abs(t_stat), dof)
    return z_bar, r.mean(), p_value, t_stat, dof






