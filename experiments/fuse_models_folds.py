import numpy as np
import joblib
from experiments.experiment_utils import save_joblib, extract_scores_from_results


##################################################################################################
### Manipulating results from n_fold trained models of the same class:
##################################################################################################


def model_names_fold_from_classes(checkpoint_classes, n_fold=5):
    """
    Generate the checkpoint_names of the models in the list check_point_classes, trained n_fold.
    """
    checkpoint_names = []
    for checkpoint_class in checkpoint_classes:
        checkpoint_names += [
            checkpoint_class + "_fold_" + str(i + 1) for i in range(n_fold)
        ]
    return checkpoint_names


def gather_single_results(folder_checkpoint_dict):
    """
    Gather the results from the models in the folder_checkpoint_dict.
    folder_checkpoint_dict is a dictionary where the keys are the folders and the values are
    the list of model class names.
    Return a dictionary where the keys are the checkpoint names and the values are the results
    of each model class.
    """
    results = {}
    for folder, checkpoint_list in folder_checkpoint_dict.items():
        for checkpoint_name in checkpoint_list:
            results[checkpoint_name] = joblib.load(folder + checkpoint_name + ".dict")
    return results


def identity(x):
    return x


def mean_axis0(x):
    return np.mean(x, axis=0)


def fuse_folds(
    checkpoint_class,
    results_folder,
    n_fold=None,
    checkpoint_names=None,
    observables=None,
    dl_labels=None,
    save_path=None,
):
    """
    Fuse the results of the n_fold models for a single model class indicated by "checkpoint_class".
    Parameters:
    - checkpoint_class: str. The class of the models to be fused.
    - results_folder: str. The folder where the results are stored.
    - n_fold: int. The number of folds to fuse for that model class.
        If not provided, fuse the models with names checkpoint_names in the folder.
    - checkpoint_names: list of str. The names of the models to be fused.
        Superceded by n_fold.
    - observables: list of str. The observables to be fused. If not provided,
        fuse all the observables in the maps_dict hard-coded below if present in results.
    - dl_labels: list of str. The labels of the dataloaders to be fused.
        If not provided, fuse all the dataloaders in the results.
    - save_path: str. The path where the results will be saved.
        If not provided, the function will not save the results.

    Typical example for a fusion on the spot:
    checkpoint_class = "CBM_" + "Sigmoid_1"
    results_folder = master_folder + "results/tabulartoy_25_10k_soft/"
    n_fold = 5
    save_path = None
    fusion_dict = fuse_folds(checkpoint_class = checkpoint_class, results_folder = results_folder,
                            n_fold = n_fold, save_path = save_path)
    """
    maps_dict = {
        "y_accuracy": identity,
        "y_balanced_accuracy": identity,
        "c_accuracy": identity,
        "ois": np.mean,
        "nis": np.mean,
        "cas": np.mean,
        "random": mean_axis0,
        "random_adv": mean_axis0,
        "coop": mean_axis0,
        "coop_adv": mean_axis0,
        "ICL_ij_tril": mean_axis0,
        "ICL_i": mean_axis0,
        "ICL": np.mean,
        "CTL_i": mean_axis0,
        "CTL": np.mean,
        "IC_MI_ij_tril_mix": mean_axis0,
        "IC_MI_i_mix": mean_axis0,
        "IC_MI_mix": np.mean,
        "CT_MI_i_mix": mean_axis0,
        "CT_MI_mix": np.mean,
        "IC_MI_ij_tril_pos": mean_axis0,
        "IC_MI_i_pos": mean_axis0,
        "IC_MI_pos": np.mean,
        "CT_MI_i_pos": mean_axis0,
        "CT_MI_pos": np.mean,
        "IC_MI_ij_tril_neg": mean_axis0,
        "IC_MI_i_neg": mean_axis0,
        "IC_MI_neg": np.mean,
        "CT_MI_i_neg": mean_axis0,
        "CT_MI_neg": np.mean,
        "clustering_acc_pos_vs_c": identity,
        "clustering_acc_neg_vs_c": identity,
        "MI_pos_c_gt": mean_axis0,
        "MI_pos_c_gt": mean_axis0,
        "MI_neg_c_gt": mean_axis0,
        "MI_neg_c_gt": mean_axis0,
        "avg_self_MI_pos_c_gt": np.mean,
        "avg_other_MI_pos_c_gt": np.mean,
        "avg_self_MI_neg_c_gt": np.mean,
        "avg_other_MI_neg_c_gt": np.mean,
        "avg_self_MI_mix_c_gt": np.mean,
        "avg_other_MI_mix_c_gt": np.mean,
        "CT_MI_i_pos_aligned": mean_axis0,
        "CT_MI_i_pos_unaligned": mean_axis0,
        "CT_MI_i_neg_aligned": mean_axis0,
        "CT_MI_i_neg_unaligned": mean_axis0,
        "CT_MI_pos_aligned": np.mean,
        "CT_MI_pos_unaligned": np.mean,
        "CT_MI_neg_aligned": np.mean,
        "CT_MI_neg_unaligned": np.mean,
        "CT_MI_i_alignment": mean_axis0,
        "CT_MI_alignment": np.mean,
    }

    if observables is None:
        no_obs = True
    if n_fold is not None:
        checkpoint_names = model_names_fold_from_classes(
            [checkpoint_class], n_fold=n_fold
        )
    folder_checkpoint_dict = {results_folder: checkpoint_names}
    results = gather_single_results(folder_checkpoint_dict)

    fusion_dict = {}
    if dl_labels is None:
        dl_labels = list(results[next(iter(results.keys()))].keys())
    for dl_label in dl_labels:
        dl_fusion_dict = {}
        if no_obs:
            observables = list(results[next(iter(results.keys()))][dl_label].keys())
        for observable in observables:
            if observable in maps_dict:
                temp_dict = extract_scores_from_results(
                    results, test_dl_label=dl_label, score_labels=[observable]
                )
                dl_fusion_dict[observable] = np.array(
                    [
                        maps_dict[observable](rep_scores)
                        for rep_scores in temp_dict.values()
                    ]
                )
        fusion_dict[dl_label] = dl_fusion_dict
    save_joblib(fusion_dict, save_path)
    return fusion_dict
