import os
import sys

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
from experiments.experiment_utils import get_tabulartoy_extractor_arch
from experiments.evaluate_model import evaluate_CBM
from experiments.fuse_models_folds import model_names_fold_from_classes

if __name__ == "__main__":

    ##################################################################################################
    ### Config:
    ##################################################################################################
    checkpoint_classes = [
        "CBM_" + "Sigmoid_01",
        "CBM_" + "Sigmoid_5",
    ]
    n_fold = 5
    results_folder = master_folder + "results/tabulartoy_25_10k_models/"
    save_path = master_folder + "results/results_tabulartoy_25_10k.dict"
    repeats = 5
    rerun = True

    test_dl_label = "test"
    policies = ["random", "coop"]
    ois_repeats = repeats
    intervention_repeats = repeats
    interconcept_repeats = repeats
    concepts_task_repeats = repeats

    list_observables = ["accuracies", "interventions", "leakage_scores"]

    ##################################################################################################
    ### Dataloading:
    ##################################################################################################
    data_folder = master_folder + "data/TabularToy/"
    cov = 0.25
    n_samples = 10000
    considered_concepts = ["0", "1", "2"]
    save_folder = (
        data_folder
        + "tabulartoy_"
        + str(int(cov * 100))
        + "_"
        + str(int(n_samples // 1000))
        + "k/"
    )

    train_dl, val_dl, test_dl = TT_dataloaders(
        save_folder, considered_concepts=considered_concepts, c_logits=False
    )
    dl_dict = {"train": train_dl, "val": val_dl, "test": test_dl}
    x2c_extractor = get_tabulartoy_extractor_arch

    ##################################################################################################
    ### Evaluate:
    ##################################################################################################
    checkpoint_names = model_names_fold_from_classes(checkpoint_classes, n_fold=n_fold)

    results = evaluate_CBM(
        checkpoint_names=checkpoint_names,
        results_folder=results_folder,
        x2c_extractor=x2c_extractor,
        dl_dict=dl_dict,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        list_observables=list_observables,
        ois_repeats=ois_repeats,
        policies=policies,
        intervention_repeats=intervention_repeats,
        interconcept_repeats=interconcept_repeats,
        concepts_task_repeats=concepts_task_repeats,
        rerun=rerun,
        save_path=save_path,
    )
