import os
import sys
master_folder = os.getcwd().replace("/experiments/evaluate_models", '')
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

from xai_concept_leakage.data.shapes3d_auxiliary import shapes3d_dataloaders
from experiments.experiment_utils import get_shapes3d_extractor_arch
from experiments.evaluate_model import evaluate_CBM
from experiments.fuse_models_folds import model_names_fold_from_classes

if __name__ == '__main__':

    ##################################################################################################
    ### Config:
    ##################################################################################################
    checkpoint_classes = [
        "CBM_" + "Sigmoid_1",
    ]
    n_fold = 5

    results_folder = master_folder + "results/shapes3d_dep_0_models/" 
    save_path = master_folder + "results/results_shapes3d_dep_0.dict" 
    repeats = 5
    rerun = True

    test_dl_label = "test"
    policies = ["random", "coop"]
    ois_repeats = repeats
    intervention_repeats = repeats
    interconcept_repeats = repeats
    concepts_task_repeats = repeats

    list_observables = [
            "accuracies",
            "interventions",
            "leakage_scores"
        ]

    ##################################################################################################
    ### Dataloading:
    ##################################################################################################
    corr = 0
    DATASETS_DIR = "data/shapes3d/"
    path_dataset = DATASETS_DIR + f"dep_{corr}/"

    train_dl, val_dl, test_dl = shapes3d_dataloaders(path_dataset, num_workers = 1, batch_size = 32)
    dl_dict = {"train": train_dl,
            "val": val_dl,
            "test": test_dl}

    x2c_extractor = get_shapes3d_extractor_arch

    ##################################################################################################
    ### Evaluate:
    ##################################################################################################
    checkpoint_names = model_names_fold_from_classes(checkpoint_classes, n_fold = n_fold)
    
    results = evaluate_CBM(checkpoint_names = checkpoint_names,
                            results_folder = results_folder, 
                            x2c_extractor = x2c_extractor, 
                            dl_dict = dl_dict, 
                            train_dl = train_dl, val_dl = val_dl, test_dl = test_dl,
                            list_observables = list_observables,
                            ois_repeats = ois_repeats,
                            policies = policies,
                            intervention_repeats = intervention_repeats, 
                            interconcept_repeats = interconcept_repeats, 
                            concepts_task_repeats = concepts_task_repeats,
                            rerun = rerun, save_path = save_path)
