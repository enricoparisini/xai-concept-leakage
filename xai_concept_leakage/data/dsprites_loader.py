'''
Adapted from https://github.com/mateoespinosa/concept-quality
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
from xai_concept_leakage.train.utils import extract_dims # compute_concept_class_weights)

def generate_data(
        config,
        root_dir=None,
        seed=42,
        output_dataset_vars=False,
    ):
    path_dataset = config["root_dir"]
    if config.get("considered_concepts"):
        considered_concepts = config.get("considered_concepts")
    else:
        considered_concepts = [0, 1, 2, 3, 4]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    val_ratio = config.get('val_ratio', 0.1)
    train_dl, val_dl, test_dl = dsprites_dataloaders(
        path_dataset = path_dataset, 
        val_ratio =val_ratio,
        num_workers = num_workers,
        batch_size = batch_size, 
        considered_concepts = considered_concepts, 
        c_logits = False)
    
    _, num_concepts, n_tasks = extract_dims(train_dl)
    # imbalance = compute_concept_class_weights(train_dl)
    # concept_group_map = None

    # return (
    #         train_dl,
    #         val_dl,
    #         test_dl,
    #         imbalance,
    #         (num_concepts, n_tasks, concept_group_map),
    #    )
    
    if config.get('weight_loss', False):
        attribute_count = np.zeros((num_concepts,))
        samples_seen = 0
        for _, (_, _, c) in enumerate(train_dl):
            c = c.cpu().detach().numpy()
            attribute_count += np.sum(c, axis=0)
            samples_seen += c.shape[0]
        imbalance = samples_seen/attribute_count #samples_seen / attribute_count - 1
        imbalance = torch.tensor(imbalance/max(imbalance))   #
    else:
        imbalance = None
    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    concept_group_map = dict([(i, [i]) for i in range(num_concepts)])
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (num_concepts, n_tasks, concept_group_map),
    )