"""
Adapted from https://github.com/mateoespinosa/concept-quality
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_datasets as tfds

from xai_concept_leakage.data.utils import logits_from_probs
import xai_concept_leakage.data.shapes3D as shapes3d
import xai_concept_leakage.data.latentFactorData as latentFactorData


#################################################################
# Extracting data from tensorflow_datasets object:
#################################################################
def relabel_categorical(x):
    if len(x.shape) == 1:
        _, unique_inds = np.unique(
            x,
            return_inverse=True,
        )
        return unique_inds

    result = x[:, :]
    for i in range(x.shape[-1]):
        _, unique_inds = np.unique(
            x[:, i],
            return_inverse=True,
        )
        result[:, i] = unique_inds
    return result


def cardinality_encoding(card_group_1, card_group_2):
    result_to_encoding = {}
    for i in card_group_1:
        for j in card_group_2:
            result_to_encoding[(i, j)] = len(result_to_encoding)
    return result_to_encoding


def extract_data(
    shapes3d_train_ds,
    filter_fn,
    sample_map_fn=lambda x: x,
    concept_map_fn=lambda x: x,
    step=1,
    label_fn=lambda ex: ex["label_shape"] * 8 + ex["label_scale"],
    dataset_path=None,
    force_rerun=False,
):
    if (not force_rerun) and dataset_path and os.path.exists(dataset_path):
        # Them time to load up this dataset!
        ds = np.load(dataset_path)
        return ds["X"], ds["y"], ds["c"]
    num_entries = len(shapes3d_train_ds)
    x_train = []
    y_train = []
    c_train = []
    for i, ex in enumerate(tfds.as_numpy(shapes3d_train_ds.shuffle(buffer_size=15000))):
        if i % step != 0:
            continue
        concepts = [
            ex["label_floor_hue"],
            ex["label_wall_hue"],
            ex["label_object_hue"],
            ex["label_scale"],
            ex["label_shape"],
            ex["label_orientation"],
        ]
        if not filter_fn(concepts):
            continue
        print(i, end="\r")

        x_train.append(sample_map_fn(ex["image"]))
        y_train.append(label_fn(ex))
        c_train.append(concept_map_fn(concepts))
    x_train = np.stack(x_train, axis=0) / 255.0
    y_train = relabel_categorical(np.stack(y_train, axis=0))
    c_train = relabel_categorical(np.stack(c_train, axis=0))
    if dataset_path:
        # Then serialize it to speed up things next time
        np.savez(
            dataset_path,
            X=x_train,
            y=y_train,
            c=c_train,
        )
    return x_train, y_train, c_train


############################################################################
## Construct a binary concept task in the shapes3D dataset
############################################################################
def balanced_multiclass_task_bin_concepts_label_fn(concept_dict):
    concept_vector = multiclass_task_bin_concepts_map_fn(
        [
            concept_dict["label_floor_hue"],
            concept_dict["label_wall_hue"],
            concept_dict["label_object_hue"],
            concept_dict["label_scale"],
            concept_dict["label_shape"],
            concept_dict["label_orientation"],
        ]
    )
    if concept_vector[4] == 0:
        offset = 0
        binary_label_encoding = [
            concept_vector[0],
            concept_vector[1],
        ]
    else:
        offset = 4
        binary_label_encoding = [
            concept_vector[2],
            concept_vector[3],
            concept_vector[5],
        ]

    return offset + int("".join(list(map(str, binary_label_encoding))), 2)


def balanced_multiclass_task_bin_concepts_label_fn_equivalent(concept_dict):
    concept_vector = multiclass_task_bin_concepts_map_fn(
        [
            concept_dict["label_floor_hue"],
            concept_dict["label_wall_hue"],
            concept_dict["label_object_hue"],
            concept_dict["label_scale"],
            concept_dict["label_shape"],
            concept_dict["label_orientation"],
        ]
    )
    return (1 - concept_vector[4]) * (
        2 * concept_vector[0] + concept_vector[1]
    ) + concept_vector[4] * (
        4 * concept_vector[2] + 2 * concept_vector[3] + concept_vector[5] + 4
    )


def balanced_multiclass_task_bin_concepts_label_fn_nonlinear(concept_dict):
    concept_vector = multiclass_task_bin_concepts_map_fn(
        [
            concept_dict["label_floor_hue"],
            concept_dict["label_wall_hue"],
            concept_dict["label_object_hue"],
            concept_dict["label_scale"],
            concept_dict["label_shape"],
            concept_dict["label_orientation"],
        ]
    )
    return (
        (1 - concept_vector[4]) * (2 * concept_vector[0] + concept_vector[1])
        + concept_vector[4]
        * (4 * concept_vector[2] + 2 * concept_vector[3] + concept_vector[5] + 4)
        - 3 * concept_vector[0] * concept_vector[1] * concept_vector[2]
        - concept_vector[3] * concept_vector[4] * concept_vector[5]
        - concept_vector[0] * concept_vector[2] * concept_vector[4]
        + concept_vector[1] * concept_vector[3] * concept_vector[5]
    )


def count_class_balance(y):
    one_hot = tf.keras.utils.to_categorical(y)
    return np.sum(one_hot, axis=0) / one_hot.shape[0]


def multiclass_task_bin_concepts_map_fn(concepts):
    return [
        int(concepts[0] < 5),
        int(concepts[1] < 5),
        int(concepts[2] < 5),
        int(concepts[3] < 4),
        int(concepts[4] < 2),
        int(concepts[5] < 7),
    ]


def multiclass_task_bin_concepts_label_fn(concept_dict):
    concept_vector = multiclass_task_bin_concepts_map_fn(
        [
            concept_dict["label_floor_hue"],
            concept_dict["label_wall_hue"],
            concept_dict["label_object_hue"],
            concept_dict["label_scale"],
            concept_dict["label_shape"],
            concept_dict["label_orientation"],
        ]
    )
    binary_label_encoding = [
        concept_vector[0] or concept_vector[1],
        concept_vector[2] or concept_vector[3],
        concept_vector[4] or concept_vector[5],
    ]
    return int("".join(list(map(str, binary_label_encoding))), 2)


def filter_fn_dep_0(concept):
    ranges = [
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 8)),
        list(range(4)),
        list(range(0, 15, 4)),
    ]

    concept_vector = multiclass_task_bin_concepts_map_fn(concept)
    # First filter as in small dataset to constraint the size of the data a bit
    return all([(concept[i] in ranges[i]) for i in range(len(ranges))])


floor_hue_wall_hue_sets_lower = [list(np.random.permutation(7))[:5] for _ in range(10)]

floor_hue_wall_hue_sets_upper = [
    list(3 + np.random.permutation(7))[:5] for _ in range(10)
]


def filter_fn_dep_1(concept):
    ranges = [
        list(range(0, 10)),
        list(range(0, 10, 2)),
        list(range(0, 10, 2)),
        list(range(0, 8)),
        list(range(4)),
        list(range(0, 15, 4)),
    ]

    concept_vector = multiclass_task_bin_concepts_map_fn(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False
    if concept[0] < 5:
        return concept[1] in floor_hue_wall_hue_sets_lower[concept[0]]
    else:
        return concept[1] in floor_hue_wall_hue_sets_upper[concept[0]]


wall_hue_object_hue_sets_lower = [list(np.random.permutation(7))[:5] for _ in range(10)]
wall_hue_object_hue_sets_upper = [
    list(3 + np.random.permutation(7))[:5] for _ in range(10)
]


def filter_fn_dep_2(concept):
    ranges = [
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 10, 2)),
        list(range(0, 8)),
        list(range(4)),
        list(range(0, 15, 4)),
    ]

    concept_vector = multiclass_task_bin_concepts_map_fn(concept)
    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False

    if concept[0] < 5:
        if concept[1] not in floor_hue_wall_hue_sets_lower[concept[0]]:
            return False
    else:
        if concept[1] not in floor_hue_wall_hue_sets_upper[concept[0]]:
            return False

    if concept[1] < 5:
        if concept[2] not in wall_hue_object_hue_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in wall_hue_object_hue_sets_upper[concept[1]]:
            return False
    return True


object_hue_scale_sets_lower = [list(np.random.permutation(6))[:4] for _ in range(10)]

object_hue_scale_sets_upper = [
    list(2 + np.random.permutation(6))[:4] for _ in range(10)
]


def filter_fn_dep_3(concept):
    ranges = [
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 8)),
        list(range(4)),
        list(range(0, 15, 4)),
    ]

    concept_vector = multiclass_task_bin_concepts_map_fn(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False

    if concept[0] < 5:
        if concept[1] not in floor_hue_wall_hue_sets_lower[concept[0]]:
            return False
    else:
        if concept[1] not in floor_hue_wall_hue_sets_upper[concept[0]]:
            return False

    if concept[1] < 5:
        if concept[2] not in wall_hue_object_hue_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in wall_hue_object_hue_sets_upper[concept[1]]:
            return False

    if concept[2] < 5:
        if concept[3] not in object_hue_scale_sets_lower[concept[2]]:
            return False
    else:
        if concept[3] not in object_hue_scale_sets_upper[concept[2]]:
            return False

    return True


scale_shape_sets_lower = [list(np.random.permutation(3))[:2] for _ in range(8)]

scale_shape_sets_upper = [list(1 + np.random.permutation(3))[:2] for _ in range(8)]


def filter_fn_dep_4(concept):
    ranges = [
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 8)),
        list(range(4)),
        list(range(0, 15, 2)),
    ]

    concept_vector = multiclass_task_bin_concepts_map_fn(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False

    if concept[0] < 5:
        if concept[1] not in floor_hue_wall_hue_sets_lower[concept[0]]:
            return False
    else:
        if concept[1] not in floor_hue_wall_hue_sets_upper[concept[0]]:
            return False

    if concept[1] < 5:
        if concept[2] not in wall_hue_object_hue_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in wall_hue_object_hue_sets_upper[concept[1]]:
            return False

    if concept[2] < 5:
        if concept[3] not in object_hue_scale_sets_lower[concept[2]]:
            return False
    else:
        if concept[3] not in object_hue_scale_sets_upper[concept[2]]:
            return False

    if concept[3] < 4:
        if concept[4] not in scale_shape_sets_lower[concept[3]]:
            return False
    else:
        if concept[4] not in scale_shape_sets_upper[concept[3]]:
            return False
    return True


shape_rotation_sets_lower = [list(np.random.permutation(9))[:7] for _ in range(4)]

shape_rotation_sets_upper = [list(6 + np.random.permutation(9))[:7] for _ in range(4)]


def filter_fn_dep_5(concept):
    ranges = [
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 10)),
        list(range(0, 8)),
        list(range(4)),
        list(range(0, 15)),
    ]

    concept_vector = multiclass_task_bin_concepts_map_fn(concept)

    # First filter as in small dataset to constraint the size of the data a bit
    if not all([(concept[i] in ranges[i]) for i in range(len(ranges))]):
        return False

    if concept[0] < 5:
        if concept[1] not in floor_hue_wall_hue_sets_lower[concept[0]]:
            return False
    else:
        if concept[1] not in floor_hue_wall_hue_sets_upper[concept[0]]:
            return False

    if concept[1] < 5:
        if concept[2] not in wall_hue_object_hue_sets_lower[concept[1]]:
            return False
    else:
        if concept[2] not in wall_hue_object_hue_sets_upper[concept[1]]:
            return False

    if concept[2] < 5:
        if concept[3] not in object_hue_scale_sets_lower[concept[2]]:
            return False
    else:
        if concept[3] not in object_hue_scale_sets_upper[concept[2]]:
            return False

    if concept[3] < 4:
        if concept[4] not in scale_shape_sets_lower[concept[3]]:
            return False
    else:
        if concept[4] not in scale_shape_sets_upper[concept[3]]:
            return False

    if concept[4] < 2:
        if concept[5] not in shape_rotation_sets_lower[concept[4]]:
            return False
    else:
        if concept[5] not in shape_rotation_sets_upper[concept[4]]:
            return False
    return True


#################################################################
# Define binary task:
#################################################################


def balanced_multiclass_task_bin_concepts_label_fn(concept_dict):
    concept_vector = multiclass_task_bin_concepts_map_fn(
        [
            concept_dict["label_floor_hue"],
            concept_dict["label_wall_hue"],
            concept_dict["label_object_hue"],
            concept_dict["label_scale"],
            concept_dict["label_shape"],
            concept_dict["label_orientation"],
        ]
    )
    if concept_vector[4] == 0:
        offset = 0
        binary_label_encoding = [
            concept_vector[0],
            concept_vector[1],
        ]
    else:
        offset = 4
        binary_label_encoding = [
            concept_vector[2],
            concept_vector[3],
            concept_vector[5],
        ]

    return offset + int("".join(list(map(str, binary_label_encoding))), 2)


# #################################################################
# # Create dataloaders:
# #################################################################


def shapes3d_dataloaders(
    path_dataset,
    num_workers=1,
    batch_size=32,
    considered_concepts=[0, 1, 2, 3, 4, 5],
    c_logits=False,
):
    """
    path_dataset: folder of the dataset with specified dependence, ending with /.
    TODO: Add considered_concepts
    """
    dep = path_dataset[-2]
    x_train = np.load(path_dataset + f"shapes3d_dep_{dep}_x_train.npy")
    c_train = np.load(path_dataset + f"shapes3d_dep_{dep}_c_train.npy")
    y_train = np.load(path_dataset + f"shapes3d_dep_{dep}_y_train.npy")
    x_val = np.load(path_dataset + f"shapes3d_dep_{dep}_x_val.npy")
    c_val = np.load(path_dataset + f"shapes3d_dep_{dep}_c_val.npy")
    y_val = np.load(path_dataset + f"shapes3d_dep_{dep}_y_val.npy")
    x_test = np.load(path_dataset + f"shapes3d_dep_{dep}_x_test.npy")
    c_test = np.load(path_dataset + f"shapes3d_dep_{dep}_c_test.npy")
    y_test = np.load(path_dataset + f"shapes3d_dep_{dep}_y_test.npy")
    if c_logits:
        c_train, c_test = [logits_from_probs(c) for c in [c_train, c_test]]

    x = torch.FloatTensor(x_train)
    x = torch.swapaxes(torch.swapaxes(x, -1, -2), -2, -3)
    c = torch.FloatTensor(c_train)[:, considered_concepts]
    y = torch.LongTensor(y_train)
    train_dl = torch.utils.data.DataLoader(
        list(zip(x, y, c)), batch_size=batch_size, num_workers=num_workers
    )

    x = torch.FloatTensor(x_val)
    x = torch.swapaxes(torch.swapaxes(x, -1, -2), -2, -3)
    c = torch.FloatTensor(c_val)[:, considered_concepts]
    y = torch.LongTensor(y_val)
    val_dl = torch.utils.data.DataLoader(
        list(zip(x, y, c)), batch_size=batch_size, num_workers=num_workers
    )

    x = torch.FloatTensor(x_test)
    x = torch.swapaxes(torch.swapaxes(x, -1, -2), -2, -3)
    c = torch.FloatTensor(c_test)[:, considered_concepts]
    y = torch.LongTensor(y_test)
    test_dl = torch.utils.data.DataLoader(
        list(zip(x, y, c)), batch_size=batch_size, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl
