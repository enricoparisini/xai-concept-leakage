import os
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from xai_concept_leakage.data.shapes3d_auxiliary import *


def reseed(seed=87):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


reseed(87)

DATASETS_DIR = "data/shapes3d/"
_FORCE_RERUN = False
_VAL_SIZE, _TEST_SIZE = 0.1, 0.2


import tensorflow_datasets as tfds

shapes3d_train_ds = tfds.load("shapes3d", split="train", shuffle_files=True)


print()
print(shapes3d_train_ds)

for dep, filter_fn in enumerate(
    [
        filter_fn_dep_0,
        filter_fn_dep_1,
        filter_fn_dep_2,
        filter_fn_dep_3,
        filter_fn_dep_4,
        filter_fn_dep_5,
    ]
):
    print(f"Generating shapes3D dataset with correlation scheme {dep}.")
    dataset_path = DATASETS_DIR + f"dep_{dep}/"
    out_path = dataset_path + f"shapes3d_dep_{dep}"
    Path(dataset_path).mkdir(parents=True, exist_ok=True)

    x_train, y_train, c_train = extract_data(
        shapes3d_train_ds,
        filter_fn,
        dataset_path=out_path + ".npz",
        concept_map_fn=multiclass_task_bin_concepts_map_fn,
        label_fn=balanced_multiclass_task_bin_concepts_label_fn,
        force_rerun=_FORCE_RERUN,
    )
    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
        x_train,
        y_train,
        c_train,
        test_size=_TEST_SIZE,
    )
    x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(
        x_train,
        y_train,
        c_train,
        test_size=_VAL_SIZE / (1 - _TEST_SIZE),
    )
    np.save(out_path + "_x_train.npy", x_train)
    np.save(out_path + "_c_train.npy", c_train)
    np.save(out_path + "_y_train.npy", y_train)

    np.save(out_path + "_x_val.npy", x_val)
    np.save(out_path + "_c_val.npy", c_val)
    np.save(out_path + "_y_val.npy", y_val)

    np.save(out_path + "_x_test.npy", x_test)
    np.save(out_path + "_c_test.npy", c_test)
    np.save(out_path + "_y_test.npy", y_test)
