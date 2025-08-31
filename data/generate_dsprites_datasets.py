import os
import random
import numpy as np
import tensorflow as tf
from xai_concept_leakage.data.dsprites_auxiliary import *


def reseed(seed=87):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


reseed(87)

DATASETS_DIR = "data/dsprites/"
dataset_path = DATASETS_DIR + "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
non_linear_y = False

if non_linear_y:
    label_fn = balanced_multiclass_task_label_fn_nonlinear
else:
    label_fn = balanced_multiclass_task_label_fn


print("Generating dSprite dataset with correlation scheme 0.")
_, _ = generate_dsprites_dataset(
    label_fn=label_fn,
    filter_fn=dep_0_filter_fn,
    dataset_path=os.path.join(DATASETS_DIR, "dsprites_dep_0.npz"),
    dsprites_path=dataset_path,
    concept_map_fn=multiclass_binary_concepts_map_fn,
    #     force_reload=True,
)

print("Generating dSprite dataset with correlation scheme 1.")
_, _ = generate_dsprites_dataset(
    label_fn=label_fn,
    filter_fn=dep_1_filter_fn,
    dataset_path=os.path.join(DATASETS_DIR, "dsprites_dep_1.npz"),
    dsprites_path=dataset_path,
    concept_map_fn=multiclass_binary_concepts_map_fn,
    #     force_reload=True,
)

print("Generating dSprite dataset with correlation scheme 2.")
_, _ = generate_dsprites_dataset(
    label_fn=label_fn,
    filter_fn=dep_2_filter_fn,
    dataset_path=os.path.join(DATASETS_DIR, "dsprites_dep_2.npz"),
    dsprites_path=dataset_path,
    concept_map_fn=multiclass_binary_concepts_map_fn,
    #     force_reload=True,
)

print("Generating dSprite dataset with correlation scheme 3.")
_, _ = generate_dsprites_dataset(
    label_fn=label_fn,
    filter_fn=dep_3_filter_fn,
    dataset_path=os.path.join(DATASETS_DIR, "dsprites_dep_3.npz"),
    dsprites_path=dataset_path,
    concept_map_fn=multiclass_binary_concepts_map_fn,
    #     force_reload=True,
)

print("Generating dSprite dataset with correlation scheme 4.")
_, _ = generate_dsprites_dataset(
    label_fn=label_fn,
    filter_fn=dep_4_filter_fn,
    dataset_path=os.path.join(DATASETS_DIR, "dsprites_dep_4.npz"),
    dsprites_path=dataset_path,
    concept_map_fn=multiclass_binary_concepts_map_fn,
    #     force_reload=True,
)
