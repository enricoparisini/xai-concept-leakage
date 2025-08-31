Scripts in the folder `xai_concept_leakage/` were adapted from the `cem` package, in particular:

**Added:**
- TabularToy, dSprites and 3dshapes data generation and loading to `data/`
- `metrics/mutual_information.py`

**Adapted:**
- `interventions/utils.py`
- `models/cem.py`, `construction.py`
- `train/evaluate.py`, `training.py`, `utils.py`

We also adapted `experiments/run_experiments.py`, `experiments_utils.py`.

Most of the documentation and examples from `cem` remain valid and still work.
