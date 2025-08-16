# XAI-Concept-Leakage

This repository contains the code for the paper [Leakage and Interpretability in Concept-Based Models](https://www.arxiv.org/abs/2504.14094).

Concept Bottleneck Models aim to improve interpretability by predicting high-level intermediate concepts, representing a promising approach for deployment in high-risk scenarios. 
However, they are known to suffer from information leakage, whereby models exploit unintended information encoded within the learned concepts.
We introduce an information-theoretic framework to rigorously characterise and quantify leakage, and define two complementary measures: the concepts-task leakage (CTL) and interconcept leakage (ICL) scores. We show that these measures are strongly predictive of model behaviour under interventions and outperform existing alternatives in robustness and reliability. Using this framework, we identify the primary causes of leakage and provide strong evidence that Concept Embedding Models exhibit substantial leakage regardless of the hyperparameter choice. Finally, we propose a set of practical guidelines for designing concept-based models to reduce leakage and ensure interpretability.

This repository includes code adapted from the [cem](https://github.com/mateoespinosa/cem) package and as such, it allows for the training and evaluation of a range of concept-based models including CBMs, CEMs and IntCEMs and variations thereof (see `cem_legacy.md` for the list of adapted scripts from `cem`). The code to compute our new information-theoretic scores for leakage is in `metrics/mutual_information.py`. It uses the KSG estimator for mutual information and extropy, and generalises the estimators from `sklearn/feature_selection/_mutual_info.py` to higher-dimensional objects. Wrapper functions to compute leakage scores and other evaluation metrics are defined in `experiments/evaluate_model.py`.

## Installation

To install and run this package, first clone this repository,
```bash
$ git clone https://github.com/enricoparisini/xai-concept-leakage
```
After moving to the directory with `cd xai-concept-leakage`, install the package by running
```bash
$ python3 -m pip install .
```
Check the installation with
```bash
$ python3 -c "import xai_concept_leakage"
```
You can now safely delete all subfolders in the current directory except for `data/` and `experiments/`.  

A Docker image that mirrors the latest version of this repository is available:
```bash
$ docker pull eparisini/xai-concept-leakage
```

## Examples

### Download and generate the datasets

**TabularToy**: You can generate this synthetic tabular dataset with the script:
```bash
$ python data/generate_tabulartoy_dataset.py 0.25 10000
```
Here, `0.25` sets the correlation parameter and `10000` specifies the number of samples.
The dataset can then be further explored using the notebook `data/TabularToy_generation.ipynb`.


**dSprites and 3dshapes**: The vanilla datasets can be downloaded running 
```bash
$ cd data/ && bash ./download_datasets.sh && cd -
```
from your local folder. To then generate the concept-annotated datasets with different amounts of ground-truth interconcept correlations, run the scripts `data/generate_dsprites_datasets.py` and `data/generate_shapes3d_datasets.py`.



### Model training

Examples of config files to train CBMs and CEMs can be found in `experiments/config/`. You can train models on TabularToy with
```bash
$ python experiments/run_experiments.py -c experiments/configs/tabulartoy.yaml
```
and analogously for dSprites and 3dshapes. At the end of each model training, basic evaluation metrics such as concepts, task and random intervention performance are also computed.




### Evaluation and leakage measures

A more thorough evaluation of both performance and interpretability can be carried out using the scripts in `experiments/evaluate_models/`, such as
```bash
$ python experiments/evaluate_models/evaluate_models_tabulartoy.py
```
The leakage scores and the other evaluation metrics can be assessed and visualised using the functions in `experiments/`. See the examples on TabularToy(0.25) in `Analyse_results_TabularToy.ipynb`.

