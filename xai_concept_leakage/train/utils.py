import joblib
import logging
import multiprocessing
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import sklearn.metrics
import torch
import torch.nn as nn

from pathlib import Path
from torchvision.models import densenet121


def extract_dims(train_dl):
    x_train, _, c_train = next(iter(train_dl))
    y_train = torch.cat([batch[1] for batch in train_dl])
    n_concepts = c_train.shape[1]
    input_dim = x_train.shape[1]
    if len(y_train.shape) == 1:
        n_tasks = len(torch.unique(y_train))
    else:
        n_tasks = y_train.shape[1]
    return input_dim, n_concepts, n_tasks


def compute_task_class_weights(train_dl):
    _, y_train, _ = next(iter(train_dl))
    y_mean = y_train.type(torch.float32).mean(axis=0).item()
    task_class_weights = torch.Tensor([1 / (1 - y_mean), 1 / y_mean])
    task_class_weights /= torch.min(task_class_weights)
    return task_class_weights


def compute_concept_class_weights(train_dl):
    _, _, c_train = next(iter(train_dl))
    c_mean = c_train.type(torch.float32).mean(axis=0)
    concept_class_weights = 1 / c_mean
    concept_class_weights /= torch.min(concept_class_weights)
    return concept_class_weights


def save_train_val_scores_n_losses(save_path_monitoring, cb_loss):
    print("\nSaving scores and losses to " + save_path_monitoring)
    np.save(save_path_monitoring + "_train_losses", cb_loss.train_losses)
    np.save(save_path_monitoring + "_val_losses", cb_loss.val_losses)
    np.save(save_path_monitoring + "_train_y_acc", cb_loss.train_y_accuracies)
    np.save(save_path_monitoring + "_val_y_acc", cb_loss.val_y_accuracies)
    np.save(save_path_monitoring + "_train_c_acc", cb_loss.train_c_accuracies)
    np.save(save_path_monitoring + "_val_c_acc", cb_loss.val_c_accuracies)


def save_train_val_scores_n_losses_indep(
    save_path_monitoring, cb_loss, x2c=False, c2y=False
):
    if x2c:
        print("\nSaving x2c scores and losses to " + save_path_monitoring)
        np.save(save_path_monitoring + "_train_x2c_losses", cb_loss.train_losses)
        np.save(save_path_monitoring + "_val_x2c_losses", cb_loss.val_losses)
        np.save(save_path_monitoring + "_train_c_acc", cb_loss.train_c_accuracies)
        np.save(save_path_monitoring + "_val_c_acc", cb_loss.val_c_accuracies)
    elif c2y:
        print("\nSaving c2y scores and losses to " + save_path_monitoring)
        np.save(save_path_monitoring + "_train_c2y_losses", cb_loss.train_losses)
        np.save(save_path_monitoring + "_val_c2y_losses", cb_loss.val_losses)
        np.save(save_path_monitoring + "_train_y_acc", cb_loss.train_y_accuracies)
        np.save(save_path_monitoring + "_val_y_acc", cb_loss.val_y_accuracies)
    else:
        pass


class LossTracker(Callback):
    def __init__(self, black_box=False):
        super().__init__()
        self.black_box = black_box

        self.train_loss_temp = []
        self.train_y_accuracy_temp = []
        self.train_losses = []
        self.train_y_accuracies = []

        self.val_loss_temp = []
        self.val_y_accuracy_temp = []
        self.val_losses = []
        self.val_y_accuracies = []

        if not self.black_box:
            self.train_c_accuracy_temp = []
            self.val_c_accuracy_temp = []
            self.train_c_accuracies = []
            self.val_c_accuracies = []

    def _avg_of_empty(self, vec):
        if vec == []:
            return 1.0
        else:
            return np.mean(vec)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.train_loss_temp.append(outputs["loss"].item())
        self.train_y_accuracy_temp.append(outputs["log"]["y_accuracy"])
        if not self.black_box:
            self.train_c_accuracy_temp.append(outputs["log"]["c_accuracy"])

    def on_validation_batch_end(
        self, trainer, module, outputs, batch, batch_idx, another_id
    ):
        self.val_loss_temp.append(outputs["val_loss"].item())
        self.val_y_accuracy_temp.append(outputs["val_y_accuracy"])
        if not self.black_box:
            self.val_c_accuracy_temp.append(outputs["val_c_accuracy"])

    def on_train_epoch_end(self, trainer, pl_module):
        #         print("self.train_y_accuracy_temp:")
        #         print(self.train_y_accuracy_temp)
        mean_loss_epoch = self._avg_of_empty(self.train_loss_temp)
        self.train_losses.append(mean_loss_epoch)
        mean_y_accuracy = self._avg_of_empty(self.train_y_accuracy_temp)
        self.train_y_accuracies.append(mean_y_accuracy)
        self.train_loss_temp = []
        self.train_y_accuracy_temp = []
        if not self.black_box:
            #             print("self.train_c_accuracy_temp:")
            #             print(self.train_c_accuracy_temp)
            mean_c_accuracy = self._avg_of_empty(self.train_c_accuracy_temp)
            self.train_c_accuracies.append(mean_c_accuracy)
            self.train_c_accuracy_temp = []

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss_epoch = self._avg_of_empty(self.val_loss_temp)
        self.val_losses.append(mean_loss_epoch)
        mean_y_accuracy = self._avg_of_empty(self.val_y_accuracy_temp)
        self.val_y_accuracies.append(mean_y_accuracy)
        self.val_loss_temp = []
        self.val_y_accuracy_temp = []
        if not self.black_box:
            #             print("self.val_c_accuracy_temp:")
            #             print(self.val_c_accuracy_temp)
            mean_c_accuracy = self._avg_of_empty(self.val_c_accuracy_temp)
            self.val_c_accuracies.append(mean_c_accuracy)
            self.val_c_accuracy_temp = []


################################################################################
## HELPER FUNCTIONS
################################################################################


def _save_result(fun, kwargs, output_filepath):
    result = fun(**kwargs)
    joblib.dump(result, output_filepath)
    return result


def execute_and_save(
    fun,
    kwargs,
    result_dir,
    filename,
    rerun=False,
):
    output_filepath = os.path.join(
        result_dir,
        filename,
    )
    if (not rerun) and os.path.exists(output_filepath):
        return joblib.load(output_filepath)
    context = multiprocessing.get_context("spawn")
    p = context.Process(
        target=_save_result,
        kwargs=dict(
            fun=fun,
            kwargs=kwargs,
            output_filepath=output_filepath,
        ),
    )
    p.start()
    p.join()
    if p.exitcode:
        raise ValueError(f"Subprocess failed!")
    p.kill()
    return joblib.load(output_filepath)


def load_call(
    function,
    keys,
    run_name,
    old_results=None,
    rerun=False,
    kwargs=None,
):
    old_results = old_results or {}
    kwargs = kwargs or {}
    if not isinstance(keys, (tuple, list)):
        keys = [keys]

    outputs = []
    for key in keys:
        if key.endswith("_" + run_name):
            real_key = key[: len(run_name) + 1]
        else:
            real_key = key
        rerun = rerun or (
            os.environ.get(f"RERUN_METRIC_{real_key.upper()}", "0") == "1"
        )
        if real_key in old_results:
            outputs.append(old_results[real_key])
        else:
            rerun = True
            logging.debug(
                f"Restarting run because we could not find {real_key} in "
                f"old results for {run_name}."
            )
            break
    if not rerun:
        return outputs, True

    return function(**kwargs), False


def _to_val(x):
    if len(x) >= 2 and (x[0] == "[") and (x[-1] == "]"):
        return eval(x)
    try:
        return int(x)
    except ValueError:
        # Then this is not an int
        pass

    try:
        return float(x)
    except ValueError:
        # Then this is not an float
        pass

    if x.lower().strip() in ["true"]:
        return True
    if x.lower().strip() in ["false"]:
        return False

    return x


def extend_with_global_params(config, global_params):
    for param_path, value in global_params:
        var_names = list(map(lambda x: x.strip(), param_path.split(".")))
        current_obj = config
        for path_entry in var_names[:-1]:
            if path_entry not in config:
                current_obj[path_entry] = {}
            current_obj = current_obj[path_entry]
        current_obj[var_names[-1]] = _to_val(value)


def compute_bin_accuracy(y_pred, y_true):
    y_probs = y_pred.reshape(-1).cpu().detach()
    y_pred = y_probs > 0.5
    y_true = y_true.reshape(-1).cpu().detach()
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class="ovo",
        )
    except:
        y_auc = 0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    except:
        y_f1 = 0
    return (y_accuracy, y_auc, y_f1)


def compute_accuracy(
    y_pred,
    y_true,
    binary_output=False,
):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1) or binary_output:
        return compute_bin_accuracy(
            y_pred=y_pred,
            y_true=y_true,
        )
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    used_classes = np.unique(y_true.reshape(-1).cpu().detach())
    y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    y_true = y_true.reshape(-1).cpu().detach()
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(
            y_true,
            y_probs,
            multi_class="ovo",
        )
    except:
        y_auc = 0.0
    y_f1 = 0.0
    return (y_accuracy, y_auc, y_f1)


def wrap_pretrained_model(c_extractor_arch, pretrain_model=True):
    def _result_x2c_fun(output_dim):
        try:
            model = c_extractor_arch(pretrained=pretrain_model)
            if output_dim:
                if c_extractor_arch == densenet121:
                    model.classifier = torch.nn.Linear(
                        1024,
                        output_dim,
                    )
                elif hasattr(model, "fc"):
                    model.fc = torch.nn.Linear(512, output_dim)
        except:
            model = c_extractor_arch(
                output_dim=output_dim,
            )
        return model

    return _result_x2c_fun


################################################################################
## HELPER CLASSES
################################################################################


class EmptyEnter(object):
    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return None

    def __exit__(self, *args, **kwargs):
        pass


class ActivationMonitorWrapper:
    def __init__(
        self,
        model,
        trainer,
        activation_freq,
        single_frequency_epochs,
        output_dir,
        test_dl,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.activation_freq = activation_freq
        self.single_frequency_epochs = single_frequency_epochs
        self.output_dir = output_dir
        self.test_dl = test_dl
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.epoch = 0
        self.trainer = trainer
        self.model = model

    @property
    def current_epoch(self):
        return self.trainer.current_epoch

    def fit(self, *args, **kwargs):
        if self.epoch == 0:
            self.dump_activations()
        true_max_epochs = self.trainer.max_epochs
        while self.epoch < true_max_epochs:
            if self.epoch < self.single_frequency_epochs:
                next_size = 1
            else:
                next_size = min(
                    true_max_epochs - self.epoch,
                    self.activation_freq,
                )
            self.trainer.fit_loop.max_epochs = next_size + self.epoch
            self.trainer.fit_loop.current_epoch = self.epoch
            self.trainer.fit(*args, **kwargs)
            self.epoch += next_size
            self.dump_activations()

    def dump_activations(self):
        batch_results = self.trainer.predict(self.model, self.test_dl)
        out_semantics = np.concatenate(
            list(map(lambda x: x[0], batch_results)),
            axis=0,
        )
        out_embs = np.concatenate(
            list(map(lambda x: x[1], batch_results)),
            axis=0,
        )

        out_acts = np.concatenate(
            list(map(lambda x: x[2], batch_results)),
            axis=0,
        )
        np.save(
            os.path.join(
                self.output_dir,
                f"test_embedding_semantics_on_epoch_{self.epoch}.npy",
            ),
            out_semantics,
        )
        np.save(
            os.path.join(
                self.output_dir,
                f"test_embedding_vectors_on_epoch_{self.epoch}.npy",
            ),
            out_embs,
        )
        np.save(
            os.path.join(
                self.output_dir,
                f"test_model_output_on_epoch_{self.epoch}.npy",
            ),
            out_acts,
        )


class WrapperModule(pl.LightningModule):
    def __init__(
        self,
        model,
        n_tasks,
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        optimizer="sgd",
        top_k_accuracy=2,
        binary_output=False,
        weight_loss=None,
        sigmoidal_output=False,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.binary_output = binary_output
        self.model = model
        if self.n_tasks > 1 and (not binary_output):
            self.loss_task = torch.nn.CrossEntropyLoss(weight=weight_loss)
        elif not sigmoidal_output:
            self.loss_task = torch.nn.BCEWithLogitsLoss(weight=weight_loss)
        else:
            self.loss_task = torch.nn.BCELoss(weight=weight_loss)
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        if (not isinstance(top_k_accuracy, list)) and top_k_accuracy:
            top_k_accuracy = [top_k_accuracy]
        self.top_k_accuracy = top_k_accuracy
        if sigmoidal_output:
            self.sig = torch.nn.Sigmoid()
            self.acc_sig = lambda x: x
        else:
            # Then we assume the model already outputs a sigmoidal vector
            self.sig = lambda x: x
            self.acc_sig = torch.nn.Sigmoid() if self.binary_output else lambda x: x

    def forward(self, x):
        return self.sig(self.model(x))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def _run_step(self, batch, batch_idx, train=False):
        x, y = batch
        y_logits = self(x)
        loss = self.loss_task(
            y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
            y,
        )
        # compute accuracy
        (y_accuracy, y_auc, y_f1) = compute_accuracy(
            y_true=y,
            y_pred=self.acc_sig(y_logits),
            binary_output=self.binary_output,
        )

        result = {
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "loss": loss.detach(),
        }
        if (
            (self.top_k_accuracy is not None)
            and (self.n_tasks > 2)
            and (not self.binary_output)
        ):
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                result[f"y_top_{top_k_val}_accuracy"] = y_top_k_accuracy
        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            self.log(name, val, prog_bar=("accuracy" in name))
        return {
            "loss": loss,
            "log": {
                "y_accuracy": result["y_accuracy"],
                "y_auc": result["y_auc"],
                "y_f1": result["y_f1"],
                "loss": result["loss"],
            },
        }

    def validation_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("accuracy" in name))
        return {"val_" + key: val for key, val in result.items()}

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result["loss"]

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
