import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch

from torchvision.models import resnet50

from xai_concept_leakage.models.cbm import ConceptBottleneckModel, compute_accuracy
from xai_concept_leakage.models.cem import ConceptEmbeddingModel
import xai_concept_leakage.train.utils as utils

class IntAwareConceptBottleneckModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,

        intervention_discount=1,
        intervention_task_discount=1.1,
        intervention_weight=5,
        concept_map=None,
        use_concept_groups=True,
        task_loss_weight=0,

        include_only_last_trajectory_loss=True,
        intervention_task_loss_weight=1,
        rollout_init_steps=0,
        int_model_layers=None,
        int_model_use_bn=True,
        top_k_accuracy=None,
        num_rollouts=1,

        # Parameters regarding how we select how many concepts to intervene on
        # in the horizon of a current trajectory (this is the lenght of the
        # trajectory)
        max_horizon=6,
        initial_horizon=2,
        horizon_rate=1.005,
    ):
        assert task_loss_weight == 0, (
            f'IntCBM only supports task_loss_weight=0 as this loss is included '
            f'as part of the trajectory loss. It was given task_loss_weight = '
            f'{task_loss_weight}'
        )
        self.num_rollouts = num_rollouts
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False

        super(IntAwareConceptBottleneckModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            extra_dims=extra_dims,
            bool=bool,
            sigmoidal_prob=sigmoidal_prob,
            sigmoidal_extra_capacity=sigmoidal_extra_capacity,
            bottleneck_nonlinear=bottleneck_nonlinear,
            output_latent=output_latent,
            x2c_model=x2c_model,
            c_extractor_arch=c_extractor_arch,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            use_concept_groups=use_concept_groups,
        )

        units = [
            n_concepts + # Bottleneck
            n_concepts # Prev interventions
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())

        self.concept_rank_model = torch.nn.Sequential(*layers)

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        self.current_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.rollout_init_steps = rollout_init_steps
        self.intervention_weight = intervention_weight
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.max_horizon = max_horizon
        self.emb_size = 1
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight

        self._horizon_distr = lambda init, end: np.random.randint(
            init,
            end,
        )

    def get_concept_int_distribution(
        self,
        x,
        c,
        prev_interventions=None,
        competencies=None,
        horizon=1,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros(c.shape).to(x.device)
        outputs = self._forward(
            x,
            c=c,
            y=None,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )

        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        prev_interventions = outputs[3]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]
        return self._prior_int_distribution(
            prob=c_sem,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            c=c,
            competencies=competencies,
            prev_interventions=prev_interventions,
            horizon=horizon,
            train=False,
        )

    def _prior_int_distribution(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        competencies=None,
        prev_interventions=None,
        horizon=1,
        train=False,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros(prob.shape).to(
                pos_embeddings.device
            )
        if (competencies is None):
            # Then we will always see competencies as perfect if they are not
            # given or if we are asked to explicitly ignore them
            if self.use_concept_groups:
                competencies = torch.ones(
                    (prob.shape[0], len(self.concept_map))
                ).to(pos_embeddings.device)
            else:
                competencies = torch.ones(prob.shape).to(pos_embeddings.device)
        # Shape is [B, n_concepts, emb_size]
        prob = prev_interventions * c + (1 - prev_interventions) * prob
        embeddings = (
            torch.unsqueeze(prob, dim=-1) * pos_embeddings +
            (1 - torch.unsqueeze(prob, dim=-1)) * neg_embeddings
        )
        # Zero out embeddings of previously intervened concepts
        if self.use_concept_groups:
            available_groups = torch.zeros(
                (embeddings.shape[0], len(self.concept_map))
            ).to(
                embeddings.device
            )
            for group_idx, (_, group_concepts) in enumerate(
                self.concept_map.items()
            ):
                available_groups[:, group_idx] = torch.logical_not(torch.any(
                    prev_interventions[:, group_concepts] == 1,
                    dim=-1,
                ))
            max_horizon = len(self.concept_map)
        else:
            available_groups = (1 - prev_interventions).to(embeddings.device)
            max_horizon = self.n_concepts
        used_groups = 1 - available_groups
        cat_inputs = [
            torch.reshape(embeddings, [-1, self.emb_size * self.n_concepts]),
            prev_interventions,
        ]
        rank_input = torch.concat(
            cat_inputs,
            dim=-1,
        )
        next_concept_group_scores = self.concept_rank_model(
            rank_input
        )
        if train:
            return next_concept_group_scores
        next_concept_group_scores = torch.where(
            used_groups == 1,
            torch.ones(used_groups.shape).to(used_groups.device) * (-1000),
            next_concept_group_scores,
        )
        return torch.nn.functional.softmax(
            next_concept_group_scores,
            dim=-1,
        )

    def _concept_update_with_competencies(
        self,
        c,
        competencies,
        assume_mutually_exclusive=False,
    ):
        # Else we assume we are given binary competencies
        correctly_selected = torch.bernoulli(competencies).to(c.device)
        c_updated = (
            c * correctly_selected +
            (1 - c) * (1 - correctly_selected)
        )
        return c_updated

    def _compute_task_loss(self, y, y_pred_logits):
        """
        Computes the task-specific loss for the predicted task labels
        given the ground-truth task labels.
        """
        task_loss = 0.0
        return task_loss

    def _expected_rollout_y_logits(
        self,
        c_pred,
        c,
        c_for_interventions,
        new_int,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs,
        competencies=None,
    ):
        if (competencies is not None):
            weights = [
                (competencies.to(c.device), c),
                (1 - competencies.to(c.device), 1 - c),
            ]
        else:
            weights = [(1, c)]
        expected_rollout_y_logits = 0
        for weight, used_ground_truth_concepts in weights:
            # Here is the one tricky bit: we will make sure that the
            # concepts that we have previously intervened on use the concept
            # according to the provided label by the intervention (these are the
            # ones we are simulating were given by the user WHICH
            # COULD POTENTIALLY BE WRONG!) while the
            # new concept we are currently rolling will use the GROUND TRUTH
            # concept labels known at training time to compute the expectation
            # So: (1) set all the probabilities of already intervened concepts
            #         using the set of concept labels provided for intervention
            probs = (
                c_pred * (1 - intervention_idxs) +
                c_for_interventions * intervention_idxs
            )

            # Then: (2) For the newly intervened concepts, update only their
            #           probabilities using the provided ground truth values
            #           which, on the expectation, will be weighted with weight
            #           `weight
            probs = (
                probs * (1 - new_int) +
                used_ground_truth_concepts * new_int
            )

            # Compute the bottleneck using the mixture of embeddings based
            # on their assigned probabilities
            c_rollout_pred = (
                (
                    pos_embeddings *
                    torch.unsqueeze(probs, dim=-1)
                ) + (
                    neg_embeddings *
                    (1 - torch.unsqueeze(probs, dim=-1))
                )
            )
            # Flatten as the downstream model takes the entire bottleneck
            # as a single vector
            c_rollout_pred = c_rollout_pred.view(
                (-1, self.emb_size * self.n_concepts)
            )

            # Predict the output task logits with the given
            rollout_y_logits = self.c2y_model(c_rollout_pred)

            # And add this to the current expectation
            expected_rollout_y_logits += weight * rollout_y_logits
        return expected_rollout_y_logits

    def get_target_mask(
        self,
        y,
        c,
        c_pred,
        c_for_interventions,
        pos_embeddings,
        neg_embeddings,
        concept_group_scores,
        prev_intervention_idxs,
        competencies=None,
    ):
        # Generate as a label the concept which increases the
        # probability of the correct class the most when
        # intervened on
        target_int_logits = torch.ones(
            concept_group_scores.shape,
        ).to(c.device) * (-np.Inf)
        for target_concept in range(target_int_logits.shape[-1]):
            if self.use_concept_groups:
                new_int = torch.zeros(
                    prev_intervention_idxs.shape
                ).to(prev_intervention_idxs.device)
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_map.items()
                ):
                    if group_idx == target_concept:
                        new_int[:, group_concepts] = 1
                        break
            else:
                new_int = torch.zeros(
                    prev_intervention_idxs.shape
                ).to(prev_intervention_idxs.device)
                new_int[:, target_concept] = 1

            # Make this intervention and lets see how the y logits change
            # on expectation (expectation taken over the competency of the
            # user on the current intervention)
            if competencies is not None:
                if self.use_concept_groups:
                    # [DESIGN DECISION] We will average all the target competencies
                    #                   of the group to get a single group level
                    #                   competency score for the group
                    target_competencies = torch.unsqueeze(
                        torch.mean(
                            competencies[:, target_concept],
                            dim=-1,
                        ),
                        dim=-1,
                    )
                else:
                    target_competencies = torch.unsqueeze(
                        competencies[:, target_concept],
                        dim=-1,
                    )
            else:
                target_competencies = None

            updated_int = torch.clamp(
                prev_intervention_idxs.detach() + new_int,
                0,
                1,
            )
            rollout_y_logits = self._expected_rollout_y_logits(
                intervention_idxs=updated_int,
                new_int=new_int,
                c_pred=c_pred.detach(),
                c_for_interventions=c_for_interventions,
                c=c,
                pos_embeddings=pos_embeddings.detach(),
                neg_embeddings=neg_embeddings.detach(),
                competencies=target_competencies,
            )

            if self.n_tasks > 1:
                one_hot_y = torch.nn.functional.one_hot(y, self.n_tasks)
                target_int_logits[:, target_concept] = \
                    rollout_y_logits[
                        one_hot_y.type(torch.BoolTensor)
                    ]
            else:
                pred_y_prob = torch.sigmoid(
                    torch.squeeze(rollout_y_logits, dim=-1)
                )
                target_int_logits[:, target_concept] = torch.where(
                    y == 1,
                    torch.log(
                        (pred_y_prob + 1e-15) /
                        (1 - pred_y_prob + 1e-15)
                    ),
                    torch.log(
                        (1 - pred_y_prob + 1e-15) /
                        (pred_y_prob+ 1e-15)
                    ),
                )

        target_int_labels = torch.argmax(target_int_logits, -1)
        pred_int_labels = concept_group_scores.argmax(-1)
        curr_acc = (
            pred_int_labels == target_int_labels
        ).float().mean()
        return target_int_labels, curr_acc

    def _setup_intervention_trajectory(
        self,
        prev_num_of_interventions,
        intervention_idxs,
        free_groups,
    ):
        # The limit of how many concepts we can intervene at most
        int_basis_lim = (
            len(self.concept_map) if self.use_concept_groups
            else self.n_concepts
        )
        # And the limit of how many concepts we will intervene at most during
        # this training round
        horizon_lim = int(self.horizon_limit.detach().cpu().numpy()[0])

        # Here we first determine how many concepts we will intervene on in at
        # the begining of the trajectory before we even start talling up
        # losses from this intervention trajectory.
        # We will also sample the length of the trajectory for this training
        # step (current_horizon) as well as the normalization coefficients
        # for the trajectory-dependent losses (task_trajectory_weight and
        # trajectory_weight)
        if prev_num_of_interventions != int_basis_lim:
            bottom = min(
                horizon_lim,
                int_basis_lim - prev_num_of_interventions - 1,
            )  # -1 so that we at least intervene on one concept
            if bottom > 0:
                initially_selected = np.random.randint(0, bottom)
            else:
                initially_selected = 0

            # Get the maximum size of any current trajectories:
            end_horizon = min(
                int(horizon_lim),
                self.max_horizon,
                int_basis_lim - prev_num_of_interventions - initially_selected,
            )

            # And select the number of steps T we will run the current
            # trajectory for
            current_horizon = self._horizon_distr(
                init=1 if end_horizon > 1 else 0,
                end=end_horizon,
            )

            # At the begining of the trajectory, we start with a total of
            # `initially_selected`` concepts already intervened on. So to
            # indicate that, we will update the intervention_idxs matrix
            # accordingly
            for sample_idx in range(intervention_idxs.shape[0]):
                probs = free_groups[sample_idx, :].detach().cpu().numpy()
                probs = probs/np.sum(probs)
                if self.use_concept_groups:
                    selected_groups = set(np.random.choice(
                        int_basis_lim,
                        size=initially_selected,
                        replace=False,
                        p=probs,
                    ))
                    for group_idx, (_, group_concepts) in enumerate(
                        self.concept_map.items()
                    ):
                        if group_idx in selected_groups:
                            intervention_idxs[sample_idx, group_concepts] = 1
                else:
                    intervention_idxs[
                        sample_idx,
                        np.random.choice(
                            int_basis_lim,
                            size=initially_selected,
                            replace=False,
                            p=probs,
                        )
                    ] = 1
            discount = 1
            trajectory_weight = 0
            for i in range(current_horizon):
                trajectory_weight += discount
                discount *= self.intervention_discount
            task_discount = 1
            task_trajectory_weight = 1
            for i in range(current_horizon):
                task_discount *= self.intervention_task_discount
                if (
                    (not self.include_only_last_trajectory_loss) or
                    (i == current_horizon - 1)
                ):
                    task_trajectory_weight += task_discount
            task_discount = 1
        else:
            # Else we will peform no intervention in this training step!
            current_horizon = 0
            task_discount = 1
            task_trajectory_weight = 1
            trajectory_weight = 1

        return (
            current_horizon,
            task_discount,
            task_trajectory_weight,
            trajectory_weight,
        )

    def _intervention_rollout_loss(
        self,
        y,
        c,
        y_pred_logits,
        c_pred,
        pos_embeddings,
        neg_embeddings,
        prev_interventions,
        competencies=None,
    ):
        intervention_task_loss = 0.0
        int_mask_accuracy = -1.0
        current_horizon = -1
        intervention_loss = 0.0

        # First, figure out which concepts/concept groups are free for
        # us to intervene on next
        if prev_interventions is not None:
            # This will be not None in the case of RandInt, so we can ASSUME
            # they have all been intervened the same number of times before
            intervention_idxs = prev_interventions[:]
            if self.use_concept_groups:
                # If we are working with concet groups, then we will operate
                # in the group space rather than at an individual concept's
                # resolution
                free_groups = torch.ones(
                    (prev_interventions.shape[0], len(self.concept_map))
                ).to(
                    c.device
                )
                # ASSUMPTION: THEY HAVE ALL BEEN INTERVENED ON THE SAME
                # NUMBER OF CONCEPTS
                cpu_ints = intervention_idxs[0, :].detach().cpu().numpy()
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_map.items()
                ):
                    if np.any(cpu_ints[group_concepts] == 1):
                        free_groups[:, group_idx] = 0
                prev_num_of_interventions = int(
                    len(self.concept_map) - np.sum(
                        free_groups[0, :].detach().cpu().numpy(),
                        axis=-1,
                    ),
                )
            else:
                # Otherwise, we are working at the concept-resolution level
                free_groups = 1 - intervention_idxs
                prev_num_of_interventions = int(np.max(
                    np.sum(
                        intervention_idxs.detach().cpu().numpy(),
                        axis=-1,
                    ),
                    axis=-1,
                ))
        else:
            # Else, we start from a fresh slate without any previous
            # interventions
            intervention_idxs = torch.zeros(c.shape).to(c.device)
            prev_num_of_interventions = 0
            if self.use_concept_groups:
                free_groups = torch.ones(
                    (c.shape[0], len(self.concept_map))
                ).to(c.device)
            else:
                free_groups = torch.ones(c.shape).to(c.device)

        # Then time to perform a forced training time intervention
        # We will set some of the concepts as definitely intervened on
        if competencies is None:
            # Then we are not given any competencies so nothing to change
            # from the ground truth labels
            c_for_interventions = c
        else:
            # Update the ground truth concept labels to take
            # into account the competency levels of the user intervening!
            c_for_interventions = self._concept_update_with_competencies(
                c=c,
                competencies=competencies,
            )

        # Update the intervention idxs so that we included a small random
        # number of initial interventions (a-la RandInt) at the begining
        # of this trajectory. While we do that, we will also compute the
        # size of the current trajectory T (i.e., the current_horizon) as well
        # as the discounts and weights that come with a trajectory of this
        # size and the initial number of intervened concepts
        (
            current_horizon,
            task_discount,
            task_trajectory_weight,
            trajectory_weight,
        ) = self._setup_intervention_trajectory(
            prev_num_of_interventions,
            intervention_idxs,
            free_groups,
        )
        discount = 1

        # Then we initialize the intervention trajectory task loss to
        # that of the unintervened model as this loss is not going to
        # be taken into account if we don't do this
        intervention_task_loss = self.loss_task(
            (
                y_pred_logits if y_pred_logits.shape[-1] > 1
                else y_pred_logits.reshape(-1)
            ),
            y,
        )
        intervention_task_loss = (
            intervention_task_loss / task_trajectory_weight
        )

        # The mask accuracy is just for logging purposes
        int_mask_accuracy = 0.0 if current_horizon else -1
        if (
            self.current_steps.detach().cpu().numpy()[0] <
            self.rollout_init_steps
        ):
            current_horizon = 0
        # Time to perform as many monte carlo rollouts as requested!
        for _ in range(self.num_rollouts):
            # And as many steps in the trajectory as suggested
            for i in range(current_horizon):
                # And generate a probability distribution over previously
                # unseen concepts to indicate which one we should intervene
                # on next!
                concept_group_scores = self._prior_int_distribution(
                    prob=c_pred,
                    pos_embeddings=pos_embeddings,
                    neg_embeddings=neg_embeddings,
                    competencies=competencies,
                    prev_interventions=intervention_idxs,
                    c=c_for_interventions,
                    horizon=(current_horizon - i),
                    train=True,
                )

                target_int_labels, curr_acc = self.get_target_mask(
                    y=y,
                    c=c,
                    c_for_interventions=c_for_interventions,
                    c_pred=c_pred,
                    pos_embeddings=pos_embeddings,
                    neg_embeddings=neg_embeddings,
                    concept_group_scores=concept_group_scores,
                    prev_intervention_idxs=intervention_idxs,
                    competencies=competencies,
                )
                int_mask_accuracy += curr_acc/current_horizon

                new_loss = self.loss_interventions(
                    concept_group_scores,
                    target_int_labels,
                )
                # Update the next-concept predictor loss
                intervention_loss += (
                    discount * new_loss/trajectory_weight
                )

                # Update the discount (before the task trajectory loss to
                # start discounting from the first intervention so that the
                # loss of the unintervened model is highest
                discount *= self.intervention_discount
                task_discount *= self.intervention_task_discount

                # Sample the next concepts we will intervene on using a hard
                # Gumbel softmax
                if self.intervention_weight == 0:
                    selected_groups = torch.FloatTensor(
                        np.eye(concept_group_scores.shape[-1])[np.random.choice(
                            concept_group_scores.shape[-1],
                            size=concept_group_scores.shape[0]
                        )]
                    ).to(concept_group_scores.device)
                else:
                    selected_groups = torch.nn.functional.gumbel_softmax(
                        concept_group_scores,
                        dim=-1,
                        hard=True,
                        tau=1,
                    )
                if self.use_concept_groups:
                    for sample_idx in range(intervention_idxs.shape[0]):
                        for group_idx, (_, group_concepts) in enumerate(
                            self.concept_map.items()
                        ):
                            if selected_groups[sample_idx, group_idx] == 1:
                                intervention_idxs[
                                    sample_idx,
                                    group_concepts,
                                ] = 1
                else:
                    intervention_idxs += selected_groups

                if (
                    (not self.include_only_last_trajectory_loss) or
                    (i == (current_horizon - 1))
                ):
                    # Then we will also update the task loss with the loss
                    # of performing this intervention!
                    probs = (
                        c_pred * (1 - intervention_idxs) +
                        c * intervention_idxs
                    )
                    c_rollout_pred = (
                        pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                            neg_embeddings * (
                                1 - torch.unsqueeze(probs, dim=-1)
                            )
                        )
                    )
                    c_rollout_pred = c_rollout_pred.view(
                        (-1, self.emb_size * self.n_concepts)
                    )
                    rollout_y_logits = self.c2y_model(c_rollout_pred)
                    rollout_y_loss = self.loss_task(
                        (
                            rollout_y_logits
                            if rollout_y_logits.shape[-1] > 1 else
                            rollout_y_logits.reshape(-1)
                        ),
                        y,
                    )
                    intervention_task_loss += (
                        task_discount *
                        rollout_y_loss / task_trajectory_weight
                    )

            if (
                (
                    self.current_steps.detach().cpu().numpy()[0] >=
                        self.rollout_init_steps
                )
            ) and (
                self.horizon_limit.detach().cpu().numpy()[0] < (
                    (len(self.concept_map) + 1) if self.use_concept_groups
                    else (self.n_concepts + 1)
                )
            ):
                self.horizon_limit *= self.horizon_rate


        intervention_loss = intervention_loss/self.num_rollouts
        intervention_task_loss = intervention_task_loss/self.num_rollouts
        int_mask_accuracy = int_mask_accuracy/self.num_rollouts

        self.current_steps += 1
        return intervention_loss, intervention_task_loss, int_mask_accuracy

    def _compute_concept_loss(self, c, c_pred):
        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            concept_loss = self.loss_concept(c_pred, c)
        else:
            concept_loss = 0.0
        return concept_loss

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        # prev_interventions will contain the RandInt intervention mask if
        # we are running this a train time!
        prev_interventions = outputs[3]
        latent = outputs[4]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]

        # First we compute the task loss!
        task_loss = self._compute_task_loss(
            y=y,
            y_pred_logits=y_logits,
        )

        # Then the rollout and imitation learning losses
        c_used = c

        if train and (intervention_idxs is None):
            (
                intervention_loss,
                intervention_task_loss,
                int_mask_accuracy,
            ) = self._intervention_rollout_loss(
                c=c_used,
                c_pred=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                y=y,
                y_pred_logits=y_logits,
                prev_interventions=prev_interventions,
                competencies=competencies,
            )
        else:
            intervention_loss = 0
            intervention_task_loss = 0
            int_mask_accuracy = 0


        if isinstance(intervention_task_loss, (float, int)):
            intervention_task_loss_scalar = (
                self.intervention_task_loss_weight * intervention_task_loss
            )
        else:
            intervention_task_loss_scalar = (
                self.intervention_task_loss_weight *
                intervention_task_loss.detach()
            )

        if isinstance(intervention_loss, (float, int)):
            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss
        else:
            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss.detach()


        # Finally, compute the concept loss
        concept_loss = self._compute_concept_loss(
            c=c,
            c_pred=c_sem,
        )
        if isinstance(concept_loss, (float, int)):
            concept_loss_scalar = self.concept_loss_weight * concept_loss
        else:
            concept_loss_scalar = \
                self.concept_loss_weight * concept_loss.detach()

        loss = (
            self.concept_loss_weight * concept_loss +
            self.intervention_weight * intervention_loss +
            self.intervention_task_loss_weight * intervention_task_loss
        )

        loss += self._extra_losses(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            c_pred=c_logits,
            y_pred=y_logits,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "mask_accuracy": int_mask_accuracy,
            "concept_loss": concept_loss_scalar,
            "intervention_task_loss": intervention_task_loss_scalar,
            "task_loss": 0, # As the actual task loss is included above!
            "intervention_loss": intervention_loss_scalar,
            "loss": loss.detach() if not isinstance(loss, float) else loss,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
            "horizon_limit": self.horizon_limit.detach().cpu().numpy()[0],
        }
        result["current_steps"] = \
            self.current_steps.detach().cpu().numpy()[0]
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

class IntAwareConceptEmbeddingModel(
    ConceptEmbeddingModel,
    IntAwareConceptBottleneckModel,
):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,

        top_k_accuracy=None,

        intervention_task_discount=1.1,
        intervention_weight=5,
        concept_map=None,
        use_concept_groups=True,

        rollout_init_steps=0,
        int_model_layers=None,
        int_model_use_bn=True,
        num_rollouts=1,

        # Parameters regarding how we select how many concepts to intervene on
        # in the horizon of a current trajectory (this is the lenght of the
        # trajectory)
        max_horizon=6,
        initial_horizon=2,
        horizon_rate=1.005,

        # Experimental/debugging arguments
        intervention_discount=1,
        include_only_last_trajectory_loss=True,
        task_loss_weight=0,
        intervention_task_loss_weight=1,
    ):
        assert task_loss_weight == 0, (
            f'IntCEM only supports task_loss_weight=0 as this loss is included '
            f'as part of the trajectory loss. It was given task_loss_weight = '
            f'{task_loss_weight}'
        )
        self.num_rollouts = num_rollouts
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False

        ConceptEmbeddingModel.__init__(
            self,
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=False,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            use_concept_groups=use_concept_groups,
        )
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map

        # Else we construct it here directly
        units = [
            n_concepts * emb_size + # Bottleneck
            n_concepts # Prev interventions
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        self.concept_rank_model = torch.nn.Sequential(*layers)

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        self.current_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.rollout_init_steps = rollout_init_steps
        self.intervention_weight = intervention_weight
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.max_horizon = max_horizon
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight
        self.use_concept_groups = use_concept_groups
        self._horizon_distr = lambda init, end: np.random.randint(
            init,
            end,
        )

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            if self.use_concept_groups:
                group_mask = np.random.binomial(
                    n=1,
                    p=self.training_intervention_prob,
                    size=len(self.concept_map),
                )
                mask = torch.zeros((c_true.shape[-1],)).to(c_true.device)
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_map.items()
                ):
                    if group_mask[group_idx] == 1:
                        mask[group_concepts] = 1
                intervention_idxs = torch.tile(
                    mask,
                    (c_true.shape[0], 1),
                )
            else:
                mask = torch.bernoulli(
                    self.ones * self.training_intervention_prob,
                )
                intervention_idxs = torch.tile(
                    mask,
                    (c_true.shape[0], 1),
                )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return (
            prob * (1 - intervention_idxs) + intervention_idxs * c_true,
            intervention_idxs,
        )

    def _prior_int_distribution(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        competencies=None,
        prev_interventions=None,
        horizon=1,
        train=False,
    ):
        return IntAwareConceptBottleneckModel._prior_int_distribution(
            self=self,
            prob=prob,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            c=c,
            competencies=competencies,
            prev_interventions=prev_interventions,
            horizon=horizon,
            train=train,
        )