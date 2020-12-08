import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim import Adam

from dvrl.training.models import RLDataValueEstimator, DVRLPredictionModel, GumbelDataValueEstimator


class DVRL(pl.LightningModule):
    def __init__(self, hparams, dve_model: RLDataValueEstimator, prediction_model: DVRLPredictionModel, val_dataloader,
                 test_dataloader,
                 val_split):
        """
        Implements the DVRL framework.
        :param hparams: this should be a dict, NameSpace or OmegaConf object that implements hyperparameter-storage
        :param prediction_model: this is the core predictor model, this is passed separately since DVRL is agnostic
        to the prediction model.

        ** Note **: In this iteration, the prediction model is constrained to be a torch module and hence shallow models
         won't work
        """
        super().__init__()
        # saving hparams is deprecated, if does not work, follow this -
        # https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html#lightningmodule-hyperparameters

        self.hparams = hparams
        self.dve = dve_model
        self.prediction_model = prediction_model
        self.validation_dataloader = val_dataloader
        self.baseline_delta = 0.0
        self.val_split = val_split
        self.exploration_threshold = self.hparams.exploration_threshold

        self.val_model = copy.deepcopy(self.prediction_model)
        trainer = Trainer(gpus=1, max_epochs=10, callbacks=[EarlyStopping(monitor='loss')])
        trainer.fit(model=self.val_model, train_dataloader=val_dataloader)
        self.val_model.eval()
        self.val_model.requires_grad_(False)
        self.dve.set_val_model(self.val_model)
        self.validation_performance = None

        self.init_test_dataloader = test_dataloader
        self.test_acc = pl.metrics.Accuracy(compute_on_step=False)

    def configure_optimizers(self):
        return Adam(self.dve.parameters(), lr=self.hparams.dve_lr)

    def forward(self, x, y):
        """
        Calls the data value estimator after encoding it with the encoder
        :param x: features
        :param y: labels
        :return: value of given data
        """
        # inference should just call forward pass on the model
        return self.dve(x, y)

    def on_train_start(self) -> None:
        ori_model = copy.deepcopy(self.prediction_model)
        trainer = Trainer(gpus=1, max_epochs=100)
        trainer.fit(model=ori_model, train_dataloader=self.train_dataloader(),
                    val_dataloaders=self.validation_dataloader)
        trainer.test(ori_model, test_dataloaders=self.init_test_dataloader)
        self.validation_performance = ori_model.valid_acc.compute()

    def training_step(self, batch, batch_idx):
        is_corrupted = None
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, is_corrupted = batch

        estimated_dv = torch.sigmoid(self(x, y)).squeeze()

        selection_vector = torch.bernoulli(estimated_dv).detach()

        if selection_vector.sum() == 0:
            # exception when selection probability is 0
            estimated_dv_ = 0.5 * torch.ones_like(estimated_dv)
            selection_vector = torch.bernoulli(estimated_dv_).detach()

        # calling detach here since we don't want to track gradients of ops in prediction model wrt to dve
        training_accuracy = self.prediction_model.dvrl_fit(x, y, selection_vector)

        log_prob = torch.sum(
            selection_vector * torch.log(estimated_dv + self.hparams.epsilon) + (
                    1.0 - selection_vector) * torch.log(
                1.0 - estimated_dv + self.hparams.epsilon))

        exploration_bonus = torch.max(torch.mean(estimated_dv.squeeze()) - self.exploration_threshold,
                                      torch.tensor(0.0, device=estimated_dv.device)) + torch.max(
            (1.0 - self.exploration_threshold) - torch.mean(estimated_dv.squeeze()),
            torch.tensor(0.0, device=estimated_dv.device))

        cross_entropy_loss_sum = 0.0

        accuracy_tracker = pl.metrics.Accuracy(compute_on_step=False)

        if is_corrupted is not None:
            with torch.no_grad():
                self.dve.eval()
                corrupted_indices = torch.where(is_corrupted)[0]
                clean_indices = torch.where(~is_corrupted)[0]

                self.log('mean_corrupted_dve', self(x[corrupted_indices], y[corrupted_indices]).mean(), prog_bar=True)
                self.log('mean_clean_dve', self(x[clean_indices], y[clean_indices]).mean(), prog_bar=True)

            self.dve.train()

        for val_batch in self.validation_dataloader:
            if len(val_batch) == 2:
                x_val, y_val = val_batch
            else:
                x_val, y_val, val_corrupted = val_batch
            with torch.no_grad():
                self.prediction_model.eval()
                logits = self.prediction_model(x_val.cuda()).cpu()
                accuracy_tracker(logits.detach().cpu(), y_val.detach().cpu())
                cross_entropy_loss_sum += F.cross_entropy(logits,
                                                          y_val,
                                                          reduction='sum')
        mean_cross_entropy_loss = cross_entropy_loss_sum / self.val_split
        val_accuracy = accuracy_tracker.compute()
        dve_loss = -(val_accuracy - self.validation_performance) * log_prob + 1.e3 * exploration_bonus
        self.baseline_delta = (self.hparams.T - 1) * self.baseline_delta / self.hparams.T + \
                              mean_cross_entropy_loss / self.hparams.T
        self.log('val_accuracy', val_accuracy, prog_bar=True, on_step=True)
        self.log('training_accuracy', training_accuracy, prog_bar=True, on_step=True)
        self.log('estimated_dv_sum', estimated_dv.sum(), prog_bar=True, on_step=True)
        self.log('estimated_dv_mean', estimated_dv.mean(), prog_bar=True, on_step=True)
        self.log('estimated_dv_std', estimated_dv.std(), prog_bar=True, on_step=True)
        self.log('exploration_bonus', exploration_bonus, prog_bar=True, on_step=True)
        # self.log('ori_validation_accuracy', self.validation_performance, prog_bar=True, on_step=True)
        return {'loss': dve_loss, 'val_accuracy': val_accuracy}


class GumbelDVRL(DVRL):
    def __init__(self, hparams, dve_model: GumbelDataValueEstimator, prediction_model: DVRLPredictionModel,
                 val_dataloader,
                 test_dataloader, val_split):
        super().__init__(hparams, dve_model, prediction_model, val_dataloader, test_dataloader, val_split)
        assert type(dve_model) == GumbelDataValueEstimator

    def training_step(self, batch, batch_idx):
        is_corrupted = None
        if len(batch) == 2:
            x, y = batch
        else:
            x, y, is_corrupted = batch

        estimated_dv = self(x, y)

        selection_vector = F.gumbel_softmax(estimated_dv, hard=True)[:, 1]

        exploration_bonus = torch.max(selection_vector.mean() - self.exploration_threshold,
                                      torch.tensor(0.0, device=selection_vector.device)) + torch.max(
            (1.0 - self.exploration_threshold) - selection_vector.mean(),
            torch.tensor(0.0, device=selection_vector.device))

        # calling detach here since we don't want to track gradients of ops in prediction model wrt to dve
        training_accuracy = self.prediction_model.dvrl_fit(x, y, selection_vector.detach())

        accuracy_tracker = pl.metrics.Accuracy(compute_on_step=False)

        if is_corrupted is not None:
            with torch.no_grad():
                self.dve.eval()
                corrupted_indices = torch.where(is_corrupted)[0]
                clean_indices = torch.where(~is_corrupted)[0]

                self.log('mean_corrupted_dve', selection_vector.detach()[corrupted_indices].mean(), prog_bar=True)
                self.log('mean_clean_dve', selection_vector.detach()[clean_indices].mean(), prog_bar=True)

            self.dve.train()

        for val_batch in self.validation_dataloader:
            if len(val_batch) == 2:
                x_val, y_val = val_batch
            else:
                x_val, y_val, val_corrupted = val_batch
            with torch.no_grad():
                self.prediction_model.eval()
                logits = self.prediction_model(x_val.cuda()).cpu()
                accuracy_tracker(logits.detach().cpu(), y_val.detach().cpu())

        val_accuracy = accuracy_tracker.compute()
        advantage = (val_accuracy - self.baseline_delta)
        dve_loss = (-advantage * selection_vector).mean() + exploration_bonus
        self.baseline_delta = (self.hparams.T - 1) * self.baseline_delta / self.hparams.T + \
                              val_accuracy / self.hparams.T
        self.log('val_accuracy', val_accuracy, prog_bar=True, on_step=True)
        self.log('training_accuracy', training_accuracy, prog_bar=True, on_step=True)
        self.log('estimated_dv_sum', selection_vector.detach().sum(), prog_bar=True, on_step=True)
        self.log('estimated_dv_mean', selection_vector.detach().mean(), prog_bar=True, on_step=True)
        self.log('estimated_dv_std', selection_vector.detach().std(), prog_bar=True, on_step=True)
        # self.log('ori_validation_accuracy', self.validation_performance, prog_bar=True, on_step=True)
        return {'loss': dve_loss, 'val_accuracy': val_accuracy}
