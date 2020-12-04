import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam

from dvrl.training.models import RLDataValueEstimator, DVRLPredictionModel


class DVRL(pl.LightningModule):
    def __init__(self, hparams, prediction_model: DVRLPredictionModel, val_dataloader, val_split):
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
        self.dve = RLDataValueEstimator(dve_hidden_dim=self.hparams.dve_hidden_dim,
                                        dve_num_layers=self.hparams.dve_num_layers,
                                        dve_comb_dim=self.hparams.dve_comb_dim,
                                        num_classes=self.hparams.num_classes)
        self.prediction_model = prediction_model
        self.validation_dataloader = val_dataloader
        self.baseline_delta = 0.0
        self.val_split = val_split

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

    def training_step(self, batch, batch_idx):
        x, y = batch
        estimated_dv = self(x, y)

        selection_vector = torch.bernoulli(estimated_dv)

        if torch.sum(selection_vector) == 0:
            # exception when selection probability is 0
            estimated_dv_ = 0.5 * torch.ones_like(estimated_dv)
            selection_vector = torch.bernoulli(estimated_dv_)

        # calling detach here since we don't want to track gradients of ops in prediction model wrt to dve
        self.prediction_model.dvrl_fit(x.detach(), y.detach(), selection_vector.detach())

        log_prob = torch.sum(
            selection_vector * torch.log(estimated_dv + self.hparams.epsilon) + (1.0 - selection_vector) * torch.log(
                1.0 - estimated_dv + self.hparams.epsilon))

        cross_entropy_loss_sum = 0.0
        y_acc = []
        for val_batch in self.validation_dataloader:
            x_val, y_val = val_batch
            with torch.no_grad():
                y_hat = self.prediction_model(x_val.cuda()).cpu()
                y_acc.append((torch.argmax(F.softmax(y_hat, dim=1)) == y_val).float())
                cross_entropy_loss_sum += F.cross_entropy(y_hat,
                                                          y_val,
                                                          reduction='sum')
        mean_cross_entropy_loss = cross_entropy_loss_sum / self.val_split
        dve_loss = -(mean_cross_entropy_loss - self.baseline_delta) * log_prob
        val_accuracy = torch.mean(torch.cat(y_acc))
        self.baseline_delta = (self.hparams.T - 1) * self.baseline_delta / self.hparams.T + \
                              mean_cross_entropy_loss / self.hparams.T
        self.log('val_accuracy', val_accuracy, prog_bar=True, on_step=True)
        self.log('estimated_dv_sum', estimated_dv.sum(), prog_bar=True, on_step=True)
        return {'loss': dve_loss, 'val_accuracy': val_accuracy}
