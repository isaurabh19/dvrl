import pytorch_lightning as pl
import torch

from dvrl.training.models import RLDataValueEstimator, DVRLPredictionModel


class DVRL(pl.LightningModule):
    def __init__(self, hparams, prediction_model: DVRLPredictionModel, val_dataloader):
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
        self.dve = RLDataValueEstimator(dve_input_dim=self.hparams.dve_input_dim,
                                        dve_hidden_dim=self.hparams.dve_hidden_dim,
                                        dve_num_layers=self.hparams.dve_comb_dim,
                                        dve_comb_dim=self.hparams.dve_comb_dim)
        self.prediction_model = prediction_model
        self.val_dataloader = val_dataloader

    def configure_optimizers(self):
        return [], []

    ### Skipping dataloaders here since they will be different for each task

    def forward(self, data):
        """

        :param data: both x and y i.e features and labels
        :return: value of given data
        """
        x, y = data
        # inference should just call forward pass on the model
        return self.dve(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Generate selection probability
        estimated_dv = self.dve(x, y)

        selection_vector = torch.bernoulli(estimated_dv)

        if torch.sum(selection_vector) == 0:
            # exception when selection probability is 0
            estimated_dv = 0.5 * torch.ones_like(estimated_dv)
            selection_vector = torch.bernoulli(estimated_dv)

        self.prediction_model.dvrl_fit(x, y, selection_vector)
