import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class DVRLPredictionModel(pl.LightningModule):
    def __init__(self, hparams, prediction_model: torch.nn.Module, encoder_model: nn.Module = nn.Identity):
        super().__init__()
        self.hparams = hparams
        self.prediction_model = prediction_model
        self.encoder_model = encoder_model

    def forward(self, x_in):
        # inference should just call forward pass on the model
        return self.prediction_model(self.encoder_model(x_in))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.predictor_lr)

    def dvrl_fit(self, x, y, selection_vector):
        # not sure if this should be in training step or a custom method like this, will need to think about it.
        dataset = TensorDataset(x, y, selection_vector)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.inner_batch_size, pin_memory=True)

        optimizer = self.optimizers()

        for inner_iteration in range(self.hparams.num_inner_iterations):
            # should ideally pick a random sample of size dataloader batch size, TODO
            x_pred_in, y_pred_in, s_pred_in = next(iter(dataloader))

            optimizer.zero_grad()

            outputs = self(x)
            loss = F.cross_entropy(outputs, y, reduction='none')
            # we ask for unreduced cross-entropy so that we can multiply it with s_pred_in and then reduce
            loss = loss * s_pred_in
            loss.mean().backward()

            optimizer.step()


class RLDataValueEstimator(nn.Module):
    def __init__(self, dve_input_dim: int, dve_hidden_dim: int, dve_num_layers: int, dve_comb_dim: int,
                 activation_fn=F.relu):
        super().__init__()
        self.input_layer = nn.Linear(dve_input_dim, dve_hidden_dim)
        self.hidden_layers = [nn.Linear(dve_hidden_dim, dve_hidden_dim) for _ in range(dve_num_layers - 3)]
        self.comb_layer = nn.Linear(dve_hidden_dim, dve_comb_dim)
        self.output_layer = nn.Linear(dve_comb_dim, 1)
        self.activation_fn = activation_fn

    def forward(self, x_input, y_input):
        # concat x, y as in https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py#L192
        model_inputs = torch.cat([x_input, y_input], dim=1)

        # affine transform input dim to hidden dim
        model_inputs = self.activation_fn(self.input_layer(model_inputs))

        # pass through hidden layers
        for layer in self.hidden_layers:
            model_inputs = self.activation_fn(layer(model_inputs))

        # affine transform to dve_comb_dim
        # Note: At this point, the original TF code concats with y_hat_input which I don't understand fully yet.
        model_inputs = self.activation_fn(self.comb_layer(model_inputs))

        # project to 1D and squash using sigmoid
        return F.sigmoid(self.output_layer(model_inputs))
