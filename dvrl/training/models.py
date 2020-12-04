import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

from dvrl.utils.metrics import AccuracyTracker


class DVRLPredictionModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.hparams.pred_hidden_dim, self.hparams.pred_hidden_dim) for _ in
             range(self.hparams.pred_num_layers - 1)])
        self.output_layer = nn.Linear(self.hparams.pred_hidden_dim, self.hparams.num_classes)
        self.activation_fn = self.hparams.activation_fn
        self.encoder_model = models.resnet18(pretrained=True)
        self.encoder_model.fc = nn.Linear(512, self.hparams.num_classes)
        self.input_layer = nn.Linear(512, self.hparams.pred_hidden_dim)
        self.optimizer = Adam(self.parameters(), lr=self.hparams.predictor_lr)

    def forward(self, x_in):
        # inference should just call forward pass on the model
        return self.encoder_model(x_in)
        # x_in = self.activation_fn(self.input_layer(x_in))
        # for layer in self.hidden_layers:
        #     x_in = self.activation_fn(layer(x_in))
        # return self.output_layer(x_in)

    def dvrl_fit(self, x, y, selection_vector) -> float:
        self.train()
        # not sure if this should be in training step or a custom method like this, will need to think about it.
        dataset = TensorDataset(x, y, selection_vector)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.inner_batch_size, pin_memory=False)

        optimizer = self.optimizer

        accuracy_tracker = AccuracyTracker()

        for inner_iteration in range(self.hparams.num_inner_iterations):
            # should ideally pick a random sample of size dataloader batch size, TODO
            x_pred_in, y_pred_in, s_pred_in = next(iter(dataloader))

            optimizer.zero_grad()

            outputs = self(x_pred_in)
            loss = F.cross_entropy(outputs, y_pred_in, reduction='none')
            # we ask for unreduced cross-entropy so that we can multiply it with s_pred_in and then reduce
            loss = loss * s_pred_in.squeeze()
            loss.mean().backward()

            optimizer.step()

            accuracy_tracker.track(y_pred_in.detach().cpu(), outputs.detach().cpu())

        return accuracy_tracker.compute()


class RLDataValueEstimator(pl.LightningModule):
    def __init__(self, dve_hidden_dim: int, dve_num_layers: int, dve_comb_dim: int,
                 num_classes: int, activation_fn=F.relu):
        super().__init__()
        self.encoder_model = models.resnet18(pretrained=True)
        # self.encoder_model.fc = nn.Linear(512, 1000)
        self.input_layer = nn.Linear(1000 + num_classes, dve_hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(dve_hidden_dim, dve_hidden_dim) for _ in range(dve_num_layers - 3)])
        self.comb_layer = nn.Linear(dve_hidden_dim, dve_comb_dim)
        self.output_layer = nn.Linear(dve_comb_dim, 1)
        self.activation_fn = activation_fn
        self.num_classes = num_classes

    def forward(self, x_input, y_input):
        # concat x, y as in https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py#L192
        x_input = self.encoder_model(x_input)
        model_inputs = torch.cat([x_input, F.one_hot(y_input, num_classes=self.num_classes)], dim=1)

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
