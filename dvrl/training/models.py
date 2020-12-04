from typing import List, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


class DVRLPredictionModel(pl.LightningModule):
    def __init__(self, hparams, encoder_model: nn.Module, encoder_out_dim: int):
        super().__init__()
        self.hparams = hparams
        self.encoder_model = encoder_model
        self.mlp = nn.Sequential(nn.Linear(encoder_out_dim, 20), nn.ReLU(), nn.Linear(20, self.hparams.num_classes))
        self.activation_fn = self.hparams.activation_fn
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        self.optimizer = None

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.predictor_lr)

    def forward(self, x_in):
        return self.mlp(self.activation_fn(self.encoder_model(x_in)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_acc_step', self.train_acc(logits, y))
        self.log('predictor_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_acc_step', self.valid_acc(logits, y))
        self.log('val_loss', loss)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.log('train_acc_full', self.train_acc.compute())

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        self.log('val_acc_full', self.valid_acc.compute())

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.log('test_acc_full', self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_acc_step', self.test_acc(logits, y))
        self.log('test_loss', loss)

    def dvrl_fit(self, x, y, selection_vector) -> float:
        if self.optimizer is None:
            self.optimizer = self.configure_optimizers()

        self.train()
        # not sure if this should be in training step or a custom method like this, will need to think about it.
        dataset = TensorDataset(x, y, selection_vector)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.inner_batch_size, pin_memory=False)

        optimizer = self.optimizer

        accuracy_tracker = pl.metrics.Accuracy()

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

            # accuracy_tracker.track(y_pred_in.detach().cpu(), outputs.detach().cpu())
            accuracy_tracker(outputs.detach().cpu(), y_pred_in.detach().cpu())

        return accuracy_tracker.compute()


class RLDataValueEstimator(pl.LightningModule):
    def __init__(self, encoder_model: nn.Module, num_classes: int, encoder_out_dim: int, activation_fn=F.relu):
        super().__init__()
        self.encoder_model = encoder_model
        self.mlp = nn.Sequential(nn.Linear(encoder_out_dim + num_classes, 20), nn.ReLU(), nn.Linear(20, 1))
        self.activation_fn = activation_fn
        self.num_classes = num_classes

    def forward(self, x_input, y_input):
        # concat x, y as in https://github.com/google-research/google-research/blob/master/dvrl/dvrl.py#L192
        x_input = self.encoder_model(x_input)
        model_inputs = torch.cat([x_input, F.one_hot(y_input, num_classes=self.num_classes)], dim=1)
        return F.sigmoid(self.mlp(self.activation_fn(model_inputs)))
