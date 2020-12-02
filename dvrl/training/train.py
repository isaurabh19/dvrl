import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning import Trainer

from dvrl.training.dvrl import DVRL
from dvrl.training.models import DVRLPredictionModel

# DATA
DATA_PATH = 'data/raw'
datamodule = CIFAR10DataModule(DATA_PATH + '/cifar')
datamodule.prepare_data()  # downloads data to given path
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()
test_dataloader = datamodule.test_dataloader()

hp_params = {}  # Create HP params dict
pred_arch = None  # some torch nn module.
pred_model = DVRLPredictionModel(hp_params, pred_arch)
# this will be replaced with an actual model - turns out there is no pretrained
# CIFAR model
encoder_model = torch.nn.Identity

dvrl_model = DVRL(hp_params, pred_model, val_dataloader, datamodule.val_split, encoder_model=encoder_model)
trainer = Trainer()
trainer.fit(dvrl_model, train_dataloader)
