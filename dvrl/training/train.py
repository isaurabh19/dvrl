from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from dvrl.training.dvrl import DVRL
from dvrl.training.models import DVRLPredictionModel
from pytorch_lightning import Trainer

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
dvrl_model = DVRL(hp_params, pred_model, val_dataloader)
trainer = Trainer()
trainer.fit(dvrl_model, train_dataloader)
