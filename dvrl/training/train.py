from pytorch_lightning import Trainer

from dvrl.data.make_dataset import CIFAR10DataModuleWithImageNetPreprocessing
from dvrl.training.dvrl import DVRL
from dvrl.training.models import DVRLPredictionModel


def run_dvrl(hp_params):
    # DATA
    DATA_PATH = 'data/raw'
    datamodule = CIFAR10DataModuleWithImageNetPreprocessing(DATA_PATH + '/cifar')
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    pred_model = DVRLPredictionModel(hp_params)
    dvrl_model = DVRL(hp_params, pred_model, val_dataloader, datamodule.val_split)
    trainer = Trainer()
    trainer.fit(dvrl_model, train_dataloader)


if __name__ == '__main__':
    run_dvrl({})
