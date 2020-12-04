from pytorch_lightning import Trainer

from dvrl.data.make_dataset import CIFAR10DataModuleWithImageNetPreprocessing
from dvrl.training.dvrl_modules import DVRL
from dvrl.training.models import DVRLPredictionModel


def run_dvrl(prediction_hparams, dvrl_hparams):
    # DATA
    DATA_PATH = 'data/raw'
    datamodule = CIFAR10DataModuleWithImageNetPreprocessing(DATA_PATH + '/cifar',
                                                            batch_size=dvrl_hparams.get('outer_batch_size', 32))
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    pred_model = DVRLPredictionModel(prediction_hparams)
    dvrl_model = DVRL(dvrl_hparams, pred_model, val_dataloader, datamodule.val_split)
    trainer = Trainer(gpus=1)
    trainer.fit(dvrl_model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)


if __name__ == '__main__':
    run_dvrl({})
