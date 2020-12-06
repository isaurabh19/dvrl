import copy

from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning import Trainer
from torchvision import models

from dvrl.data.make_dataset import CIFAR10DataModuleWithImageNetPreprocessing
from dvrl.training.dvrl_modules import DVRL
from dvrl.training.models import DVRLPredictionModel, RLDataValueEstimator, SimpleConvNet

DATA_PATH = 'data/raw'


def run_dvrl(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
             encoder_model, encoder_out_dim):
    pred_model = DVRLPredictionModel(prediction_hparams, copy.deepcopy(encoder_model), encoder_out_dim=encoder_out_dim)
    dve_model = RLDataValueEstimator(copy.deepcopy(encoder_model), num_classes=dvrl_hparams['num_classes'],
                                     encoder_out_dim=encoder_out_dim)
    dvrl_model = DVRL(dvrl_hparams, dve_model, pred_model, val_dataloader, val_split)
    trainer = Trainer(gpus=1, max_epochs=25)
    trainer.fit(dvrl_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=pred_model, test_dataloaders=test_dataloader)
    return pred_model, dve_model


def run_cifar_dvrl(prediction_hparams, dvrl_hparams):
    datamodule = CIFAR10DataModuleWithImageNetPreprocessing(DATA_PATH + '/cifar',
                                                            batch_size=dvrl_hparams.get('outer_batch_size', 32))
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    val_split = datamodule.val_split
    encoder_model = models.resnet18(pretrained=True)
    return run_dvrl(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
                    encoder_model=encoder_model, encoder_out_dim=1000)


def run_fashion_mnist_dvrl(prediction_hparams, dvrl_hparams):
    # DATA
    datamodule = FashionMNISTDataModule(data_dir=DATA_PATH + '/fashion')
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    val_dataloader = datamodule.val_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    test_dataloader = datamodule.test_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))

    val_split = datamodule.val_split
    encoder_model = SimpleConvNet()
    return run_dvrl(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
                    encoder_model, encoder_out_dim=50)


def run_mnist_dvrl(prediction_hparams, dvrl_hparams):
    # DATA
    datamodule = MNISTDataModule(data_dir=DATA_PATH + '/mnist_')
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    val_dataloader = datamodule.val_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    test_dataloader = datamodule.test_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))

    val_split = datamodule.val_split
    encoder_model = SimpleConvNet()
    return run_dvrl(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
                    encoder_model, encoder_out_dim=50)


if __name__ == '__main__':
    run_dvrl({})
