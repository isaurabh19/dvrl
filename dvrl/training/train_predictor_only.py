import copy

from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning import Trainer
from torchvision import models

from dvrl.data.make_dataset import CIFAR10DataModuleWithImageNetPreprocessing
from dvrl.training.models import DVRLPredictionModel, SimpleConvNet

DATA_PATH = 'data/raw'


def run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader, encoder_model,
                   encoder_out_dim):
    pred_model = DVRLPredictionModel(prediction_hparams, copy.deepcopy(encoder_model), encoder_out_dim=encoder_out_dim)
    trainer = Trainer(gpus=1, max_epochs=5)
    trainer.fit(pred_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=pred_model, test_dataloaders=test_dataloader)
    return pred_model


def run_cifar_prediction(prediction_hparams):
    datamodule = CIFAR10DataModuleWithImageNetPreprocessing(DATA_PATH + '/cifar')
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    encoder_model = models.resnet18(pretrained=True)
    return run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader,
                          encoder_model=encoder_model, encoder_out_dim=1000)


def run_fashion_mnist_prediction(prediction_hparams):
    # DATA
    datamodule = FashionMNISTDataModule(data_dir=DATA_PATH + '/fashion')
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    encoder_model = SimpleConvNet()
    return run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader,
                          encoder_model, encoder_out_dim=50)


def run_mnist_prediction(prediction_hparams):
    # DATA
    datamodule = MNISTDataModule(data_dir=DATA_PATH + '/mnist_')
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    encoder_model = SimpleConvNet()
    return run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader,
                          encoder_model, encoder_out_dim=50)


if __name__ == '__main__':
    run_prediction({})
