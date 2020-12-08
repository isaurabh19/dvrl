import copy

from pytorch_lightning import Trainer
from torchvision import models

from dvrl.data.make_dataset import CorruptedMNISTDataModule, CorruptedFashionMNISTDataModule, CorruptedCIFARDataModule
from dvrl.training.models import DVRLPredictionModel, SimpleConvNet
from dvrl.training.train import run_dvrl
from dvrl.training.train_gumbel import run_gumbel

DATA_PATH = 'data/raw'


def run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader, encoder_model,
                   encoder_out_dim):
    pred_model = DVRLPredictionModel(prediction_hparams, copy.deepcopy(encoder_model), encoder_out_dim=encoder_out_dim)
    trainer = Trainer(gpus=1, max_epochs=100)
    trainer.fit(pred_model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=pred_model, test_dataloaders=test_dataloader)
    return pred_model


def run_cifar_prediction_corrupted(prediction_hparams):
    datamodule = CorruptedCIFARDataModule(DATA_PATH + '/cifar', batch_size=256,
                                          noise_ratio=prediction_hparams['noise_ratio'],
                                          max_train_data_size=prediction_hparams['max_train_data_size'])
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    encoder_model = models.resnet18(pretrained=True)
    return run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader,
                          encoder_model=encoder_model, encoder_out_dim=1000)


def run_cifar_dvrl_corrupted(prediction_hparams, dvrl_hparams):
    # DATA
    datamodule = CorruptedCIFARDataModule(DATA_PATH + '/cifar', noise_ratio=prediction_hparams['noise_ratio'],
                                          batch_size=dvrl_hparams.get('outer_batch_size', 32),
                                          max_train_data_size=prediction_hparams['max_train_data_size'])
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    val_split = datamodule.val_split
    encoder_model = models.resnet18(pretrained=False)
    dvrl_method = dvrl_hparams.get('dve_method', 'dvrl')
    print(f'using {dvrl_method}')
    runner = run_gumbel if dvrl_method == 'gumbel' else run_dvrl
    return runner(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
                  encoder_model, encoder_out_dim=1000)


def run_fashion_mnist_prediction_corrupted(prediction_hparams):
    # DATA
    datamodule = CorruptedFashionMNISTDataModule(data_dir=DATA_PATH + '/mnist_',
                                                 noise_ratio=prediction_hparams['noise_ratio'],
                                                 max_train_data_size=prediction_hparams['max_train_data_size'])
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader(batch_size=256)
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    encoder_model = SimpleConvNet()
    return run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader,
                          encoder_model, encoder_out_dim=50)


def run_fashion_mnist_dvrl_corrupted(prediction_hparams, dvrl_hparams):
    # DATA
    datamodule = CorruptedFashionMNISTDataModule(data_dir=DATA_PATH + '/mnist_',
                                                 noise_ratio=prediction_hparams['noise_ratio'],
                                                 max_train_data_size=prediction_hparams['max_train_data_size'])
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    val_dataloader = datamodule.val_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    test_dataloader = datamodule.test_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))

    val_split = datamodule.val_split
    encoder_model = SimpleConvNet()
    dvrl_method = dvrl_hparams.get('dve_method', 'dvrl')
    print(f'using {dvrl_method}')
    runner = run_gumbel if dvrl_method == 'gumbel' else run_dvrl
    return runner(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
                  encoder_model, encoder_out_dim=50)


def run_mnist_prediction_corrupted(prediction_hparams):
    # DATA
    datamodule = CorruptedMNISTDataModule(data_dir=DATA_PATH + '/mnist_', noise_ratio=prediction_hparams['noise_ratio'],
                                          max_train_data_size=prediction_hparams['max_train_data_size'])
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader(batch_size=256)
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    encoder_model = SimpleConvNet()
    return run_prediction(prediction_hparams, train_dataloader, val_dataloader, test_dataloader,
                          encoder_model, encoder_out_dim=50)


def run_mnist_dvrl_corrupted(prediction_hparams, dvrl_hparams):
    # DATA
    datamodule = CorruptedMNISTDataModule(data_dir=DATA_PATH + '/mnist_', noise_ratio=prediction_hparams['noise_ratio'],
                                          max_train_data_size=prediction_hparams['max_train_data_size'])
    datamodule.prepare_data()  # downloads data to given path
    train_dataloader = datamodule.train_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    val_dataloader = datamodule.val_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    test_dataloader = datamodule.test_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))

    val_split = datamodule.val_split
    encoder_model = SimpleConvNet()
    dvrl_method = dvrl_hparams.get('dve_method', 'dvrl')
    print(f'using {dvrl_method}')
    runner = run_gumbel if dvrl_method == 'gumbel' else run_dvrl
    return runner(dvrl_hparams, prediction_hparams, train_dataloader, val_dataloader, test_dataloader, val_split,
                  encoder_model, encoder_out_dim=50)


if __name__ == '__main__':
    import torch.nn.functional as F

    prediction_hparams = {'activation_fn': F.relu,
                          'predictor_lr': 1e-3,
                          'num_classes': 10,
                          'inner_batch_size': 256,
                          'num_inner_iterations': 15,
                          'noise_ratio': 0.0
                          }
    run_mnist_prediction_corrupted(prediction_hparams)
