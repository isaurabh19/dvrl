from typing import Optional, Sized
import matplotlib.pyplot as plt
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from dvrl.training.adversarial_attack import test
from dvrl.training.train_predictor_only import run_prediction
from dvrl.training.models import SimpleConvNet
import torch
from torch.utils.data import TensorDataset, DataLoader, Sampler
from torchvision.datasets import MNIST
from torchvision import transforms as transform_lib

DATA_PATH = 'data/raw'
use_cuda = True


def full_dataset_attack(pred_model, device, test_dataloader):
    accuracies = []
    examples = []
    for eps in [0.1, 0.25, 0.05]:
        acc, ex = test(pred_model, device, test_dataloader, eps)
        accuracies.append(acc)
        examples.append(ex)
    return accuracies, examples


def run_adversarial_attack_mnist(dvrl_hparams, dve_model, pred_model):
    pred_model.eval()
    dve_model.eval()
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    datamodule = MNISTDataModule(data_dir=DATA_PATH + '/mnist_')
    datamodule.prepare_data()  # downloads data to given path
    test_dataloader = datamodule.test_dataloader(batch_size=dvrl_hparams.get('outer_batch_size', 32))
    full_dataset_attack(pred_model, device, test_dataloader)
    # test_datavalues = get_data_values(dve_model, test_dataloader)
    # sorted_datavalues, indices = torch.sort(test_datavalues, descending=True)


class YourSampler(Sampler):
    def __init__(self, indices):
        super().__init__(None)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


class RemovingSamplesExperiment:

    def __init__(self, dataset, dve_model, train_data_loader: DataLoader, val_data_loader, test_data_loader,
                 predictor_hparams, encoder_out_dim=50):
        self.dataset = train_data_loader.dataset
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader
        self.predictor_hparams = predictor_hparams
        self.encoder_out_dim = encoder_out_dim
        self.dve_model = dve_model
        self.dve_model.eval()

    def get_data_values(self):
        data_values = []
        x_val = []
        y_val = []
        for x, y in self.train_data_loader:
            with torch.no_grad():
                data_values.append(self.dve_model(x, y))
            x_val.append(x)
            y_val.append(y)
        return torch.cat(data_values, dim=0).squeeze(), torch.cat(x_val, dim=0), torch.cat(y_val, dim=0)

    def generate_sub_dataloader(self, indices, batch_size, x_val, y_val):
        # sub_dataset = [self.dataset[i] for i in indices.squeeze()]
        # x, y = zip(*sub_dataset)
        # x = torch.stack(x)
        # print(x.size(), torch.tensor(y).size())
        sampler = YourSampler(indices)
        new_dataset = TensorDataset(x_val, y_val)
        return DataLoader(new_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    def run_experiment(self):
        train_data_values, x_values, y_values = self.get_data_values()
        _, asc_indices = torch.sort(train_data_values)
        desc_indices = torch.flip(asc_indices, dims=[0])
        slice_incr = int(len(train_data_values) / 10)
        fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for indices in [asc_indices, desc_indices]:
            print(indices.size())
            slice = 0
            count = 0
            accuracies = []
            while count < 6:
                new_train_dataloader = self.generate_sub_dataloader(indices[slice:], 2000, x_values, y_values)
                new_pred_model = run_prediction(self.predictor_hparams, new_train_dataloader, self.val_data_loader,
                                                self.test_data_loader, SimpleConvNet(), self.encoder_out_dim)
                accuracies.append(new_pred_model.test_acc.compute())
                slice += slice_incr
                count += 1
            print("Acc vs Fraction removed", list(zip(accuracies, fractions)))


if __name__ == "__main__":
    transforms = transform_lib.Compose([
        transform_lib.ToTensor()
    ])
    dataset = MNIST(DATA_PATH + '/mnist_', train=True, download=False, transform=transforms)
    # generate_sub_dataloader(dataset, [0, 1, 2], {'batch_size': 32})
