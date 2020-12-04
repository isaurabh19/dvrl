from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from dvrl.training.adversarial_attack import test
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
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
    test_datavalues = get_data_values(dve_model, test_dataloader)
    sorted_datavalues, indices = torch.sort(test_datavalues, descending=True)


class RemovingSamplesExperiment:

    def __init__(self, dataset, dve_model, train_data_loader:DataLoader, test_data_loader):
        self.dataset = dataset
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.dve_model = dve_model
        self.dve_model.eval()

    def get_data_values(self):
        with torch.no_grad:
            data_values = []
            for x, y in self.train_data_loader:
                data_values.append(self.dve_model(x, y))
            return torch.cat(data_values, dim=0)

    def generate_sub_dataloader(self, indices, hparams: {}):
        sub_dataset = [self.dataset.__getitem__(i) for i in indices]
        x, y = zip(*sub_dataset)
        x = torch.stack(x)
        new_dataset = TensorDataset(x, torch.tensor(y))
        return DataLoader(new_dataset, batch_size=hparams.batch_size)

    def run_experiment(self):
        train_data_values = self.get_data_values()
        sorted_dv, indices = torch.sort(train_data_values)



if __name__ == "__main__":
    transforms = transform_lib.Compose([
        transform_lib.ToTensor()
    ])
    dataset = MNIST(DATA_PATH + '/mnist_', train=True, download=False, transform=transforms)
    generate_sub_dataloader(dataset, [0, 1, 2], {'batch_size': 32})
