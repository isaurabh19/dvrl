from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from dvrl.training.adversarial_attack import test
import torch

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


def get_data_values(dve_model, dataloader):
    data_values = []
    for x, y in dataloader:
        data_values.append(dve_model(x, y))
    return torch.cat(data_values, dim=0)


def run_data_valuation_prediction():
    pass

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


