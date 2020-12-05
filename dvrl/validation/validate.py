import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as transform_lib
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

DATA_PATH = 'data/raw'
# datamodule = MNISTDataModule(data_dir=DATA_PATH + '/mnist_')
# datamodule.prepare_data()  # downloads data to given path
# train_dataloader = datamodule.train_dataloader(batch_size= 32)
# for batch in train_dataloader:
#     x, y = batch
#     print(x.size(), y.size())
# x = torch.rand(10)
# y = torch.rand(10)
# output = torch.stack([x, y], dim=1)
# print(output)
# print(output[:, 0])
transforms = transform_lib.Compose([
    transform_lib.ToTensor()
])
dataset = MNIST(DATA_PATH + '/mnist_', train=False, download=True, transform=transforms)

img, t = dataset.__getitem__(0)
subdataset = [dataset.__getitem__(i) for i in [0, 2, 4, 6, 8, 10]]
imgs, targets = zip(*subdataset)
imgs = torch.stack(imgs)
new_dataset = TensorDataset(imgs, torch.tensor(targets))
dl = DataLoader(new_dataset, batch_size=2)
for batch in dl:
    x, y = batch
    print(x.size(), y.size())


# fin = torch.rand(10)
# print(fin)
# sorted_fin, indices = torch.sort(fin, descending=True)
# print(indices)
# print(indices[2:])
# mean_fin = torch.mean(sorted_fin)
# print(mean_fin, torch.std(sorted_fin))
# print(sorted_fin[sorted_fin > mean_fin])
