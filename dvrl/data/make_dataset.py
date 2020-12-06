import numpy as np
import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


class CIFAR10DataModuleWithImageNetPreprocessing(CIFAR10DataModule):
    def default_transforms(self):
        return transform_lib.Compose([
            transform_lib.ToTensor(),
            imagenet_normalization()
        ])


class CorruptionDataSet(Dataset):
    def __init__(self, dataset: Dataset,
                 num_classes: int,
                 noise_ratio: float = 0.0):
        self.dataset = dataset
        self.num_classes = num_classes
        self.noise_ratio = noise_ratio

    def __getitem__(self, index) -> T_co:
        data, label = self.dataset[index]

        bit = torch.rand(1).item()
        is_corrupted = False

        if bit < self.noise_ratio:
            label = np.random.randint(self.num_classes)
            is_corrupted = True

        return data, label, torch.tensor(is_corrupted)

    def __len__(self):
        return len(self.dataset)


class CorruptedMNISTDataModule(MNISTDataModule):

    def __init__(self, data_dir: str = "./", val_split: int = 5000, num_workers: int = 16, normalize: bool = False,
                 seed: int = 42, batch_size: int = 32, noise_ratio: float = 0.0, *args, **kwargs):
        super().__init__(data_dir, val_split, num_workers, normalize, seed, batch_size, *args, **kwargs)
        assert 0.0 <= noise_ratio <= 1.0
        self.noise_ratio = noise_ratio

    def train_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.train_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            CorruptionDataSet(dataset_train, num_classes=10, noise_ratio=self.noise_ratio),
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()
        dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            CorruptionDataSet(dataset_val, num_classes=10, noise_ratio=0.0),
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self, batch_size=32, transforms=None):
        """
        MNIST test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()

        dataset = MNIST(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            CorruptionDataSet(dataset, num_classes=10, noise_ratio=0.0), batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True, pin_memory=True
        )
        return loader
