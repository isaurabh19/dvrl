from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
from torch.utils.data import Dataset

class CIFAR10DataModuleWithImageNetPreprocessing(CIFAR10DataModule):
    def default_transforms(self):
        return transform_lib.Compose([
            transform_lib.ToTensor(),
            imagenet_normalization()
        ])

class TabularDatasets(Dataset):

    def __init__(self, dataset_url):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass