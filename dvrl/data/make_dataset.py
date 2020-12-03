from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib


class CIFAR10DataModuleWithImageNetPreprocessing(CIFAR10DataModule):
    def default_transforms(self):
        return transform_lib.Compose([
            transform_lib.ToTensor(),
            imagenet_normalization()
        ])
