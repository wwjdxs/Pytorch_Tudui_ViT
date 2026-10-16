"""小型共享数据视图：训练/验证共用一份CIFAR原始图片，避免复制大数组。"""
from torch.utils.data import Dataset
from torchvision import transforms


class CIFARSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset  # 两个子集都引用同一份图片数据，不复制图片。
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, target = self.dataset[int(self.indices[index])]
        return self.transform(image), target


def get_transform(training=False, normalize=True):
    operations = []
    if training:
        operations += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    operations.append(transforms.ToTensor())
    if normalize:
        operations.append(transforms.Normalize((0.5,)*3, (0.5,)*3))
    return transforms.Compose(operations)
