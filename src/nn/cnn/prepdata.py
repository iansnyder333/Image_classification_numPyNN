import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from constants import *


# MNIST dataset

mnist_train = dsets.MNIST(
    root="Image_classification_numPyNN/data/MNIST_data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

mnist_test = dsets.MNIST(
    root="Image_classification_numPyNN/data/MNIST_data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

# dataset loader
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True
)
fasion_mnist_train = dsets.FashionMNIST(
    root="Image_classification_numPyNN/data/FashionMNIST_data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

fasion_mnist_test = dsets.FashionMNIST(
    root="Image_classification_numPyNN/data/FashionMNIST_data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
fasion_data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True
)
