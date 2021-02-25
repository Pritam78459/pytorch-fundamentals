import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

dataset = MNIST(root = 'data/', download=True)
print(len(dataset))

test_dataset = MNIST(root = 'data/', train=False)
print(len(test_dataset))

dataset = MNIST(root = 'data/', train=True, transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label)

