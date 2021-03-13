from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
def get_train_class_count(dataset,classes):
    #dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    class_count = {}
    for _, index in dataset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count