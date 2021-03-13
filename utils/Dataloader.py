import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms
def calculate_mean_std(dataset):
  loader = torch.utils.data.DataLoader(dataset,
                          batch_size=128,
                          num_workers=0,
                          shuffle=False)

  mean = 0.
  std = 0.
  for images, _ in loader:
      batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)

  mean /= len(loader.dataset)
  std /= len(loader.dataset)
  mean=mean.tolist()
  std=std.tolist()
  return mean,std 



def train_test_loader(batch_size,num_workers):
  trainset = torchvision.datasets.CIFAR10(root='./data',download=True,transform=transforms.ToTensor())
  mean,std = calculate_mean_std(trainset)
  # mean =[0.5,0.5,0.5]
  # std =[0.5,0.5,0.5]
  train_transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.RandomCrop(32, padding=4),
   transforms.RandomHorizontalFlip(),
  transforms.Normalize(mean, std)
  ])
  test_transform = transforms.Compose(
  [transforms.ToTensor(),
  transforms.Normalize(mean, std)
  ])

#   train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
    
#   test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])

  train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
  test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
  SEED = 1
  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)
  # For reproducibility
  torch.manual_seed(SEED)
  if cuda:
      torch.cuda.manual_seed(SEED)
  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=Args.batch_size)
  # train dataloader
  trainloader = torch.utils.data.DataLoader(train, **dataloader_args)
  # test dataloader
  testloader = torch.utils.data.DataLoader(test, **dataloader_args)
  return trainloader,testloader

def get_classes():
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return classes
    