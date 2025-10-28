'''ImageNet, TinyImageNet, CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from PIL import Image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class ImageNet(datasets.ImageNet):
    def __init__(self, args):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_data = datasets.ImageNet(root=args.data_path, split='train', download=False, transform=transform)
        self.test_data = datasets.ImageNet(root=args.data_path, split='val', download=False, transform=transform)
        
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

        self.num_classes = 1000
        self.img_size = (3, 224, 224)
    
    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self):
        # Get normalized tensor
        img = self.test_data[0][0]
        # Denormalize for visualization
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3))
        plt.imshow(img)
        plt.show()

# TinyImageNet is not available in torchvision.datasets by default.
# You need to implement a custom TinyImageNet dataset class or use an external implementation.
# The following is a placeholder for TinyImageNet, which raises NotImplementedError if used.

class TinyImageNet:
    def __init__(self, args):
        raise NotImplementedError("TinyImageNet dataset is not available in torchvision.datasets. Please implement a custom loader or use an external implementation.")


class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 

        self.CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        self.CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [transforms.Resize((args.resize, args.resize))]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD),
            ]
        if args.noise > 0.0:
            self.train_transform_list += [AddGaussianNoise(mean=0., std=args.noise)]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 100

    def shape(self):
        return self.train_data[0][0].shape
    
    def visual(self): 
        img = self.test_data[0][0]
        # Use self.CIFAR100_STD and self.CIFAR100_MEAN instead of hardcoded values
        img = img * torch.tensor(self.CIFAR100_STD).view(3, 1, 1) + torch.tensor(self.CIFAR100_MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3)) 
        plt.imshow(img)
        plt.axis('off')  # Optional: cleaner look
        plt.show()
        
class CIFAR10(datasets.CIFAR10): 
    def __init__(self, args):

        self.CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR10_STD = (0.2470, 0.2435, 0.2616)

        # Transformations
        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [transforms.Resize((args.resize, args.resize))]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD),
            ]
        if args.noise > 0.0:
            self.train_transform_list += [AddGaussianNoise(mean=0., std=args.noise)]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 10

    def shape(self): 
        return self.train_data[0][0].shape
    
    def visual(self): 
        img = self.test_data[0][0]
        # Use self.CIFAR10_STD and self.CIFAR10_MEAN instead of hardcoded values
        img = img * torch.tensor(self.CIFAR10_STD).view(3, 1, 1) + torch.tensor(self.CIFAR10_MEAN).view(3, 1, 1)
        img = img.permute(1, 2, 0).clamp(0, 1)
        plt.figure(figsize=(6, 3)) 
        plt.imshow(img)
        plt.axis('off')  # Optional: cleaner look
        plt.show()