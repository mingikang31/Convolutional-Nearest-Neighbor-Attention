'''ImageNet, TinyImageNet, CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
import os 
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from PIL import Image
import torch.distributed as dist 

# Huggingface Datasets + GPT Tokenizer
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer 

# Timm 
from timm.data import create_transform 
from timm.data.mixup import Mixup 

# Distributed Data Parallel 
from torch.utils.data.distributed import DistributedSampler

"""
Swin Transformer Config for Hyperparameters
https://github.com/microsoft/Swin-Transformer/blob/main/config.py
"""
# Mixup/CutMix setup
mixup_fn = Mixup(
    mixup_alpha=0.8,      # Mixup alpha
    cutmix_alpha=1.0,     # CutMix alpha
    cutmix_minmax=None,
    prob=1.0,             # Probability of applying mixup or cutmix
    switch_prob=0.5,      # Probability of switching between mixup and cutmix
    mode='batch',
    label_smoothing=0.1,  # Label smoothing
    num_classes=1000      # ImageNet classes
)


'''ImageNet1K Dataset Class'''
class ImageNet1K:
    def __init__(self, args):
        self.cache_dir = os.path.join(args.data_path, f"imagenet1k_hf_cache")
        self.input_size = getattr(args, 'input_size', 224) # Default to 224 for Swin/ResNet
        self.num_classes = 1000
        self.img_size = (3, 224, 224) 

        # Train Transformation 
        self.train_transform_func = create_transform(
            input_size=self.input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1', # "Standard" Industry Augmentation
            interpolation='bicubic', 
            re_prob=0.25, # Random Erasing
            re_mode='pixel', 
            re_count=1
        )

        # Validation Transformation
        crop_pct = 224 / 256
        size = int(self.input_size / crop_pct)
        self.val_transform_func = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.CenterCrop(self.input_size), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Loading/Download Dataset 
        if os.path.exists(os.path.join(self.cache_dir, "dataset_dict.json")):
            print(f"Loading cached ImageNet file from {self.cache_dir}")
            self.dataset = load_from_disk(self.cache_dir)
        else: 
            print("Downloading/Loading ImageNet-1k from Hugging Face")
            self.dataset = load_dataset(
                "ILSVRC/imagenet-1k", 
                cache_dir=os.path.join(args.data_path, "hf_downloads")
            )

            print(f"Saving preprocessed dataset to {self.cache_dir}")
            self.dataset.save_to_disk(self.cache_dir)

        # Dataset transforms 
        self.dataset['train'].set_transform(self.process_train)
        self.dataset['validation'].set_transform(self.process_val)

        # Distributed Sampler
        if args.ddp:
            self.train_sampler = DistributedSampler(self.dataset['train'], shuffle=True)
            self.val_sampler = DistributedSampler(self.dataset['validation'], shuffle=False)
        else: 
            self.train_sampler = None
            self.val_sampler = None
            
        # Data Loaders 
        self.train_loader = DataLoader(
            dataset=self.dataset['train'], 
            batch_size=args.batch_size if not args.ddp else args.ddp_batch_size,
            shuffle=(self.train_sampler is None),
            sampler = self.train_sampler, 
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor, 
            pin_memory=args.pin_memory, 
            drop_last=True)

        self.val_loader = DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=args.batch_size if not args.ddp else args.ddp_batch_size, 
            shuffle=False, 
            sampler=self.val_sampler,
            num_workers=args.num_workers, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory, 
            drop_last=False)

        self.test_loader = self.val_loader

    def process_train(self, examples):
        images = [img.convert("RGB") for img in examples['image']]

        # Apply timm transform to each image 
        pixel_values = [self.train_transform_func(img) for img in images]
        return {
            "pixel_values": pixel_values, 
            "labels": examples['label']
        }
        
    def process_val(self, examples):
        images = [img.convert("RGB") for img in examples['image']]

        # Apply timm transform to each image 
        pixel_values = [self.val_transform_func(img) for img in images]
        return {
            "pixel_values": pixel_values, 
            "labels": examples['label']
        }

'''Wikitext-103 Dataset Class'''
class WikiText103:
    def __init__(self, args):
        self.max_seq_length = args.max_seq_length
        self.block_size = self.max_seq_length
        self.cache_dir = os.path.join(args.data_path, f"wikitext103_cache_{args.max_seq_length}")
        
        # Tokenizer 
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if os.path.exists(os.path.join(self.cache_dir, "dataset_dict.json")):
            print(f"Loading cached WikiText from {self.cache_dir}")
            self.lm_dataset = load_from_disk(self.cache_dir)
        else: 
            print("Downloading/Loading WikiText103 from Hugging Face")
            os.makedirs(self.cache_dir, exist_ok=True)

            # Dataset
            self.original_dataset = load_dataset("Salesforce/wikitext", 
                                                 "wikitext-103-v1", 
                                                cache_dir=os.path.join(args.data_path, "hf_downloads"))
        
            self.tokenized_dataset = self.original_dataset.map(
                self.tokenize_function, 
                batched=True, 
                remove_columns=["text"]
            )
            
            self.lm_dataset = self.tokenized_dataset.map(
                self.group_texts, 
                batched=True
            )
            self.lm_dataset.set_format(type="torch", columns=["input_ids"])
            print(f"Saving preprocessed dataset to {self.cache_dir}")
            self.lm_dataset.save_to_disk(self.cache_dir)

        # Data Loaders 
        self.train_loader = DataLoader(
            dataset=self.lm_dataset["train"], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor, 
            pin_memory=args.pin_memory)
        
        self.test_loader = DataLoader(
            dataset=self.lm_dataset["test"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers // 2, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)
        
        self.val_loader = DataLoader(
            dataset=self.lm_dataset["validation"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers // 2, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)

    def group_texts(self, examples): 
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

'''CIFAR-10 and CIFAR-100 Dataset Classes'''
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 

        self.CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        self.CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [
                transforms.Resize((args.resize, args.resize))
                ]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        self.train_loader = DataLoader(
            dataset=self.train_data, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)

        
        self.test_loader = DataLoader(
            dataset=self.test_data, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 100

    def shape(self):
        return self.train_data[0][0].shape
        
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
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
        self.train_loader = DataLoader(
            dataset=self.train_data, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)
        
        self.test_loader = DataLoader(
            dataset=self.test_data, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 10

    def shape(self): 
        return self.train_data[0][0].shape
    