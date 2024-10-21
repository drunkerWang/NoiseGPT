import random
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from mydatasets.imagenet_classnames import get_classnames


def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

class ilsvrc12_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, exemplar=False, num_exemplar=None, noisy_targets=None): 
        self.root = root_dir+'/ilsvrc12/'
        self.val_root = os.path.join(self.root, 'val')
        self.transform = transform
        self.mode = mode
        self.label_to_class_mapping = get_classnames('openai')
        self.num_classes = len(self.label_to_class_mapping)
        self.exemplar = exemplar

        if self.mode=='test':
            self.val_imgs = []
            self.val_targets = []
            val_dataset = datasets.ImageFolder(self.val_root, transform)
            # print(val_dataset.imgs)
            for idx in range(len(val_dataset)):
                # print(idx)
                img, target = val_dataset.imgs[idx]
                if target < self.num_classes:
                    self.val_imgs.append(img)
                    self.val_targets.append(target) 
            # print(self.val_targets)          
            self.val_targets = np.array(self.val_targets)  
            self.val_imgs = np.array(self.val_imgs)

            if self.exemplar:
                exemplar_indeces = []
                total_indeces = np.arange(len(self.val_imgs))

                for i in range(self.num_classes):
                    cls_idx = total_indeces[self.val_targets == i]
                    select_idx = np.random.choice(cls_idx, size=num_exemplar, replace=False)
                    exemplar_indeces.append(select_idx)
                exemplar_indeces = np.concatenate(exemplar_indeces, axis=0)
                self.indices = np.array(exemplar_indeces)

                self.val_imgs = self.val_imgs[self.indices]
                self.val_targets = self.val_targets[self.indices]   

            else:
                if noisy_targets is not None:
                    self.clean_val_targets = self.val_targets
                    self.val_targets = np.array(noisy_targets)                                                     
                    
    def __getitem__(self, index):       
        if self.mode=='test':
            img_path = self.val_imgs[index]
            image = Image.open( os.path.join(self.root, img_path)).convert('RGB')  
            img = self.transform(image)
            target = self.val_targets[index] 
            text = self.label_to_class_mapping[target]
            return {
                "images": img, 
                "texts": text,
                "index": str(index),
                "labels": target,
                "image_paths": img_path
                }
           
    def __len__(self):
        return len(self.val_imgs)    


def get_ilsvrc12(args, processor, mode, return_classnames=False, return_clean_label=False):

    if args.noise_type == 'symmetric':
        noisy_labels = torch.load(f'/home/dev01/data/noisy_dataset/ilsvrc12_{args.noise_type}{args.noise_level}.pt')

    image_text_dataset = ilsvrc12_dataset(args.data_location, mode=mode, noisy_targets=noisy_labels, transform=processor)
    image_text_dataloader = DataLoader(image_text_dataset,
                                    shuffle=False,
                                    sampler=None,
                                    batch_size=args.batch_size,
                                )
    
    return_list = []
    if args.exemplar:
        exemplar_dataset = ilsvrc12_dataset(args.data_location, mode=mode, transform=processor, exemplar=True, num_exemplar=args.num_exemplar+2)
        exemplar_dataloader = DataLoader(exemplar_dataset,
                                        shuffle=False,
                                        sampler=None,
                                        batch_size=args.batch_size,
                                    )
        return_list = [image_text_dataloader, exemplar_dataloader, image_text_dataset.num_classes]
    else:
        return_list = [image_text_dataloader, image_text_dataset.num_classes]

    if return_classnames:
        return_list.append((image_text_dataset.label_to_class_mapping))

    if return_clean_label:
        if mode=='test':
            return_list.append(image_text_dataset.clean_val_targets)
        else:
            raise ValueError

    return tuple(return_list)
 

