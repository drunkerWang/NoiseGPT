from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from mydatasets.imagenet_classnames import get_classnames

class webvision_dataset(Dataset): 
    def __init__(self, root_dir, transform, mode, pred=[], probability=[], log='', exemplar=False, num_exemplar=None, noisy_targets=None): 
        self.root = root_dir+'/webvision/'
        self.transform = transform
        self.mode = mode  
        self.label_to_class_mapping = get_classnames('openai')
        self.num_classes = len(self.label_to_class_mapping)
        print(self.num_classes)

        self.exemplar = exemplar
     
        if self.mode=='test':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            self.val_targets = []
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<self.num_classes:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target
                    self.val_targets.append(target)
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
        else:    
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_classes:
                    train_imgs.append(img)
                    self.train_labels[img]=target            
            if self.mode == 'all':
                self.train_imgs = train_imgs
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                
                    self.probability = [probability[i] for i in pred_idx]            
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))            
                    log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    log.flush()                          
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                    self.train_imgs = [train_imgs[i] for i in pred_idx]                           
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))             
                    
    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path] 
            prob = self.probability[index]
            image = Image.open(self.root+img_path).convert('RGB')    
            img1 = self.transform(image) 
            img2 = self.transform(image) 
            return img1, img2, target, prob              
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(self.root+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index        
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_targets[index]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image) 
            if target >= len(self.label_to_class_mapping) or target < 0:
                raise IndexError(f"Target {target} is out of range for mapping with length {len(self.label_to_class_mapping)}, the index is {index}, the img_path is {img_path}")
            text = self.label_to_class_mapping[target]
            return {
                "images": img, 
                "texts": text,
                "index": str(index),
                "labels": target,
                "image_paths": img_path
                }
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)    


def get_webvision(args, processor, mode, return_classnames=False, return_clean_label=False):

    if args.noise_type == 'symmetric':
        noisy_labels = torch.load(f'/home/dev01/data/noisy_dataset/webvision_{args.noise_type}{args.noise_level}.pt')

    image_text_dataset = webvision_dataset(args.data_location, mode=mode, noisy_targets=noisy_labels, transform=processor)
    image_text_dataloader = DataLoader(image_text_dataset,
                                    shuffle=False,
                                    sampler=None,
                                    batch_size=args.batch_size,
                                )
    
    return_list = []
    if args.exemplar:
        exemplar_dataset = webvision_dataset(args.data_location, mode=mode, transform=processor, exemplar=True, num_exemplar=args.num_exemplar+2)
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