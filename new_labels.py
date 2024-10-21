import torch
import random
import argparse
import os
import templates as templates
from mydatasets.get_webvision import webvision_dataset
from mydatasets.get_ilsvrc12 import ilsvrc12_dataset
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'imagenet', 'cifar100', 'webvision', 'ilsvrc12'])
parser.add_argument("--noise_type", type=str, default=None, choices=['symmetric', 'asymmetric'])
parser.add_argument("--noise_level", type=float, default=None)
parser.add_argument("--data_location", type=str, default="/home/dev01/data/dataset")
args = parser.parse_args()
text_template_mapping = {
    'mnist': 'mnist_template',
    'cifar10': 'cifar_template',
    'cifar100': 'cifar_template',
    'iwildcam': 'iwildcam_template',
    'imagenet': 'openai_imagenet_template',
    'domainbed': 'cifar_template',
}

def get_noisy_cifar10(args):

    transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

    # use noisy labels instead ground truth labels to for MLLM
    noise_file = torch.load(args.data_location+'/cifar-10-batches-py/CIFAR-10_human.pt')
    clean_labels = noise_file['clean_label']

    # if args.gen_noise == 'symmetric':
    #     noisy_labels = clean_labels
    #     if args.noise_level is None:
    #         raise ValueError 
        
    #     num_label = len(clean_labels)
    #     num_noise = int(num_label*args.noise_level)
    #     noise_idx = random.sample(range(0, num_label), num_noise)
    #     # print(noise_idx)
    #     for i in range(num_noise):
    #         label_i = noisy_labels[noise_idx[i]]
    #         noise_i = label_i
    #         while noise_i==label_i:
    #             noise_i = random.choice(range(0, 10))
    #         # print(label_i, noise_i)
    #         noisy_labels[noise_idx[i]] = noise_i
    #     if args.save_noisy_labels:
    #         torch.save(noisy_labels, f'/home/wanghaoyu/data/noisy_dataset/CIFAR-100_{args.noise_type}-{args.noise_level}.pt')
    # elif args.gen_noise == 'asymmetric':
    #     pass


    noisy_labels = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(args.noise_level*50000)            
    noise_idx = idx[:num_noise]

    for i in range(50000):
        if i in noise_idx:
            if args.noise_type=='symmetric':
                noiselabel = random.randint(0,9)
                noisy_labels.append(noiselabel)
            elif args.noise_type=='asymmetric':   
                noiselabel = transition[clean_labels[i]]
                noisy_labels.append(noiselabel)                    
        else:    
            noisy_labels.append(clean_labels[i])   

    if not os.path.exists(f'/home/dev01/data/noisy_dataset/CIFAR-10_{args.noise_type}{args.noise_level}.pt'):
        print("saving noisy labels ...")
        torch.save(noisy_labels, f'/home/dev01/data/noisy_dataset/CIFAR-10_{args.noise_type}{args.noise_level}.pt')   

    pass

def get_noisy_cifar100(args):
    
    # use noisy labels instead ground truth labels to for VLM to figure out
    noise_file = torch.load(args.data_location+'/cifar-100-python/CIFAR-100_human.pt')
    clean_labels = noise_file['clean_label']

    # if args.gen_noise == 'symmetric':
    #     noisy_labels = clean_labels
    #     if args.noise_level is None:
    #         raise ValueError 
        
    #     num_label = len(clean_labels)
    #     num_noise = int(num_label*args.noise_level)
    #     noise_idx = random.sample(range(0, num_label), num_noise)
    #     # print(noise_idx)
    #     for i in range(num_noise):
    #         label_i = noisy_labels[noise_idx[i]]
    #         noise_i = label_i
    #         while noise_i==label_i:
    #             noise_i = random.choice(range(0, 100))
    #         # print(label_i, noise_i)
    #         noisy_labels[noise_idx[i]] = noise_i
    #     if args.save_noisy_labels:
    #         torch.save(noisy_labels, f'/home/wanghaoyu/data/noisy_dataset/CIFAR-100_{args.noise_type}-{args.noise_level}.pt')
    # elif args.gen_noise == 'asymmetric':
    #     pass

    noisy_labels = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(args.noise_level*50000)            
    noise_idx = idx[:num_noise]

    for i in range(50000):
        if i in noise_idx:
            if args.noise_type=='symmetric':
                noiselabel = random.randint(0,99)
                noisy_labels.append(noiselabel)
            elif args.noise_type=='asymmetric':   
                print('no such noise for CIFAR-100!')                   
        else:    
            noisy_labels.append(clean_labels[i])   

    if not os.path.exists(f'/home/dev01/data/noisy_dataset/CIFAR-100_{args.noise_type}{args.noise_level}.pt'):
        print("saving noisy labels ...") 
        torch.save(noisy_labels, f'/home/dev01/data/noisy_dataset/CIFAR-100_{args.noise_type}{args.noise_level}.pt')

    pass

def get_noisy_webvision(args):
    transform = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    image_text_dataset = webvision_dataset(args.data_location, mode='test', transform=transform)
     
    # use noisy labels instead ground truth labels to for MLLM
    clean_labels = image_text_dataset.val_targets

    noisy_labels = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(args.noise_level*50000)            
    noise_idx = idx[:num_noise]

    for i in range(50000):
        if i in noise_idx:
            if args.noise_type=='symmetric':
                noiselabel = random.randint(0,1000)
                noisy_labels.append(noiselabel)                
        else:    
            noisy_labels.append(clean_labels[i])   

    if not os.path.exists(f'/home/dev01/data/noisy_dataset/webvision_{args.noise_type}{args.noise_level}.pt'):
        print("saving noisy labels ...")
        torch.save(noisy_labels, f'/home/dev01/data/noisy_dataset/webvision_{args.noise_type}{args.noise_level}.pt')   

    pass

def get_noisy_ilsvrc12(args):
    transform = transforms.Compose([
            transforms.Resize(320),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
    print('preparing dataset')
    image_text_dataset = ilsvrc12_dataset(args.data_location, mode='test', transform=transform)
     
    # use noisy labels instead ground truth labels to for MLLM
    clean_labels = image_text_dataset.val_targets

    noisy_labels = []
    idx = list(range(50000))
    random.shuffle(idx)
    num_noise = int(args.noise_level*50000)            
    noise_idx = idx[:num_noise]

    for i in range(50000):
        if i in noise_idx:
            if args.noise_type=='symmetric':
                noiselabel = random.randint(0,1000)
                noisy_labels.append(noiselabel)                
        else:    
            noisy_labels.append(clean_labels[i])   

    if not os.path.exists(f'/home/dev01/data/noisy_dataset/ilsvrc12_{args.noise_type}{args.noise_level}.pt'):
        print("saving noisy labels ...")
        torch.save(noisy_labels, f'/home/dev01/data/noisy_dataset/ilsvrc12_{args.noise_type}{args.noise_level}.pt')   

    pass


if __name__ == '__main__':
    get_noisy_dataset = locals()[f'get_noisy_{args.dataset}']
    get_noisy_dataset(args)   