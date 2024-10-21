import torch
from torch.utils.data import DataLoader
from mydatasets.image_text_cifar import image_text_CIFAR
from torch.utils.data.sampler import RandomSampler, SequentialSampler


def get_cifar10(args, processor, return_classnames=False, return_clean_label=False):
    # use noisy labels instead ground truth labels to for VLM to figure out
    noise_file = torch.load(args.data_location+'/cifar-10-batches-py/CIFAR-10_human.pt')
    clean_labels = noise_file['clean_label']

    if args.noise_type == 'human':
        noisy_labels = noise_file[args.noise_level]
    else:
        noisy_labels = torch.load(f'/home/dev01/data/noisy_dataset/CIFAR-10_{args.noise_type}{args.noise_level}.pt')

    image_text_dataset = image_text_CIFAR(args.data_location, noisy_targets=noisy_labels, train=True, transform=processor)
    print(image_text_dataset.targets, image_text_dataset.num_classes)
    # sampler = RandomSampler(image_text_dataset, replacement=True, num_samples=image_text_dataset.__len__())
    image_text_dataloader = DataLoader(image_text_dataset,
                                    shuffle=False,
                                    sampler=None,
                                    batch_size=args.batch_size,
                                )
    
    return_list = []
    if args.exemplar:
        exemplar_dataset = image_text_CIFAR(args.data_location, train=True, transform=processor, exemplar=True, num_exemplar=args.num_exemplar+2)
        # exemplar_sampler = SequentialSampler(exemplar_dataset)
        exemplar_dataloader = DataLoader(exemplar_dataset,
                                        shuffle=False,
                                        sampler=None,
                                        batch_size=args.batch_size,
                                    )
        return_list = [image_text_dataloader, exemplar_dataloader, image_text_dataset.num_classes]
    else:
        return_list = [image_text_dataloader, image_text_dataset.num_classes]

    if return_classnames:
        return_list.append((list(image_text_dataset.label_to_class_mapping.values())))

    if return_clean_label:
        return_list.append(clean_labels)

    return tuple(return_list)

