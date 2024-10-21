import os
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

# parser = argparse.ArgumentParser()
# parser.add_argument("--data_location", type=str, default="/home/wanghaoyu/data/dataset")
# parser.add_argument("--save_location", type=str, default="/home/wanghaoyu/data/distribution_exemplar/ImageNet-V")
# parser.add_argument("--dataset", type=str, default="ImageNet-V")
# parser.add_argument("--distribution_num", type=int, default=10)
# args = parser.parse_args()

# def mixup_average(args, exemplar_embeds, embed_device, weight=None):
#     if weight is None:
#         weight = 1/(args.num_exemplar+1)
#     embed_device = embed_device
    
#     distribution_exemplar = []
#     for i in range(len(exemplar_embeds)):
#         if i == 0:
#             average_image = exemplar_embeds[i].mean(dim=0)
#             # print(average_image)
#         else:
#             average_image = exemplar_embeds[i][:-1].mean(dim=0)
#             average_image = replace_query_with_weight(average_image, query_sample=exemplar_embeds[i][-1], weight=weight)
#         distribution_exemplar.append(average_image)
#     distribution_exemplar = torch.stack(distribution_exemplar)    
#     distribution_exemplar = distribution_exemplar.to(device=embed_device)
#     # print(distribution_exemplar.shape, '\t', distribution_exemplar.type)
#     return distribution_exemplar

def mixup_average(args, exemplar_embeds, embed_device, weight=None):
    embeds = exemplar_embeds
    exp_embeds = exemplar_embeds
    if weight is None:
        raise ValueError

    for i in range(args.num_retrieve):
        # print('exemplar \n', exp_embeds[2], '\n', exp_embeds[-1])
        # print('varying \n', exemplar_embeds[2], '\n', exemplar_embeds[-1])
        exp_embeds[2+i] = replace_query_with_weight(embeds[2+i], query_sample=embeds[-1], weight=weight)
    
    exp_embeds = torch.stack(exp_embeds)    
    # exp_embeds = exemplar_embeds.to(device=embed_device)
    return exp_embeds

def varying_weight_mixup(args, exemplar_embeds, embed_device):
    # print(exemplar_embeds)
    # print(exemplar_embeds[2])
    embeds = exemplar_embeds
    exp_embeds = exemplar_embeds
    weight = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

    for i in range(args.num_retrieve):
        # print(embeds[2+i])
        image_embed = embeds[2+i]
        query = exp_embeds[-1]
        num_token = image_embed.shape[0]
        num_mask = int((num_token-1) * weight[i])
        mask_idx = torch.randint(1, num_token, (num_mask,))

        for idx in mask_idx:
            image_embed[idx] = query[idx]
        image_embed[0] = query[0]   
        exp_embeds[2+i] = image_embed

        # print(exp_embeds[2+i])
     
    exp_embeds = torch.stack(exp_embeds) 
    return exp_embeds   

def mixup_kmeans(args, exemplar_embeds, embed_device, weight=None):
    if weight is None:
        weight = 1/(args.num_exemplar+1)
    embed_device = embed_device

    distribution_exemplar = []
    for i in range(len(exemplar_embeds)):
        if i == 0:
            image_data = exemplar_embeds[i].to(torch.float32).cpu().numpy()
        else:
            image_data = exemplar_embeds[i][:-1].to(torch.float32).cpu().numpy()
        # print(image_data[0].shape, image_data[0])
        num_images, height, width = image_data.shape
        pixels = image_data.reshape(-1, height*width)
        num_clusters = 1
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(pixels)

        cluster_center = kmeans.cluster_centers_.astype(np.uint8)
        reconstructed_image = cluster_center.reshape(height, width)
        reconstructed_image = torch.tensor(reconstructed_image).to(torch.bfloat16).to(device=embed_device)
        # print(reconstructed_image)
            
        if i != 0:
            dis_exem = add_query_with_weight(reconstructed_image, query_sample=exemplar_embeds[i][-1], weight=weight)
        else:
            dis_exem = reconstructed_image
        distribution_exemplar.append(dis_exem)


    distribution_exemplar = torch.stack(distribution_exemplar).to(torch.bfloat16).to(device=embed_device)
    return distribution_exemplar

def add_query_with_weight(distribution, query_sample, weight):
    image_embed = torch.add((1-weight)*distribution, weight*query_sample)
    return image_embed

def replace_query_with_weight(distribution, query_sample, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Weight should be in the range [0, 1].")
    if torch.allclose(distribution, query_sample):
        print('do not mix same sample')
    query = query_sample
    image_embed = distribution
    # print(distribution.shape)
    num_token = distribution.shape[0]
    
    num_mask = int((num_token-1) * weight)

    # mask_idx = random.sample(range(1, num_token), num_mask)
    mask_idx = torch.randint(1, num_token, (num_mask,))
    # print(mask_idx)
    for idx in mask_idx:
        image_embed[idx] = query[idx]
    # image_embed[mask_idx] = query_sample[mask_idx]
    # print(image_embed, '\n', query_sample)
    image_embed[0] = query[0]
    # print(image_embed[0])

    # image_embed[0] = torch.add((1-weight)*distribution[0], weight*query_sample[0])

    return image_embed 

# kmeans_exemplar(args=args)
# # average_exemplar(args=args)