from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import json

def get_exemplar_dict(args, exemplar_dataloader, clip_model=None, vit_g=False, classification_head=None, return_features_dict=False, return_logits_dict=False):
    exemplar_texts_dict = {}
    exemplar_images_dict = {}
    exemplar_features_dict = {}
    exemplar_logits_dict = {}
    print('Emunerating all exemplars.')
    for epoch, data in enumerate(tqdm(exemplar_dataloader)):
        batch_images, batch_texts, batch_y, batch_paths = data['images']['pixel_values'][0].to(torch.bfloat16), \
            data['texts'], data['labels'], data['image_paths']

        for each in range(batch_y.shape[0]):
            images, texts, y, batch_path = torch.unsqueeze(batch_images[each], 0), batch_texts[each], batch_y[each], batch_paths[each]
            with torch.no_grad():
                if vit_g:
                    image_embeds = clip_model.encode_image(images.to('cuda:1', torch.bfloat16))
                    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
                    logits = 100. * image_embeds @ classification_head
                else:
                    image_embeds = clip_model.get_image_features(images.to('cuda:1', torch.bfloat16), return_dict=True)
                    logits = classification_head(image_embeds.to('cuda:1'))

            if y.item() not in exemplar_texts_dict.keys():
                exemplar_images_dict[y.item()] = [images.to('cuda:1')]
                if return_logits_dict:
                    exemplar_logits_dict[y.item()] = [logits.to('cuda:1')]
                if return_features_dict:
                    exemplar_features_dict[y.item()] = [image_embeds.to('cuda:1')]
            else:
                exemplar_images_dict[y.item()].append(images.to('cuda:1'))
                if return_logits_dict:
                    exemplar_logits_dict[y.item()].append(logits.to('cuda:1'))
                if return_features_dict:
                    exemplar_features_dict[y.item()].append(image_embeds.to('cuda:1'))
            exemplar_texts_dict[y.item()] = texts

    if return_features_dict:
        for y_i in exemplar_texts_dict.keys():
            exemplar_features_dict[y_i] = torch.cat(exemplar_features_dict[y_i], dim=0)
    if return_logits_dict:
        for y_i in exemplar_texts_dict.keys():
            exemplar_logits_dict[y_i] = torch.cat(exemplar_logits_dict[y_i], dim=0)
            exemplar_logits_dict[y_i] = exemplar_logits_dict[y_i].softmax(dim=1)
    
    return_list = [exemplar_images_dict, exemplar_texts_dict]
    if return_logits_dict:
        return_list.append(exemplar_logits_dict)
    if return_features_dict:
        return_list.append(exemplar_features_dict)
    # print(exemplar_texts_dict)
    return return_list


class Prompt_Builder:
    def __init__(self, args, processor, model, tran_mat, exemplar_images_dict, exemplar_texts_dict, exemplar_description_dict=None) -> None:
        image_palceholder="图"
        sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
        sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
        processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
        replace_token="".join(32*[image_palceholder])
        self.exemplar_images_dict = exemplar_images_dict
        self.exemplar_texts_dict = exemplar_texts_dict
        if exemplar_description_dict:
            self.exemplar_description_dict = exemplar_description_dict
        self.tran_mat = tran_mat
 
        if args.dataset == 'imagenet' or 'cifar10' or 'cifar100' :
            self.prompt = f'This image {replace_token} shows a photo of <#text>, True or False'        
        elif args.dataset == 'iwildcam':
            self.prompt = f'This image {replace_token} shows a wild animal photo of <#text>, True or False'
        elif args.dataset == 'mnist':
            self.prompt = f'This image {replace_token} shows a handwritten digit photo of <#text>, True or False'
        elif args.dataset == 'domainbed' and 'Spawrious' in args.chosen_name:
            self.prompt = f'This image {replace_token} shows a dog breed photo of <#text>, True or False'
        else:
            self.prompt = f'This image {replace_token} shows a photo of <#text>, True or False'
            self.prompt_mapping = {
                'male': f'Question: Is the person in this image {replace_token} a male?',
                'wearing_hat': f'Question: Is the person in this image {replace_token} wearing a hat?',
                'smiling': f'Question: Is the person in this image {replace_token} smiling',
                'eyeglasses': f'Question: Is the person in this image {replace_token} wearing eyeglasses?',
                'blond_hair': f'Question: Does the person in this image {replace_token} have blond hair?',
                'mustache': f'Question: Does the person in this image {replace_token} have mustache?',
                'attractive': f'Question: Is the person in this image {replace_token} attractive?',
                'wearing_lipstick': f'Question: Does the person in this image {replace_token} wearing lipstick?',
                'wearing_necklace': f'Question: Does the person in this image {replace_token} wearing necklace?',
                'wearing_necktie': f'Question: Does the person in this image {replace_token} wearing necktie?',
                'young': f'Question: Is the person in this image {replace_token} young?',
                'bald': f'Question: Is the person in this image {replace_token} bald?',
                }

            self.prompt_mapping_negative = {
                'male': f'Question: Is the person in this image {replace_token} a female?',
                'wearing_hat': f'Question: Is the person in this image {replace_token} not wearing a hat?',
                'smiling': f'Question: Is the person in this image {replace_token} not smiling',
                'eyeglasses': f'Question: Is the person in this image {replace_token} not wearing eyeglasses?',
                'blond_hair': f'Question: Does the person in this image {replace_token} not have blond hair?',
                'mustache': f'Question: Does the person in this image {replace_token} not have mustache?',
                'attractive': f'Question: Is the person in this image {replace_token} not attractive?',
                'wearing_lipstick': f'Question: Does the person in this image {replace_token} not wearing lipstick?',
                'wearing_necklace': f'Question: Does the person in this image {replace_token} not wearing necklace?',
                'wearing_necktie': f'Question: Does the person in this image {replace_token} not wearing necktie?',
                'young': f'Question: Is the person in this image {replace_token} not young?',
                'bald': f'Question: Is the person in this image {replace_token} not bald?',
                }
        

            self.prompt_mapping['y'] = self.prompt_mapping[args.target_attribute.lower()]
            del self.prompt_mapping[args.target_attribute.lower()]
            self.prompt_mapping_negative['y'] = self.prompt_mapping_negative[args.target_attribute.lower()]
            del self.prompt_mapping_negative[args.target_attribute.lower()]
    
    
    def retrieve_same_logit(self, args, logits, exemplar_logits_dict, target_y):
        with torch.no_grad():
            logits_value, logits_pred = torch.max(logits.view(1, -1), dim=-1)
            logits_sim = exemplar_logits_dict[target_y].to('cuda:1')[:, logits_pred] - logits_value
        return logits_sim
    

    def retrieve_sim_logit_list(self, args, top_n_pred, logits, exemplar_logits_dict):
        retrieve_idxs = []
        logits_sims = []
        for target_y in top_n_pred:
            logits_sim = self.retrieve_same_logit(args, torch.squeeze(logits), exemplar_logits_dict, target_y)
            logits_sims.append(logits_sim.view(1, -1))
        logits_sims = torch.cat(logits_sims, dim=0)
        
        for top_n_i in range(len(top_n_pred)):
            retrieve_idxs.append([])
        retrieve_idx = torch.arange(args.num_retrieve).to(logits.device)
        for sim in range(args.num_retrieve):
            for top_n_i in range(len(top_n_pred)):
                sim_argsort = torch.argsort(logits_sims[top_n_i], dim=-1, descending=False)
                retrieve_idx = retrieve_idx[sim_argsort]
                sim_i = 0
                while retrieve_idx[sim_i] in retrieve_idxs[top_n_i]:
                    sim_i += 1
                retrieve_idxs[top_n_i].append(retrieve_idx[sim_i])

        return retrieve_idxs
    

    def retrieve_similarity(self, args, query_feature, exemplar_features_dict, target_y):
        with torch.no_grad():
            similarity = torch.cosine_similarity(query_feature, exemplar_features_dict[target_y].to('cuda:1'))
        return similarity
    

    def retrieve_sim_feature_list(self, args, top_n_pred, image_embeds, exemplar_features_dict):
        retrieve_idxs = []
        similarities = []
        for target_y in top_n_pred:
            similarity = self.retrieve_similarity(args, torch.squeeze(image_embeds), exemplar_features_dict, target_y)
            similarities.append(similarity.view(1, -1))
        similarities = torch.cat(similarities, dim=0)
        sort_sim = torch.sort(similarities, dim=-1, descending=True).values

        for top_n_i in range(len(top_n_pred)):
            retrieve_idxs.append([])
        retrieve_idx = torch.arange(args.num_retrieve).to(image_embeds.device)
        for sim in range(args.num_retrieve):
            sim_avg = torch.mean(sort_sim[:, sim])
            for top_n_i in range(len(top_n_pred)):
                sim_argsort = torch.argsort(similarities[top_n_i] - sim_avg, dim=-1, descending=False)
                retrieve_idx = retrieve_idx[sim_argsort]
                sim_i = 0
                while retrieve_idx[sim_i] in retrieve_idxs[top_n_i]:
                    sim_i += 1
                retrieve_idxs[top_n_i].append(retrieve_idx[sim_i])

        return retrieve_idxs
    

    def get_noisy_classes(self, top_n_pred, return_classes=False):
        # source noise
        self.noisy_classes = torch.argsort(self.tran_mat[:, top_n_pred], dim=0, descending=True)
        # target noise
        self.noisy_classes_hat = torch.argsort(self.tran_mat[top_n_pred, :], dim=-1, descending=True)
        if return_classes:
            return self.noisy_classes, self.noisy_classes_hat

    def get_inputs(self, args, dis_exem, target=None, round_i=None, ex=None, top_n_pred=None):
        import matplotlib.pyplot as plt
        import torchvision
        prompts, images = [], []
          
        if target is not None:
            if ex is None:
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[target]) + '? Answer: True')
                images.append(dis_exem[0])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[target]) + '? Answer: False')
                images.append(dis_exem[1])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[target]) + '? Answer: ')
                images.append(dis_exem[-1])
            else:
                # print(ex)
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[target]) + '? Answer: True')
                images.append(dis_exem[0])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[target]) + '? Answer: False')
                images.append(dis_exem[1])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[target]) + '? Answer: ')
                images.append(dis_exem[2+ex])
                # print(dis_exem[2]) 
                         
        else:
            if ex is None:
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: True')
                images.append(dis_exem[0])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: False')
                images.append(dis_exem[1])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: ')
                images.append(dis_exem[-1])
            else:
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: True')
                images.append(dis_exem[0])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: False')
                images.append(dis_exem[1])
                prompts.append(self.prompt.replace('<#text>', self.exemplar_texts_dict[top_n_pred[round_i]]) + '? Answer: ')
                images.append(dis_exem[2+ex])  

        img_mask = torch.tensor([[1 for i in range(len(images))]])
        images = torch.stack(images, dim=0)
        # print(images.shape)
        # print(images.shape, images.type(), images)
        prompts = '\n'.join(prompts)
        return images, prompts, img_mask

    # def get_distribution_exemplar(self, args, model, y=None, image_x=None):
    #     num_class = len(self.exemplar_images_dict)

    #     rand_idx = random.choice(range(len(self.exemplar_images_dict[y])))
    #     positive_sample = self.exemplar_images_dict[y][rand_idx]
    #     rand_idx = random.choice(range(len(self.exemplar_images_dict[y])))
    #     negative_sample = self.exemplar_images_dict[(y+1)%num_class][rand_idx]

    #     sample_list = torch.concat((positive_sample, negative_sample), dim=0)
    #     # print(sample_list.shape)

    #     exemplar_pool = []
    #     for i in range(num_class):
    #         if i!=y:
    #             exemplar_pool.extend(self.exemplar_images_dict[i])
    #     # print(len(exemplar_pool))
    #     rand_idxs = random.sample(range(len(exemplar_pool)), args.num_retrieve)
    #     # print(rand_idxs)
    #     for idx in rand_idxs:
    #         sample_list = torch.concat((sample_list, exemplar_pool[idx]), dim=0)
    #     sample_list = torch.concat((sample_list, image_x), dim=0)
    #     # print(sample_list)

    #     pixel_values = sample_list
    #     img_mask = torch.tensor([[1 for i in range(len(sample_list))]])
    #     pixel_values = pixel_values.unsqueeze(0)
    #     with torch.no_grad():
    #         embedding_dict = model.vision_encode(
    #             pixel_values=pixel_values,
    #             img_mask=img_mask,
    #         )
    #     # print(embedding_dict)
    #     # print(embedding_dict.shape, embedding_dict.type(), embedding_dict)
    #     device = embedding_dict.device

    #     return embedding_dict, device

    def get_distribution_exemplar(self, args, model, y=None, image_x=None):
        num_class = len(self.exemplar_images_dict)

        rand_idx = random.choice(range(len(self.exemplar_images_dict[y])))
        positive_sample = self.exemplar_images_dict[y][rand_idx]
        rand_idx = random.choice(range(len(self.exemplar_images_dict[y])))
        negative_sample = self.exemplar_images_dict[(y+1)%num_class][rand_idx]

        sample_list = [positive_sample, negative_sample]

        exemplar_pool = []
        for i in range(num_class):
            if i!=y:
                exemplar_pool.extend(self.exemplar_images_dict[i])
                
        rand_idxs = random.sample(range(len(exemplar_pool)), args.num_retrieve)
        # print(rand_idxs)
        for idx in rand_idxs:
            sample_list.append(exemplar_pool[idx])
        sample_list.append(image_x)

        # print(len(sample_list))

        embedding_dict = []
        for i in range(len(sample_list)):
            pixel_values = sample_list[i]
            img_mask = torch.tensor([[1 for i in range(1)]])
            pixel_values = pixel_values.unsqueeze(0)
            with torch.no_grad():
                embedding = model.vision_encode(
                    pixel_values=pixel_values,
                    img_mask=img_mask,
                )
                embedding = embedding.squeeze(0)
            embedding_dict.append(embedding)
        # print(len(embedding_dict))
        device = embedding_dict[0].device

        return embedding_dict, device

    def get_varying_exemplar(self, args, model, y=None, image_x=None):
        num_class = len(self.exemplar_images_dict)

        rand_idx = random.choice(range(len(self.exemplar_images_dict[y])))
        positive_sample = self.exemplar_images_dict[y][rand_idx]
        rand_idx = random.choice(range(len(self.exemplar_images_dict[y])))
        negative_sample = self.exemplar_images_dict[(y+1)%num_class][rand_idx]

        sample_list = [positive_sample, negative_sample]

        exemplar_pool = []
        for i in range(num_class):
            if i!=y:
                exemplar_pool.extend(self.exemplar_images_dict[i])
                
        rand_idxs = random.sample(range(len(exemplar_pool)), 1)
        # print(rand_idxs)
        for i in range(args.num_retrieve):
            sample_list.append(exemplar_pool[rand_idxs[0]])
        sample_list.append(image_x)

        embedding_dict = []
        for i in range(len(sample_list)):
            pixel_values = sample_list[i]
            img_mask = torch.tensor([[1 for i in range(1)]])
            pixel_values = pixel_values.unsqueeze(0)
            with torch.no_grad():
                embedding = model.vision_encode(
                    pixel_values=pixel_values,
                    img_mask=img_mask,
                )
                embedding = embedding.squeeze(0)
            embedding_dict.append(embedding)
        device = embedding_dict[0].device

        return embedding_dict, device