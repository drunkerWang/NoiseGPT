import os
import sys
import json
import time
import torch
import myclip
import argparse
import mydatasets
import vlm_forward
import mixup
import templates as templates
import torch.nn.functional as F
from statistics import stdev
from transformers import CLIPModel
from myeva.eva_clip import build_eva_model_and_transforms
from model.utils import ClassificationHead, get_zeroshot_classifier
from accelerate.utils import get_balanced_memory
from accelerate import init_empty_weights, infer_auto_device_map
from get_prompts import get_exemplar_dict, Prompt_Builder
from model.instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor

import matplotlib.pyplot as plt
import torchvision
def show_images(images, texts, label):
    img = torchvision.transforms.ToPILImage()(images)

    plt.imshow(img)
    plt.axis('off')
    plt.savefig('/home/wanghaoyu/data/result/'+str(texts)+str(label)+'.png')
    print('image saved')
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="mmicl", choices=['mmicl', 'otter'])
parser.add_argument("--vit_type", type=str, default="vit-l", choices=['vit-l', 'vit-g', 'vit-b', 'rn-50'])
parser.add_argument("--model_ckpt", type=str, default=
    "/home/dev01/models/models--BleachNick--MMICL-Instructblip-T5-xxl/snapshots/ed4ddb6c60ff260c3c03ff149b7e91ce3496690e")
parser.add_argument("--clip_ckpt", type=str, default=
    "/home/dev01/models/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
parser.add_argument("--processor_ckpt", type=str, default=
    "/home/dev01/models/models--Salesforce--instructblip-flan-t5-xxl/snapshots/1a621c99c4ac000b7a4be30f78cd51040160cdc2")
parser.add_argument("--chosen_name", type=str, default='', choices=[
    'ImageNet', 'ImageNetV2', 'ImageNetA', 'ImageNetSketch', 'ImageNetR', 'ImageNetV', 'ImageNetAValClasses', 
    'ImageNetRValClasses', 'ImageNetVValClasses', 'VLCS', 'PACS', 'OfficeHome', 'DomainNet', 
    "SpawriousO2O_easy", "SpawriousO2O_medium", "SpawriousO2O_hard", "SpawriousM2M_easy", "SpawriousM2M_medium", "SpawriousM2M_hard",])
parser.add_argument("--dataset", type=str, default='imagenet', choices=['mnist', 'cifar10', 'cifar100', 'iwildcam', 'celebA', 'imagenet', 'domainbed', 'webvision', 'ilsvrc12'])
parser.add_argument("--noise_type", type=str, default='symmetric', choices=['symmetric', 'asymmetric', 'human'])
parser.add_argument("--noise_level", type=str, default='clean_label', choices=['noisy_label', 'worse_label', 'aggre_label', 'random_label1', '0.2', '0.4', '0.5', '0.8', '0.9'])
parser.add_argument('--targets', nargs='+', type=int, default=[0], help='target domain(s) (DomainBed datasets only)')
parser.add_argument("--split", type=str, default='val')
parser.add_argument('--groupby_fields', default=[
    'male', 'wearing_hat', 'smiling', 'eyeglasses', 'blond_hair', 'mustache', 'attractive', 
    'wearing_lipstick','wearing_necklace', 'wearing_necktie', 'young', 'bald', 'from_source_domain'])
parser.add_argument("--target_attribute", type=str, default='Eyeglasses')
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--exemplar", action='store_false', default=True)
parser.add_argument("--task", type=str, default="distribution_shift")
parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers per GPU.")
parser.add_argument("--data_location", type=str, default="/home/dev01/data/dataset")
parser.add_argument("--expname", type=str, default="")
parser.add_argument("--prompt_type", type=int, default=0)
parser.add_argument("--start_iteration", type=int, default=0)
parser.add_argument("--stop_iteration", type=int, default=5000)
parser.add_argument("--threshold", type=float, default=1.0)
parser.add_argument("--threshold_diag", type=float, default=2.0)
parser.add_argument("--mix_method", type=str, default='average', choices=['average', 'kmeans'])
parser.add_argument("--mix_weight", type=float, default=None)
parser.add_argument("--num_exemplar", type=int, default=10)
parser.add_argument("--num_retrieve", type=int, default=5)
parser.add_argument("--num_query", type=int, default=3)
parser.add_argument("--top_n", type=int, default=3)
parser.add_argument("--save_corrected_labels", type=str, default='True')
args = parser.parse_args()
config = InstructBlipConfig.from_pretrained(args.model_ckpt)
args.groupby_fields.remove(args.target_attribute.lower())
args.groupby_fields.append('y')
text_template_mapping = {
    'mnist': 'mnist_template',
    'cifar10': 'cifar_template',
    'cifar100': 'cifar_template',
    'iwildcam': 'iwildcam_template',
    'imagenet': 'openai_imagenet_template',
    'domainbed': 'cifar_template',
}

if 'mmicl' in args.model_type:
    with init_empty_weights():
        model = InstructBlipForConditionalGeneration(config)
        max_memory = get_balanced_memory(
            model,
            max_memory={0: '23000MB', 1: '23000MB', 2: '23000MB', 3: '23000MB'},
            dtype=None,
            low_zero=False,
        )
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["T5Block"], dtype=torch.bfloat16)   

    model = InstructBlipForConditionalGeneration.from_pretrained(
        args.model_ckpt,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        config=config,)
    model.eval()

if args.vit_type == 'vit-l':
    clip_model = CLIPModel.from_pretrained(args.clip_ckpt).to('cuda:1', dtype=torch.bfloat16)
elif args.vit_type == 'vit-g':
    eva_clip_path = args.clip_ckpt
    model_name = "EVA_CLIP_g_14"
    clip_model, _ = build_eva_model_and_transforms(model_name, pretrained=eva_clip_path)
    clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
elif args.vit_type == 'vit-b':
    clip_model, _ = myclip.load('ViT-B/16')
    clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
elif args.vit_type == 'rn-50':
    clip_model, _ = myclip.load('RN50')
    clip_model = clip_model.to('cuda:1', dtype=torch.bfloat16)
clip_model.eval()

processor = InstructBlipProcessor.from_pretrained(
    args.processor_ckpt
)

get_image_text_loader_fn = getattr(mydatasets, 'get_' + args.dataset)
if args.dataset == 'webvision' or args.dataset == 'ilsvrc12':
    image_text_dataloader, exemplar_dataloader, n_classes, classnames, clean_labels = get_image_text_loader_fn(args, processor, mode='test', return_classnames=True, return_clean_label=True)
    # print(classnames)
    assert n_classes == len(classnames)
    image_text_dataset = image_text_dataloader.dataset

    text_templates = getattr(templates, text_template_mapping['imagenet'])
    classification_head = get_zeroshot_classifier(clip_model, text_templates, classnames, vit_g=args.vit_type!='vit-l').to('cuda:1', dtype=torch.bfloat16)

    tran_mat = None
elif args.dataset != 'celebA':
    image_text_dataloader, exemplar_dataloader, n_classes, classnames, clean_labels = get_image_text_loader_fn(args, processor, return_classnames=True, return_clean_label=True)
    # print(classnames)
    assert n_classes == len(classnames)
    image_text_dataset = image_text_dataloader.dataset

    text_templates = getattr(templates, text_template_mapping[args.dataset])
    classification_head = get_zeroshot_classifier(clip_model, text_templates, classnames, vit_g=args.vit_type!='vit-l').to('cuda:1', dtype=torch.bfloat16)

    tran_mat = None
else:
    image_text_dataloader, n_classes = get_image_text_loader_fn(args, processor)
    tran_mat = None

exemplar_images_dict, exemplar_texts_dict, exemplar_logits_dict, exemplar_features_dict = get_exemplar_dict(args, exemplar_dataloader, clip_model=clip_model, vit_g=args.vit_type!='vit-l', classification_head=classification_head, return_features_dict=True, return_logits_dict=True)
prompt_builder = Prompt_Builder(args, processor, model, tran_mat, exemplar_images_dict, exemplar_texts_dict)
vlm_forward_fn = getattr(vlm_forward, 'vlm_forward_' + args.model_type)
mixup_function = getattr(mixup, 'mixup_' + args.mix_method)
varying_function = getattr(mixup, 'varying_weight_mixup')

logits_finetune=[]
clean_history = []
noisy_history = []
clean_history2 = []
noisy_history2 = []
perturb_history = []
varying_weight_history = []
correction_history = clean_labels

n, n_noise, n_detection, n_detection_sub, n_correction, n_correction_sub, n_OOD, n_OOD_sub, n_clip, clip_correct = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
for epoch, data in enumerate(image_text_dataloader):
    batch_images_learning, batch_texts, batch_y, input_paths = data['images']['pixel_values'][0].to('cuda:1', torch.bfloat16), data['texts'], data['labels'], data['image_paths']
    for each in range(batch_y.shape[0]):
        input_path=input_paths[each]
        n += 1
        # if n < 50000:
        #     continue
        images_learning, texts, y, clean_y = torch.unsqueeze(batch_images_learning[each], 0), batch_texts[each], batch_y[each], clean_labels[int(n-1)]
        # show_images(batch_images_learning[each], batch_texts[each], y)
        
        logit_dict_q = {'true': [], 'fal': [], 'all': [], 'no': []}
        varying_dict = {'true': [], 'fal': [], 'all': [], 'no': []}

        # # get the embeddings of exemplars of class i
        # # and mix exemplar up to get distribution exemplar
        exemplar_embeddings, embed_device = prompt_builder.get_distribution_exemplar(args, model, y=int(y), image_x=images_learning)
        distribution_exemplars = mixup_function(args, exemplar_embeddings, embed_device, weight=args.mix_weight)

        # # calculate the score of comparison_exemplar
        for ex in range(args.num_retrieve):
            vision_embeddings, prompts, img_mask = prompt_builder.get_inputs(args, dis_exem=distribution_exemplars, target=int(y), ex=ex)           
            logit = vlm_forward_fn(model, prompts, processor, img_mask=img_mask, image_embeds=vision_embeddings)
            sorted_logit = torch.argsort(logit, dim=-1, descending=True)
            # # MMICL is stable enough that the first two tokens are ``true`` and ``fal``, so in this case the following comments are working.
            if args.model_type == 'mmicl':
                generated_text = processor.batch_decode(sorted_logit[:, 0], skip_special_tokens=True)[0].strip()
                generated_text2 = processor.batch_decode(sorted_logit[:, 1], skip_special_tokens=True)[0].strip()
                softmax_value = torch.Tensor([logit[:, sorted_logit[0, 0]], logit[:, sorted_logit[0, 1]]]).softmax(0)
            logit_dict_q[generated_text.lower()].append(softmax_value[0].item())
            logit_dict_q[generated_text2.lower()].append(softmax_value[1].item())                
            
        logit_average_c = torch.tensor(logit_dict_q['true']).mean(dim=0)
        logit_std_c = torch.tensor([stdev(logit_dict_q['true'])])
  
        # # calculate the score of query exemplar
        vision_embeddings, prompts, img_mask = prompt_builder.get_inputs(args, target=int(y), dis_exem=distribution_exemplars)          
        logit = vlm_forward_fn(model, prompts, processor, img_mask=img_mask, image_embeds=vision_embeddings)
        sorted_logit = torch.argsort(logit, dim=-1, descending=True)
        
        # # MMICL is stable enough that the first two tokens are ``true`` and ``fal``, so in this case the following comments are working.
        if args.model_type == 'mmicl':
            generated_text = processor.batch_decode(sorted_logit[:, 0], skip_special_tokens=True)[0].strip()
            generated_text2 = processor.batch_decode(sorted_logit[:, 1], skip_special_tokens=True)[0].strip()
            softmax_value = torch.Tensor([logit[:, sorted_logit[0, 0]], logit[:, sorted_logit[0, 1]]]).softmax(0)
        logit_dict_q[generated_text.lower()].append(softmax_value[0].item())
        logit_dict_q[generated_text2.lower()].append(softmax_value[1].item())
        varying_dict[generated_text.lower()].append(softmax_value[0].item())
        varying_dict[generated_text2.lower()].append(softmax_value[1].item()) 
        logit_value_q = logit_dict_q['true'][-1]          
        # print(logit_dict_q['true'])

        logit_deviation = logit_value_q - logit_average_c
        evaluation_metric = logit_deviation / logit_std_c

        print('logits: {:.4f}/{:.4f},   \t\tscore: {:.4f}/{:.4f},   \t\t{}/{}'.format(
            logit_average_c, logit_value_q, logit_deviation.item(), evaluation_metric.item(),
            exemplar_texts_dict[clean_y], exemplar_texts_dict[int(y)]
        ))              

        perturb_history.append(logit_dict_q['true'])

        # # calculate the score of varying perturbation weight
        exemplar_embeddings, embed_device = prompt_builder.get_varying_exemplar(args, model, y=int(y), image_x=images_learning)
        varying_weight_exemplars = varying_function(args, exemplar_embeddings, embed_device)
        # print(varying_weight_exemplars[2:])
        for ex in range(9):
            vision_embeddings, prompts, img_mask = prompt_builder.get_inputs(args, target=int(y), ex=ex, dis_exem=varying_weight_exemplars)           
            logit = vlm_forward_fn(model, prompts, processor, img_mask=img_mask, image_embeds=vision_embeddings)
            sorted_logit = torch.argsort(logit, dim=-1, descending=True)
            # # MMICL is stable enough that the first two tokens are ``true`` and ``fal``, so in this case the following comments are working.
            if args.model_type == 'mmicl':
                generated_text = processor.batch_decode(sorted_logit[:, 0], skip_special_tokens=True)[0].strip()
                generated_text2 = processor.batch_decode(sorted_logit[:, 1], skip_special_tokens=True)[0].strip()
                softmax_value = torch.Tensor([logit[:, sorted_logit[0, 0]], logit[:, sorted_logit[0, 1]]]).softmax(0)
            varying_dict[generated_text.lower()].append(softmax_value[0].item())
            varying_dict[generated_text2.lower()].append(softmax_value[1].item()) 
        # print(varying_dict['true'])
        
        varying_weight_history.append(varying_dict['true'])
        if clean_y != y:
            n_noise += 1
            noisy_history.append(evaluation_metric.to(torch.float32).cpu().numpy())
            noisy_history2.append(logit_deviation.to(torch.float32).cpu().numpy())
            perturb_history[int(n-1)].append(int(1))
            varying_weight_history[int(n-1)].append(int(1))
        else:
            clean_history.append(evaluation_metric.to(torch.float32).cpu().numpy())
            clean_history2.append(logit_deviation.to(torch.float32).cpu().numpy())
            perturb_history[int(n-1)].append(int(0))
            varying_weight_history[int(n-1)].append(int(0))

        if evaluation_metric < args.threshold:
            n_detection += 1 

        if clean_y != y and evaluation_metric < args.threshold:
            n_detection_sub += 1     

        candidate_text = exemplar_texts_dict[int(y)]
        if evaluation_metric < args.threshold:
            with torch.no_grad():
                if args.vit_type!='vit-l':
                    image_features = clip_model.encode_image(images_learning.to(1, torch.bfloat16)).to('cuda:1')
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100. * image_features @ classification_head
                else:
                    image_embeds = clip_model.get_image_features(images_learning.to(1, torch.bfloat16), return_dict=True).to('cuda:1')
                    logits = classification_head(image_embeds)
            logits = logits[0,:].softmax(0)
            max_score = torch.max(logits).item()
            top_n_pred = logits.argsort(dim=-1, descending=True).tolist()
            top_n_pred = top_n_pred[:args.top_n]
            clip_pred = logits[top_n_pred].tolist()
            clip_pred = clip_pred[:args.top_n]

            n_clip += 1
            if top_n_pred[0] == clean_y:
                clip_correct += 1

            # # compare the score of top n pred to decide which is the corrected label.
            compare_logit_average = []
            compare_logit_evaluation_metric = []
            compare_evaluation_metric = []
            compare_label = []
            compare_text = []                
            for i in range(len(top_n_pred)):
                cmp_logit_dict_c = {'true': [], 'fal': [], 'no': []}
                cmp_logit_dict_q = {'true': [], 'fal': [], 'no': []}

                # # get the embeddings of exemplars of class i
                # # and mix exemplar up to get distribution exemplar
                exemplar_embeddings, embed_device = prompt_builder.get_distribution_exemplar(args, model, y=top_n_pred[i], image_x=images_learning)
                distribution_exemplars = mixup_function(args, exemplar_embeddings, embed_device, weight=args.mix_weight)

                # # calculate the score of comparison_exemplar
                for ex in range(args.num_retrieve):
                    vision_embeddings, prompts, img_mask = prompt_builder.get_inputs(args, top_n_pred=top_n_pred, round_i=i, ex=ex, dis_exem=distribution_exemplars)

                    # # calculate the score of comparison_exemplar
                    logit = vlm_forward_fn(model, prompts, processor, img_mask=img_mask, image_embeds=vision_embeddings)
                    sorted_logit = torch.argsort(logit, dim=-1, descending=True)
                    
                    # # MMICL is stable enough that the first two tokens are ``true`` and ``fal``, so in this case the following comments are working.
                    if args.model_type == 'mmicl':
                        generated_text = processor.batch_decode(sorted_logit[:, 0], skip_special_tokens=True)[0].strip()
                        generated_text2 = processor.batch_decode(sorted_logit[:, 1], skip_special_tokens=True)[0].strip()
                        softmax_value = torch.Tensor([logit[:, sorted_logit[0, 0]], logit[:, sorted_logit[0, 1]]]).softmax(0)
                    cmp_logit_dict_q[generated_text.lower()].append(softmax_value[0].item())
                    cmp_logit_dict_q[generated_text2.lower()].append(softmax_value[1].item())
                cmp_logit_average_c = torch.tensor(cmp_logit_dict_q['true']).mean(dim=0)
                cmp_logit_std_c = torch.tensor([stdev(cmp_logit_dict_q['true'])])
                

                # # calculate the score of query exemplar
                vision_embeddings, prompts, img_mask = prompt_builder.get_inputs(args, top_n_pred=top_n_pred, round_i=i, dis_exem=distribution_exemplars)      
                logit = vlm_forward_fn(model, prompts, processor, img_mask=img_mask, image_embeds=vision_embeddings)
                # print('logits', '\t', logits.shape)
                sorted_logit = torch.argsort(logit, dim=-1, descending=True)
                    
                # # MMICL is stable enough that the first two tokens are ``true`` and ``fal``, so in this case the following comments are working.
                if args.model_type == 'mmicl':
                    generated_text = processor.batch_decode(sorted_logit[:, 0], skip_special_tokens=True)[0].strip()
                    generated_text2 = processor.batch_decode(sorted_logit[:, 1], skip_special_tokens=True)[0].strip()
                    softmax_value = torch.tensor([logit[:, sorted_logit[0, 0]], logit[:, sorted_logit[0, 1]]]).softmax(0)
                cmp_logit_dict_q[generated_text.lower()].append(softmax_value[0].item())
                cmp_logit_dict_q[generated_text2.lower()].append(softmax_value[1].item())
                cmp_logit_value_q = cmp_logit_dict_q['true'][-1]    
                # print(cmp_logit_dict_q['true'])

                cmp_logit_deviation = cmp_logit_value_q - cmp_logit_average_c
                cmp_evaluation_metric = cmp_logit_deviation / cmp_logit_std_c

                compare_evaluation_metric.append(cmp_evaluation_metric)
                compare_label.append(top_n_pred[i])
                compare_text.append(exemplar_texts_dict[top_n_pred[i]])

                print('logits: {:.4f}/{:.4f},   \t\tscore: {:.4f}/{:.4f},   \t\t{}/{}'.format(
                    cmp_logit_average_c, cmp_logit_value_q, cmp_logit_deviation.item(), cmp_evaluation_metric.item(), 
                    exemplar_texts_dict[clean_y], exemplar_texts_dict[top_n_pred[i]]
                ))

            _, idx = torch.max(torch.stack(compare_evaluation_metric), dim=0)
            idx = int(idx)
            if compare_evaluation_metric[idx] >= args.threshold_diag:
                candidate_label = compare_label[idx]
                candidate_text = compare_text[idx]
                correction_history[int(n-1)] = candidate_label
                n_correction += 1      
                if candidate_label == clean_y:
                    n_correction_sub += 1                
            else:
                # print('[THERAPY]:\t This sample is OOD!') 
                n_OOD += 1

        print('iter {}, \t\tnoise {}, \t\tdetection {}-{}, \t\tcorrection {}-{}, \t\tOOD {}, \t\trelabel {}, clip: {}'.format(
            n, n_noise, n_detection, n_detection_sub, n_correction, n_correction_sub, n_OOD, candidate_text, clip_correct))

    if n >= args.stop_iteration:
        break

timestamp = time.time()
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

from scipy.io import savemat
import numpy as np

if args.model_ckpt == '/home/dev01/models/models--BleachNick--MMICL-Instructblip-T5-xxl/snapshots/ed4ddb6c60ff260c3c03ff149b7e91ce3496690e':
    model_type = 'xxl' 
else:
    model_type = 'xl'

evaluation_metric_history_dict = {
    'clean_1':np.array(clean_history), 'noisy_1': np.array(noisy_history), 'clean_2':np.array(clean_history2), 'noisy_2': np.array(noisy_history2), 'perturb':perturb_history, 'weight':varying_weight_history}
savemat(f'/home/dev01/data/result/NoiseGPT/history/inf6_history_{model_type}_{args.dataset}_{args.noise_type}{args.noise_level}_exemplar{str(args.num_exemplar)}_retrieve{(args.num_retrieve)}_weight{args.mix_weight}_iter{args.start_iteration}-{args.stop_iteration}_PreserveQueryClstkn.mat'
        , evaluation_metric_history_dict)

detection_rate = n_detection_sub/(1 if n_detection==0 else n_detection)
correction_rate = n_correction_sub/(1 if n_correction==0 else n_correction)
clip_accuracy = clip_correct / n_clip
print('Detect Acc{}\nCorrection Acc{} '.format(detection_rate, correction_rate))
print('CLIP Acc: ', clip_accuracy)

# # save logits for fine-tune
output_dict={'Detect Acc': detection_rate, 'Correction Acc': correction_rate, 'noise': n_noise, 'detect': n_detection, 'sub detect': n_detection_sub, 'correction': n_correction, 'sub correction': n_correction_sub, 'OOD': n_OOD, 'CLIP correct': clip_correct}
with open(f'/home/dev01/data/result/NoiseGPT/performance/inf6_logits2finetune_{model_type}_{args.dataset}_{args.noise_type}{args.noise_level}_exemplar{str(args.num_exemplar)}_retrieve{(args.num_retrieve)}_weight{args.mix_weight}_iter{args.start_iteration}-{args.stop_iteration}__{time_str}.json', 'w') as f:
    json.dump(output_dict, f)
    json.dump(vars(args), f)

if args.save_corrected_labels == 'True':
    print(len(correction_history))
    correction_history = torch.tensor(correction_history)
    torch.save(correction_history, f'/home/dev01/data/noisy_dataset/{args.dataset}_{args.noise_type}{args.noise_level}_weight{args.mix_weight}_iter{args.start_iteration}-{args.stop_iteration}_corrected.pt')


