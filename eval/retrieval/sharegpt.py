import json
import cv2
from PIL import Image
import sys
sys.path.append('../..')
from model import longclip
import torch
import torch.utils.data as data
import os
import numpy as np
import sys
from train.sharegpt4v import share4v_val_dataset
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = share4v_val_dataset(data4v_root='../../datasets/ShareGPT4V/', json_name='share-captioner_coco_lcs_sam_1246k_1107.json', image_root='../../datasets/ShareGPT4V/')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(args.checkpoint, device=device, args=args)
    model.eval()
    print("model done!")
    
    img_feature_list = []
    text_list_1 = []
    text_list_2 = []
    text_list = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (image, caption) in enumerate(dataset):
            text_list.append(caption)

        text_feature = longclip.tokenize(text_list, truncate=True).to(device)
        text_feature = model.encode_text(text_feature)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        for i, (image, caption) in enumerate(dataset):            
            #image = preprocess(image).unsqueeze(0).to(device)
            image = image.to(device)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            img_feature = model.encode_image(image)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        print("text 2 image")
        i = 0
        correct = 0
        total = 0
        for i in range(text_feature.shape[0]):
            text = text_feature[i]
            sim = text @ image_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)
        
        print("image to text")
        i = 0
        correct = 0
        total = 0
        for i in range(image_embeds.shape[0]):
            img = image_embeds[i]
            sim = img @ text_feature.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)

