import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random
from torchvision.utils import save_image

data4v_root = '../datasets/ShareGPT4V/'
json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'
image_root = '../datasets/ShareGPT4V/'

class share4v_train_dataset(data.Dataset):
    def __init__(self, data4v_root=data4v_root, json_name=json_name, image_root=image_root):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[self.total_len:]
        _, self.preprocess = clip.load("ViT-L/14")
        del _
        print('share4v_train_dataset loaded, total length:', len(self.json_data))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        num_sentences = len(caption.split(". "))
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name).convert('RGB')
        image_tensor = self.preprocess(image)
        use_caption = '. '.join(caption.split(". ")[:random.randint(1, num_sentences)])
        return image_tensor, use_caption


class share4v_val_dataset(data.Dataset):
    def __init__(self, data4v_root=data4v_root, json_name=json_name, image_root=image_root):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:self.total_len]
        _, self.preprocess = clip.load("ViT-L/14")
        del _

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption



if __name__ == '__main__':
    dataset = share4v_val_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    for i, (image_tensor, caption) in enumerate(dataloader):
        print(i, caption)
        save_image(image_tensor, 'sharegpt4v_{i}.jpg')
        break
