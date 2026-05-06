import sys
sys.path.append('../..')
from model import longclip
import torch
from torchvision.datasets import CocoCaptions
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='')
args = parser.parse_args()


@torch.no_grad()
def get_text_feature():
    text_list = []
    feature_list = []
    with torch.no_grad():
        with open("../../datasets/flickr/results_20130124.token", 'r') as f:
            dataset = f.readlines()
            for data in dataset:
                image = data.split('\t')[0]
                text = data.split('\t')[1]
                text_list.append(text)
        len_list = len(text_list)
        print(len_list)
     
    #avoid OOM
    with torch.no_grad():
        for i in tqdm(range(40)):
            text = text_list[i*len_list//40: (i+1)*len_list//40]
            text = longclip.tokenize(text, truncate=True).to(device)
            feature_list.append(model.encode_text(text).to('cpu'))
    
    
    text_feature = torch.cat(feature_list, dim=0)
    print(len(text_feature), ' >>> len textfeatures')
    return text_feature
    
@torch.no_grad()
def get_image_feature():
    torch.cuda.empty_cache()
    text_list = []
    data_root = "../../datasets/flickr/flickr30k-images/"
    img_feature_list = []
    with torch.no_grad():
        with open("../../datasets/flickr/results_20130124.token", 'r') as f:
            dataset = f.readlines()
            data_len = len(dataset)
            for i in tqdm(range(data_len//5)):
                #1 image corresponding to 5 captions
                data = dataset[5*i]
                image_name = data.split('\t')[0][:-2]
                image = Image.open(data_root + image_name)
                image = preprocess(image).unsqueeze(0).to(device)
                img_feature = model.encode_image(image).to('cpu')
                img_feature_list.append(img_feature)
                #torch.cuda.empty_cache()
                del img_feature, image

            img_feature = torch.cat(img_feature_list, dim=0)
            return img_feature

def get_accuracy_t2i(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0
        total = 0
        sim = (text_feature @ image_feature.T).softmax(dim=-1)
        pbar = tqdm(range(text_feature.shape[0]))
        for i in pbar:
            pred = sim[i]
            values, topk = pred.topk(k)
            true_index = i//5
            if true_index in topk:
                pred_true = pred_true + 1
            total = total + 1
            pbar.set_description(f"Top {k} accuracy: {pred_true/total:.4f}")
        print(pred_true/text_feature.shape[0])

def get_accuracy_i2t(text_feature, image_feature, k):
    with torch.no_grad():
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        text_feature = text_feature.cuda()
        image_feature = image_feature.cuda()

        pred_true = 0
        total = 0
        sim = (image_feature @ text_feature.T).softmax(dim=-1)
        pbar = tqdm(range(image_feature.shape[0]))
        for i in pbar:
            pred = sim[i]
            values, topk = pred.topk(k)
            for j in range(5):
                true_index = 5*i + j
                if true_index in topk:
                    pred_true = pred_true + 1
                    break
            total = total + 1
            pbar.set_description(f"Top {k} accuracy: {pred_true/total:.4f}")

        print(pred_true/image_feature.shape[0])

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(args.checkpoint, device=device, args=args)
    model.eval()

    text_feature = get_text_feature()
    image_feature = get_image_feature()

    get_accuracy_i2t(text_feature, image_feature, 1)
    get_accuracy_i2t(text_feature, image_feature, 5)
    get_accuracy_i2t(text_feature, image_feature, 10)
    get_accuracy_t2i(text_feature, image_feature, 1)
    get_accuracy_t2i(text_feature, image_feature, 5)
    get_accuracy_t2i(text_feature, image_feature, 10)