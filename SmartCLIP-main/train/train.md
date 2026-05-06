# Smart-CLIP Training
To run the training code for SmartCLIP, please follow the following step.

### 1. Prepare CLIP Model
First, download the checkpoints of OpenAI CLIP Model. You can refer to this page https://github.com/openai/CLIP.

Then, you can load the model from CLIP by running the following command. The positional embedding will be stretched from 77 to 248. 
```python
from model import longclip
model, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu')
```
### 2. Prepare ShareGPT4V dataset

First, download all images we used.
- LAION-CC-SBU-558K: [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- SAM: [images](https://ai.meta.com/datasets/segment-anything-downloads/). We only use 000000~000050.tar for now. (SAM images are very large and can be the bottleneck of the training speed. But I found that resizing them with PIL can cause performance drop.) 

(Optional) Therefore, if you are using non-SSD (e.g., NFS storage), it would be much faster to pre-process the images with CLIP preprocessing and store them with HDF5 files. 

Then, download the long caption of these image [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json)


Finally, organize the data as follows in `../datasets/ShareGPT4V`:

```none
ShareGPT4V
├── ...
├── data
|   ├── share-captioner_coco_lcs_sam_1246k_1107.json
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
```
Then, change the data root in `sharegpt4v.py`

### 3. Prepare COCO validation dataset
You can download the COCO2017 validation dataset from [here](http://images.cocodataset.org/zips/val2017.zip) and place it under `../datasets/coco`.

Then, change the data root in `train/train_utils.py`.

### 4. Finetune

Finally, you can run the `train.py` for fine-tuning.

We provide two scripts for training:

- `train_b16.sh`: For training with the ViT-B/16.
- `train_l14.sh`: For training with the ViT-L/14.

