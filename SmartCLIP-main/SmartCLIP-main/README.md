# Smart-CLIP
This repository is the official implementation of Smart-CLIP (CVPR2025, Highlight)

**[SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_SmartCLIP_Modular_Vision-language_Alignment_with_Identification_Guarantees_CVPR_2025_paper.pdf)**\
Shaoan Xie*, Lingjing Kong*, Yujia Zheng, Yu Yao, Zeyu Tang, Eric P. Xing, Guangyi Chen, Kun Zhang


## TLDR
-🔥[**SmartCLIP**](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_SmartCLIP_Modular_Vision-language_Alignment_with_Identification_Guarantees_CVPR_2025_paper.pdf) provides a mask-based solution to improve CLIP training with long and short texts.

## 🛠️ Usage

### Installation

Our model is based on [CLIP](https://github.com/openai/CLIP), please prepare environment for CLIP.


### how to use

Please first clone our repo from github by running the following command.

```shell
git clone https://github.com/MidPush/SmartCLIP.git
cd SmartCLIP
```

Then download our trained models
```shell
mkdir checkpoints
wget https://huggingface.co/Shaoan/SmartCLIP/resolve/main/smartclip_l14.pt
wget https://huggingface.co/Shaoan/SmartCLIP/resolve/main/smartclip_b16.pt
mv smartclip_l14.pt checkpoints/
mv smartclip_b16.pt checkpoints/
```
  


```python
from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/smartclip_l14.pt", device=device)

text = longclip.tokenize(["A cat is holding a yellow sign", "A dog is holding a yellow sign"]).to(device)
image = preprocess(Image.open("./assets/cat.webp")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image = image_features @ text_features.T
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) 
```

### Evaluation
To run text-image retrieval on COCO2017 or Flickr30k, run the following command after preparing the data
```shell
cd eval/retrieval
python coco.py --checkpoint=../../checkpoints/smartclip_l14.pt                #COCO2017
python coco.py --checkpoint=../../checkpoints/smartclip_b16.pt                #COCO2017
```
### Training
Please refer to `train/train.md` for training details.


## Citation
If you find our work helpful for your research, please consider giving a citation:
```
@inproceedings{xie2025smartclip,
  title={SmartCLIP: Modular Vision-language Alignment with Identification Guarantees},
  author={Xie, Shaoan and Lingjing, Lingjing and Zheng, Yujia and Yao, Yu and Tang, Zeyu and Xing, Eric P and Chen, Guangyi and Zhang, Kun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29780--29790},
  year={2025}
}
```

## Acknowledgements
Our code is heavily borrowed from [LongCLIP](https://github.com/beichenzbc/Long-CLIP/tree/main).
