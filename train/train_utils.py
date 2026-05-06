import torch
import torch.distributed as dist
import sys
sys.path.append("..")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

from fnmatch import fnmatch
import shutil
import hashlib
import os
watched_rules = ['*.py', '*.sh', '*.yaml', '*.yml']
exclude_rules = ['results', 'datasets', 'checkpoints', 'samples', 'outputs',
                 'training-runs', 'expr', 'uda-runs', 'runs', './checkpoints', '../checkpoints']
def calculate_checksum(filenames):
    hash = hashlib.md5()
    for fn in filenames:
        if os.path.isfile(fn):
            hash.update(open(fn, "rb").read())
    return hash.hexdigest()

def copy_src_files(files, target_dir):
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    if len(files) >= 500:
        print('Warning! there are %d files to be copied!' %(len(files)))
    for file in files:
        target_name = os.path.join(target_dir, file)
        dir_name = os.path.dirname(target_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # will create all intermediate-level directories
        shutil.copyfile(file, target_name)


def _get_watched_files(work_dir):
    rules = watched_rules
    watched_files = []
    to_match = []
    for rule in rules:
        t = rule.count('*')
        if t == 0:
            watched_files.append(rule)
        elif t == 1:
            to_match.append(rule)

    for parent, dirs, file_names in os.walk(work_dir):
        for ignore_ in exclude_rules:
            dirs_to_remove = [d for d in dirs if fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            file_names = [f for f in file_names if not fnmatch(f, ignore_)]

        for file_name in file_names:
            for each in to_match:
                if fnmatch(file_name, each):
                    watched_files.append(os.path.join(parent, file_name))
                    break
    return watched_files

import os
import re
def get_run_id(outdir):
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    return cur_run_id

def prepare_sub_directories(run_dir):

    src_dir = os.path.join(run_dir, 'src')
    files = _get_watched_files('..')
    copy_src_files(files, src_dir)


class LossManager:
    def __init__(self, file_path="loss.txt", log_batch_size=10):
        self.file_path = file_path
        self.log_batch_size = log_batch_size
        self.log_buffer = []
        self.f = open(self.file_path, "a")

    def log(self, content):
        self.f.write(content+'\n')
        self.f.flush()

@torch.no_grad()
def eval_coco(model, preprocess):
    import sys
    sys.path.append('../..')
    from model import longclip
    import torch
    from torchvision.datasets import CocoCaptions
    from tqdm import tqdm
    from PIL import Image

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #_, preprocess = longclip.load("checkpoints/lam0.0/missclip_16_epochpretrain.pt", device=device)
    #del _
    device = model.device

    model.eval()


    coco = CocoCaptions(root="../datasets/coco/val2017/", annFile="../datasets/coco/annotations/captions_val2017.json",
                        transform=None)

    dataloader = torch.utils.data.DataLoader(coco, batch_size=1000, shuffle=False, num_workers=4, drop_last=False)
    image_features = []
    text_features = []
    pred_true = 0
    rank = torch.distributed.get_rank()

    with torch.no_grad():
        for image, captions in tqdm(coco, disable=(rank!=0)):
            image_input = preprocess(image).unsqueeze(0).to(device)
            image_features.append(model.module.encode_image(image_input))

            captions = captions[0:5]
            caption_input = longclip.tokenize(captions).to(device)
            text_features.extend(model.module.encode_text(caption_input))

        image_features = torch.stack(image_features).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        #print(image_features.shape)
        text_features = torch.stack(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = image_features.squeeze() @ text_features.squeeze().T
        assert len(image_features) == 5000
        #print("I2T")
        result_dict = {}
        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        result_dict['image2text_R1'] = pred_true / 5000
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        result_dict['image2text_R5'] = pred_true / 5000
        pred_true = 0

        for i in range(5000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            for j in range(5):
                true_index = 5 * i + j
                if true_index in b:
                    pred_true = pred_true + 1
                    break
        #print(pred_true / 5000)
        result_dict['image2text_R10'] = pred_true / 5000
        pred_true = 0

        #print("T2I")
        similarity = similarity.T
        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-1:]
            true_index = i // 5
            if true_index in b:
                pred_true = pred_true + 1

        #print(pred_true / 25000)
        result_dict['text2image_R1'] = pred_true / 25000
        pred_true = 0

        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-5:]
            true_index = i // 5
            if true_index in b:
                pred_true = pred_true + 1

        #print(pred_true / 25000)
        result_dict['text2image_R5'] = pred_true / 25000
        pred_true = 0

        for i in range(25000):
            pred = similarity[i]
            b = pred.argsort()[-10:]
            true_index = i // 5
            if true_index in b:
                pred_true = pred_true + 1

        #print(pred_true / 25000)
        result_dict['text2image_R10'] = pred_true / 25000
        #print(result_dict)
        return result_dict

