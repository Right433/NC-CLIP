import os

import torch
from torchvision import datasets
import xml.etree.ElementTree as ET
from PIL import Image
class ImageNetCategory():
    """
        For ImageNet-like directory structures without sessions/conditions:
        .../{category}/{img_name}
    """

    def __init__(self):
        pass

    def __call__(self, full_path):
        img_name = full_path.split("/")[-1]
        category = full_path.split("/")[-2]
        return category


class ImageNetDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, data_dir, type='V2', transform=None):
        super(ImageNetDataset, self).__init__(root=data_dir, transform=transform)
        self.type = type

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        sample, target = super(ImageNetDataset, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        if self.type in ['O','A','R']:
            new_target = ImageNetCategory()(path)
            new_target = get_label(new_target)
        elif self.type == 'V2':
            new_target = torch.tensor(int(ImageNetCategory()(path)))
        original_tuple = (sample, new_target)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class KaggleImageNetDataset(torch.utils.data.Dataset):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, data_dir, preprocess):
        super().__init__()
        self.data_dir = data_dir
        self.image_dir = (os.path.join(data_dir, "Data/CLS-LOC/val"))
        self.xml_dir = (os.path.join(data_dir, "Annotations/CLS-LOC/val"))
        self.imgs = [os.path.join(self.image_dir, img) for img in sorted(os.listdir(self.image_dir))]
        self.xmls = [os.path.join(self.xml_dir, xml) for xml in sorted(os.listdir(self.xml_dir))]
        self.preprocess = preprocess
        assert len(os.listdir(self.image_dir)) == len(os.listdir(self.xml_dir)) == 50000
    def __len__(self):
        return len(os.listdir(self.image_dir))

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):

        # the image file path
        path = self.imgs[index]
        sample = self.preprocess(Image.open(path).convert("RGB"))
        tree = ET.parse(self.xmls[index])
        root = tree.getroot()
        # Find the first <object><name> tag and get its value
        object_name = root.find('object/name').text
        new_target = get_label(object_name)
        original_tuple = (sample, new_target)
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class ImageNetClipDataset(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """
    SOFT_LABELS = "soft_labels"
    HARD_LABELS = "hard_labels"

    def __init__(self, label_type, mappings, *args, **kwargs):
        self.label_type = label_type
        self.clip_class_mapping = mappings
        super(ImageNetClipDataset, self).__init__(*args, **kwargs)

    def _get_new_template_hard_labels(self, image_path):
        file_name = os.path.basename(image_path)
        target_class = self.clip_class_mapping[file_name]
        target_index = self.class_to_idx[target_class]
        return target_index

    def _get_new_template_soft_labels(self, image_path):
        file_name = os.path.basename(image_path)
        target_class = self.clip_class_mapping[file_name]
        return target_class

    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageNetClipDataset, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        if self.label_type == ImageNetClipDataset.HARD_LABELS:
            new_target = self._get_new_template_hard_labels(path)
        elif self.label_type == ImageNetClipDataset.SOFT_LABELS:
            new_target = self._get_new_template_soft_labels(path)
        else:
            new_target = target
        new_target = get_label(new_target)
        original_tuple = (sample, new_target,)
        return original_tuple

def get_label(fold_name):
    with open("categories.txt", "r", encoding='utf-8') as f:
        data = f.readlines()
        #print(len(data))
        for i in range(len(data)):
            if data[i][:9] == fold_name:
                return torch.tensor([i])
        print(data[i], ' >>>>>>errorr ', fold_name)



def data_loader(transform, args):
    if args.type == 'standard':
        imagenet_data = KaggleImageNetDataset(args.data_dir, transform)
    elif args.type == 'O':
        imagenet_data = ImageNetDataset('../../datasets/imagenet-o', type=args.type, transform=transform)
    elif args.type == 'A':
        imagenet_data = ImageNetDataset('./imagenet-a', type=args.type, transform=transform)
    elif args.type == 'R':
        imagenet_data = ImageNetDataset('./imagenet-r', type=args.type, transform=transform)
    elif args.type == 'V2':
        imagenet_data = ImageNetDataset('../../datasets/imagenetv2-top-images-format-val', type=args.type,transform=transform)
    data_loader = torch.utils.data.DataLoader(
        imagenet_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    return data_loader, imagenet_data

if __name__ == "__main__":
    print(get_label("n03584254"))