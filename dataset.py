import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import os

suffixes = ['/*.png', '/*.jpg', '/*.bmp', '/*.tif']
from transformers import BertTokenizer

from registry import DATASET_REGISTRY


def flexible_rescale_to_zero_one(img, precision=32):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.dtype == np.uint8:
        factor = 255
    elif img.dtype == np.uint16:
        factor = 65535
    else:
        factor = 1
    img = img.astype('float{}'.format(precision)) / factor
    return np.clip(img, 0, 1)

def flip_channels(img):
    return img[..., ::-1]

@DATASET_REGISTRY.register()
class ImageDataset_XYZ(Dataset):
    def __init__(self, option, mode="train", combined=True):
        self.mode = mode
        root = os.path.join(option.data_root, "fiveK")
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        self.set1_inf_files = list()
        self.set1_text_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", set1_input_files[i][:-1] + ".png"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_inf_files.append(os.path.join(root, "Infrared","PNG/480p",  set1_input_files[i][:-1] + ".png"))
            self.set1_text_files.append(os.path.join(root, "text", set1_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        self.set2_inf_files = list()
        self.set2_text_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", set2_input_files[i][:-1] + ".png"))
            self.set2_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set2_input_files[i][:-1] + ".png"))
            self.set2_text_files.append(os.path.join(root, "text", set2_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_expert_files = list()
        self.test_inf_files = list()
        self.test_text_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(
                os.path.join(root, "input", "PNG/480p_16bits_XYZ_WB", test_input_files[i][:-1] + ".png"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_inf_files.append(os.path.join(root, "Infrared","PNG/480p", test_input_files[i][:-1] + ".png"))
            self.test_text_files.append(os.path.join(root, "text", test_input_files[i][:-1] + ".txt"))
        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files
            self.set1_inf_files = self.set1_inf_files + self.set2_inf_files
            self.set1_text_files = self.set1_text_files + self.set2_text_files

        self.encodings_dict = torch.load(os.path.join(root, 'encodings.pt'))

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_exptC =  Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])
            img_inf =  Image.open(self.set1_inf_files[index % len(self.set1_inf_files)])
        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_exptC =  Image.open(self.test_expert_files[index % len(self.test_expert_files)])
            img_inf =  Image.open(self.test_inf_files[index % len(self.test_inf_files)])


        img_input = flexible_rescale_to_zero_one(img_input)
        # img_exptC = flexible_rescale_to_zero_one(img_exptC)
        # img_inf = flexible_rescale_to_zero_one(img_inf)
        img_input = self.transform(img_input)
        img_exptC = self.transform(img_exptC)
        img_inf = self.transform(img_inf)

        if self.mode == "train":
            if np.random.random() > 0.5:
                img_input = F.hflip(img_input)
                img_exptC = F.hflip(img_exptC)
                img_inf = F.hflip(img_inf)
            a = np.random.uniform(0.6, 1.4)
            img_input = F.adjust_brightness(img_input, a)

        filename, _ = os.path.splitext(img_name)
        return {"A_input": img_input, "A_exptC": img_exptC, "A_Inf": img_inf, "img_text": self.encodings_dict[filename],
                "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


@DATASET_REGISTRY.register()
class ImageDataset_sRGB(Dataset):
    def __init__(self, option, mode="train", combined=True):
        self.mode = mode
        root = os.path.join(option.data_root, "fiveK")

        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

        file = open(os.path.join(root, 'train_input.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        self.set1_inf_files = list()
        self.set1_text_files = list()
        for i in range(len(set1_input_files)):
            self.set1_input_files.append(os.path.join(root, "input", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set1_input_files[i][:-1] + ".jpg"))
            self.set1_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set1_input_files[i][:-1] + ".png"))
            self.set1_text_files.append(os.path.join(root, "text", set1_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'train_label.txt'), 'r')
        set2_input_files = sorted(file.readlines())
        self.set2_input_files = list()
        self.set2_expert_files = list()
        self.set2_inf_files = list()
        self.set2_text_files = list()
        for i in range(len(set2_input_files)):
            self.set2_input_files.append(os.path.join(root, "input", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_expert_files.append(os.path.join(root, "expertC", "JPG/480p", set2_input_files[i][:-1] + ".jpg"))
            self.set2_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", set2_input_files[i][:-1] + ".png"))
            self.set2_text_files.append(os.path.join(root, "text", set2_input_files[i][:-1] + ".txt"))
        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_inf_files = list()
        self.test_expert_files = list()
        self.test_text_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "input", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_expert_files.append(os.path.join(root, "expertC", "JPG/480p", test_input_files[i][:-1] + ".jpg"))
            self.test_inf_files.append(os.path.join(root, "Infrared", "PNG/480p", test_input_files[i][:-1] + ".png"))
            self.test_text_files.append(os.path.join(root, "text", test_input_files[i][:-1] + ".txt"))
        if combined:
            self.set1_input_files = self.set1_input_files + self.set2_input_files
            self.set1_expert_files = self.set1_expert_files + self.set2_expert_files
            self.set1_inf_files = self.set1_inf_files + self.set2_inf_files
            self.set1_text_files = self.set1_text_files + self.set2_text_files

        # self.set1_input_files = self.set1_input_files[:200]
        # self.set1_expert_files = self.set1_expert_files[:200]
        # self.set1_inf_files = self.set1_inf_files[:200]
        # self.set1_text_files = self.set1_text_files[:200]

        self.encodings_dict = torch.load(os.path.join(root, 'encodings.pt'))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_inf = Image.open(self.set1_inf_files[index % len(self.set1_inf_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])


        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_inf = Image.open(self.test_inf_files[index % len(self.test_inf_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
        # img_input = flexible_rescale_to_zero_one(img_input)
        # img_exptC = flexible_rescale_to_zero_one(img_exptC)
        # img_inf = flexible_rescale_to_zero_one(img_inf)
        img_input = self.transform(img_input)
        img_exptC = self.transform(img_exptC)
        img_inf = self.transform(img_inf)
        if self.mode == "train":
            if np.random.random() > 0.5:
                img_input = F.hflip(img_input)
                img_exptC = F.hflip(img_exptC)
                img_inf = F.hflip(img_inf)
            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_brightness(img_input, a)
            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_saturation(img_input, a)

        filename, _ = os.path.splitext(img_name)
        return {"A_input": img_input, "A_exptC": img_exptC, "A_Inf": img_inf, "img_text": self.encodings_dict[filename],
                "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)


@DATASET_REGISTRY.register()
class PPR_ImageDataset_sRGB(Dataset):
    def __init__(self, option, mode="train"):
        self.mode = mode
        root = os.path.join(option.data_root, "ppr10K")
        file = open(os.path.join(root, 'train_aug.txt'), 'r')
        set1_input_files = sorted(file.readlines())
        self.set1_input_files = list()
        self.set1_expert_files = list()
        self.set1_inf_files = list()
        self.set1_text_files = list()
        for i in range(len(set1_input_files)):
            gqname='_'.join(set1_input_files[i][:-1].split('_')[:2])
            self.set1_input_files.append(os.path.join(root, "source_aug_6", set1_input_files[i][:-1] + ".tif"))
            self.set1_expert_files.append(os.path.join(root, option.version, gqname + ".tif"))
            self.set1_inf_files.append(os.path.join(root, "Infrared", gqname + ".png"))
        file = open(os.path.join(root, 'test.txt'), 'r')
        test_input_files = sorted(file.readlines())
        self.test_input_files = list()
        self.test_inf_files = list()
        self.test_expert_files = list()
        self.test_text_files = list()
        for i in range(len(test_input_files)):
            self.test_input_files.append(os.path.join(root, "source", test_input_files[i][:-1] + ".tif"))
            self.test_expert_files.append(os.path.join(root, option.version, test_input_files[i][:-1] + ".tif"))
            self.test_inf_files.append(os.path.join(root, "Infrared", test_input_files[i][:-1] + ".png"))
        self.encodings_dict = torch.load(os.path.join(root, 'encodings.pt'))
        self.transform = transforms.ToTensor()

        # self.set1_input_files=self.set1_input_files[:200]
        #
        # self.set1_expert_files=self.set1_expert_files[:200]
        # self.set1_inf_files=self.set1_inf_files[:200]
    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.set1_input_files[index % len(self.set1_input_files)])[-1]
            img_name = '_'.join(img_name[:-1].split('_')[:2])
            img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            # img_input =cv2.imread(self.set1_input_files[index % len(self.set1_input_files)], cv2.IMREAD_UNCHANGED)
            # img_input = Image.open(self.set1_input_files[index % len(self.set1_input_files)])
            img_inf = Image.open(self.set1_inf_files[index % len(self.set1_inf_files)])
            img_exptC = Image.open(self.set1_expert_files[index % len(self.set1_expert_files)])

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_name = '_'.join(img_name[:-1].split('_')[:2])
            img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            # img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], cv2.IMREAD_UNCHANGED)
            # img_input = Image.open(self.test_input_files[index % len(self.test_input_files)])
            img_inf = Image.open(self.test_inf_files[index % len(self.test_inf_files)])
            img_exptC = Image.open(self.test_expert_files[index % len(self.test_expert_files)])
        # img_input = flexible_rescale_to_zero_one(img_input)
        # img_exptC = flexible_rescale_to_zero_one(img_exptC)
        # img_inf = flexible_rescale_to_zero_one(img_inf)
        img_input = self.transform(img_input)
        img_exptC = self.transform(img_exptC)
        img_inf = self.transform(img_inf)
        if self.mode == "train":
            if np.random.random() > 0.5:
                img_input = F.hflip(img_input)
                img_exptC = F.hflip(img_exptC)
                img_inf = F.hflip(img_inf)
            a = np.random.uniform(0.8, 1.2)
            img_input = F.adjust_brightness(img_input, a)


        filename, _ = os.path.splitext(img_name)
        return {"A_input": img_input, "A_exptC": img_exptC, "A_Inf": img_inf, "img_text": self.encodings_dict[filename],
                "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.set1_input_files)
        elif self.mode == "test":
            return len(self.test_input_files)
