import os
import csv
import torch
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils.bd_utils import apply_trigger
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, options=None):
        self.root = root
        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.options = options
        self.add_backdoor = options.add_backdoor
        self.target_label = options.target_label


    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, patch_size=16, patch_type='blended', patch_location='blended'):
        return apply_trigger(image, patch_size, patch_type, patch_location)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

        if self.add_backdoor:
            if self.options.backdoor_type == 'issba':

                image = Image.open(os.path.join(self.root+"/issba_bd", self.images[idx].replace('.JPEG', '_hidden.JPEG'))).convert('RGB')

            else:
                image = self.add_trigger(image, patch_size=self.options.patch_size, patch_type=self.options.patch_type,
                                     patch_location=self.options.patch_location)
            label = self.target_label
        else:
            label = self.labels[idx]

        image = self.transform(image)

        return image, label

#
# class ImageCaptionDataset(Dataset):
#     def __init__(self, path, image_key, caption_key, delimiter, processor, inmodal=False, defense=False, crop_size=150):
#         df = pd.read_csv(path, sep=delimiter)
#
#         self.root = os.path.dirname(path)
#         self.images = df[image_key].tolist()
#         self.captions_text = df[caption_key].tolist()
#         self.captions = processor.process_text(self.captions_text)
#         self.processor = processor
#
#         self.inmodal = inmodal
#         if (inmodal):
#             self.augment_captions = processor.process_text(
#                 [_augment_text(caption) for caption in df[caption_key].tolist()])
#
#         self.defense = defense
#         if self.defense:
#             self.crop_transform = transforms.RandomCrop((crop_size, crop_size))
#             self.resize_transform = transforms.Resize((224, 224))
#
#         if 'is_backdoor' in df:
#             self.is_backdoor = df['is_backdoor'].tolist()
#         else:
#             self.is_backdoor = None
#
#         logging.debug("Loaded data")
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         item = {}
#         item["image_path"] = self.images[idx]
#         image = Image.open(os.path.join(self.root, self.images[idx]))
#         item["is_backdoor"] = 'backdoor' in self.images[idx] if not self.is_backdoor else self.is_backdoor[idx]
#         item["caption"] = self.captions_text[idx]
#
#         if (self.inmodal):
#             item["input_ids"] = self.captions["input_ids"][idx], self.augment_captions["input_ids"][idx]
#             item["attention_mask"] = self.captions["attention_mask"][idx], self.augment_captions["attention_mask"][idx]
#             item["pixel_values"] = self.processor.process_image(image), self.processor.process_image(
#                 _augment_image(os.path.join(self.root, self.images[idx])))
#         else:
#             item["input_ids"] = self.captions["input_ids"][idx]
#             item["attention_mask"] = self.captions["attention_mask"][idx]
#             item["pixel_values"] = self.processor.process_image(image)
#
#         return item

def get_test_dataloader(options, transform=None):
    print(f'Test: {options.add_backdoor}')
    transform = Compose(
        [Resize(224, interpolation=Image.BICUBIC),
         CenterCrop(224),
         ToTensor(), Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    if (options.dataset == "caltech101"):
        dataset = ImageLabelDataset(root=options.data_path, transform=transform, options=options)
    elif (options.dataset == "food101"):
        dataset = torchvision.datasets.Food101(root=os.path.dirname(options.data_path), download=True,
                                               split="test", transform=transform)
    elif (options.dataset == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root=os.path.dirname(options.data_path), download=True,
                                             split="test", transform=transform)
    elif (options.dataset == "oxford_pets"):
        dataset = ImageLabelDataset(root=options.data_path, transform=transform, options=options)
    elif (options.dataset == "imagenet"):
        dataset = ImageLabelDataset(root=options.data_path, transform=transform, options=options)
    else:
        raise Exception(f"Eval test dataset type {options.data_path} is not supported")



    dataloader = DataLoader(dataset, batch_size=options.batch_size, num_workers=options.num_workers,
                                             sampler=None, shuffle=False)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader

