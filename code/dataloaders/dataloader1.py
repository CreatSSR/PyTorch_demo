import json
import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data


def train_dataloader(train_data_dir, BATCH_SIZE):
    train_data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    #train_data_dir = '../../data/train'

    train_data = ImageFolder(train_data_dir, transform = train_data_transforms)
    train_data_loader = Data.DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=1)

    return train_data_loader

def val_dataloader(val_data_dir, BATCH_SIZE):
    val_data_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    #val_data_dir = '../../data/val'
    val_data = ImageFolder(val_data_dir, transform = val_data_transforms)

    val_data_loader = Data.DataLoader(val_data, BATCH_SIZE, shuffle=True, num_workers=1)

    return val_data_loader



"""
    # 字典，类别：索引 {'calling':0, 'normal':1, 'smoking':2}
    img_list = train_data.class_to_idx
    # 将 flower_list 中的 key 和 val 调换位置
    cla_dict = dict((val, key) for key, val in img_list.items())

    # 将 cla_dict 写入 json 文件中
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
"""

