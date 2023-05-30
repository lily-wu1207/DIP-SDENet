# -*- coding: utf-8 -*-

import torch.utils.data as data
import os
from PIL import Image
import json
import torch
import numpy as np
import random
import torch
from torchvision import transforms
from util.FSC147 import transform_train
from util.FSC147 import transform_pre_train

class NormalSample(object):
    def __init__(self):

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
                mean=[0.56347245, 0.50660025, 0.45908741],
                std=[0.28393339, 0.2804536 , 0.30424776]
            )

    def __call__(self, image, dotmap=None):
        image = self.totensor(image)
        image = self.normalize(image)

        if dotmap is None:
            return image
        else:
            dotmap = torch.from_numpy(dotmap).float()
            return image, dotmap

jpg2id = lambda x: x.replace('.jpg', '')

class FSC147(data.Dataset):
    def __init__(self, root_path, t):
        super().__init__()
        self.t = t

        self.TransformTrain = transform_train(root_path)

        self.TransformPreTrain = transform_pre_train(root_path)

        with open(os.path.join(root_path, 'annotation_FSC147_384.json')) as f:
            self.annotations = json.load(f)
        with open(os.path.join(root_path, 'Train_Test_Val_FSC_147.json')) as f:
            data_split = json.load(f)
        class_dict = {}
        with open(os.path.join(root_path, 'ImageClasses_FSC147.txt')) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                class_dict[key] = val
        self.img = data_split['train']
        random.shuffle(self.img)
        self.im_dir = os.path.join(root_path, 'images_384_VarV2')
        self.gt_dir = os.path.join(root_path, 'gt_density_map_adaptive_384_VarV2')
        self.root_path = root_path
        self.scale_number = 40  # 把尺寸分为20类

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        im_id = self.img[idx]
        anno = self.annotations[im_id]
        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        w, h = image.size
        image.load()
        bboxes = anno['box_examples_coordinates']
        scale_embedding = []
        rects = list()
    
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])
           
            scale = ((x2 - x1) / w) * ((y2 - y1) / h)  # 面积的占比
            scale = scale // (0.5 / self.scale_number)
            scale = scale if scale < self.scale_number - 1 else self.scale_number - 1
            
            scale_embedding.append(int(scale))
        mean_scale = int(sum(scale_embedding) / len(scale_embedding))
            # mean_scale = sum(scale_embedding) / len(scale_embedding)  # 求平均


        dots = np.array(anno['points'])

        density_path = os.path.join(self.gt_dir, (im_id.split(".jpg")[0] + ".npy"))
        density = np.load(density_path).astype('float32')

        if(self.t == "pretrain"):
            sample = {'image': image, 'lines_boxes': rects, 'gt_density': density}
            sample = self.TransformPreTrain(sample)
            return sample['image']
        elif(self.t == "train"):
            m_flag = 0

            sample = {'image': image, 'lines_boxes': rects, 'gt_density': density, 'dots': dots, 'id': im_id,
                      'm_flag': m_flag, 'scale_embed': torch.tensor([mean_scale])}
            # print('sample ',sample['scale_embed'])
            sample = self.TransformTrain(sample)
            # print(sample['scale_embed'])
            return sample['image'], sample['gt_density'], sample['boxes'], sample['m_flag'], sample['scale_embed']


