#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   08 February 2019

from __future__ import absolute_import, print_function

import os.path as osp
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as vision_tf
#from .base import _BaseDataset
from util.imutils import RandomResizeLong,\
    random_crop_with_saliency, HWC_to_CHW, Normalize
import torch.nn.functional as F
#####################################
# from data.randaugment import RandAugmentMC
# from data.augmentations import *
######################################
def load_img_id_list(img_id_file):
    return open(img_id_file).read().splitlines()


def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_saliency_path(img_name, saliency_root='SALImages'):
    return os.path.join(saliency_root, img_name + '.png')


class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    """
    def __init__(self, opt,dataset, img_id_list_file, img_root, transform=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        self.transform = transform
        self.opt = opt

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img_id, img
    
class ClassificationDataset(ImageDataset):
    """
    Classification Dataset (base)
    """
    def __init__(self,opt,dataset, img_id_list_file, img_root, transform=None):
        super().__init__(dataset, img_id_list_file, img_root, transform)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label = torch.from_numpy(self.label_list[idx])
        return name, img, label
    
class ClassificationDatasetWithSaliency(ImageDataset):
    """
    Classification Dataset with saliency
    """
    def __init__(self,opt, dataset, img_id_list_file, img_root, saliency_root=None, crop_size=224, resize_size=(256, 512)):
        super().__init__(opt,dataset, img_id_list_file, img_root, transform=None)
        self.saliency_root = saliency_root
        self.crop_size = crop_size
        self.resize_size = resize_size

        self.resize = RandomResizeLong(resize_size[0], resize_size[1])
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.normalize = Normalize()

        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        saliency = Image.open(get_saliency_path(img_id, self.saliency_root)).convert("RGB")
        gth = Image.open(os.path.join("../dataset/VOC2012/SegmentationClassAug",img_id + '.png')).convert("RGB")
        ###输出saliency
#         if not os.path.exists("debug_img"):
#                 os.makedirs("debug_img")
#         img_temp = saliency.clone().detach().numpy()
#         for counter in range(img_temp.shape[0]):
#                         img_id_temp = img_id[counter]
#                         img_temp_i = img_temp[counter,:,:,:]
#                         img_temp_i = np.squeeze(img_temp_i)
# #                         print("img_temp shape {}".format(img_temp_i.shape))
#                         img_temp_i = img_temp_i.transpose(1,2,0)
# #                         print("img_temp shape {}".format(img_temp_i.shape))
#                         img_temp_i = Image.fromarray(np.uint8(img_temp_i))
#                         temp_save_pth = os.path.join("debug_img",str(img_id_temp)+"_img.png")
#                         img_temp_i.save(temp_save_pth) 
        
#         print("*********************************")
        if self.opt.split == 'train' and self.opt.used_save_pseudo:
            if self.opt.proto_rectify:
                lpsoft = np.load(os.path.join(self.opt.path_soft, img_id+'.npy'),allow_pickle=True)
                for i in lpsoft.item():
                    h,w = lpsoft.item()[i].shape
#                     print("h {} ,w {}".format(h,w))
                    break
                    
                lp_temp = np.zeros([21,h,w])
                for i in lpsoft.item():
                    lp_temp[i,:,:] = lpsoft.item()[i]
                lpsoft = lp_temp
                
                
        img_full = Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        img, saliency,param,img_full,gth,lpsoft = self.transform_with_mask(img, saliency,img_full,gth,lpsoft)
        ###输出saliency
        
        
        label = torch.from_numpy(self.label_list[idx])
        return img_id, img, saliency, label, img_full,param,lpsoft,gth

    def transform_with_mask(self, img, mask,img_full,gth,lpsoft=None):
        # randomly resize
        param = {}
        target_size = random.randint(self.resize_size[0], self.resize_size[1])
        
#         print("target_size .shape {}".format(target_size))
        img,target_shape = self.resize(img, target_size)
        gth,target_shape = self.resize(gth, target_size)
        mask,target_shape = self.resize(mask, target_size)
#         print("line 146 gth.shape {}".format(gth.size))
        param["randomresize"] = target_shape
#         img_full = self.resize(img_full, target_size)
        if lpsoft is not None:
            lpsoft = torch.from_numpy(lpsoft)
#             print("lpsoft.shape {}".format(lpsoft.shape))
            lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[img.size[1], img.size[0]], mode='bilinear', align_corners=True)[0]
        
        
        # randomly flip
        bool_hflip = False
        if random.random() > 0.5:
            img = vision_tf.hflip(img)
            img_full = vision_tf.hflip(img_full)
            gth = vision_tf.hflip(gth)
#             print("line 161 gth.shape {}".format(gth.size))
            mask = vision_tf.hflip(mask)
            bool_hflip = True
            if lpsoft is not None:
                inv_idx = torch.arange(lpsoft.size(2)-1,-1,-1).long()  # C x H x W
#                 print("lpsoft.size(2) {}".format(lpsoft.size(2)))
                lpsoft = lpsoft.index_select(2,inv_idx)
            
        param["hflip"] = bool_hflip
        # add color jitter
#         img = self.color(img)

        img = np.asarray(img)
        gth = np.asarray(gth).copy()
        gth[gth>20]=0
#         print("line 176 gth.shape {}".format(gth.shape))
        mask = np.asarray(mask)
        

        # normalize
        img = self.normalize(img)
        img_full = self.normalize(img_full)
        
        mask = mask / 255.
        img, mask,param_crop,gth = random_crop_with_saliency(img, mask, gth,self.crop_size)
#         print("line 186 gth.shape {}".format(gth.shape))
        img_full = cv2.resize(img_full, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        
        ### 对lp-soft 进行
        container_lp = np.zeros((21, self.crop_size,self.crop_size), np.float32)
        container_lp[:,param_crop["cont_top"]:param_crop["cont_top"]+param_crop["ch"], param_crop["cont_left"]:param_crop["cont_left"]+param_crop["cw"]] = \
        lpsoft[:,param_crop["img_top"]:param_crop["img_top"]+param_crop["ch"], param_crop["img_left"]:param_crop["img_left"]+param_crop["cw"]]
        
        lpsoft = container_lp
        
        
        param["randomcrop"] = param_crop
        # permute the order of dimensions
        img = HWC_to_CHW(img)
        mask = HWC_to_CHW(mask)
        img_full = HWC_to_CHW(img_full)
        gth = HWC_to_CHW(gth)
        # make tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        mask = torch.mean(mask, dim=0, keepdim=True)
        gth = torch.from_numpy(gth)
        gth = torch.mean(gth, dim=0, keepdim=True)
        img_full = torch.from_numpy(img_full)

        return img, mask, param,img_full,gth,lpsoft
    
    
class ClassificationDatasetWithSaliency_val(ImageDataset):
    """
    Classification Dataset with saliency
    """
    def __init__(self,opt, dataset, img_id_list_file, img_root):
        super().__init__(opt,dataset, img_id_list_file, img_root, transform=None)
        self.normalize = Normalize()

        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        gth = Image.open(os.path.join("../dataset/VOC2012/SegmentationClassAug",img_id + '.png'))#.convert("RGB")

        img,gth = self.transform_with_mask(img,gth)
        
        label = torch.from_numpy(self.label_list[idx])
        return img_id, img,label,gth

    def transform_with_mask(self, img,gth):
        img = np.asarray(img)
        gth = np.asarray(gth,dtype=np.int32).copy()
        gth[gth>20]=0
        # normalize
        img = self.normalize(img)

        img = HWC_to_CHW(img)
        
#         gth = HWC_to_CHW(gth)
        img = torch.from_numpy(img)
        
        gth = torch.from_numpy(gth)
#         print("gth .shape {}".format(gth.shape))
#         gth = torch.mean(gth, dim=0, keepdim=True)
        

        return img,gth



    
    
    
    
    
    
    
    
    
class VOCAug_stage1(data.Dataset):    
#     def __init__(self,opt, logger, augmentations=data_aug, split='train',**kwargs):
    def __init__(self,opt, logger, augmentations=None, split='train'):
        self.year = 2012
        #super(VOCAug_stage1, self).__init__(**kwargs)
        
        ###########################
        self.opt = opt
        #self.root = opt.tgt_rootpath
        self.root = opt.root
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = opt.n_class
        self.img_size = (2048, 1024)
        self.mean_rgb = [104.008,116.669,122.675]
        self.mean = np.array(self.mean_rgb)
        self.files = {}
        self.paired_files = {}

        #self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
#         self.annotations_base = os.path.join(
#             self.root, "gtFine", self.split
#         )
        #self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".png")) #find all files from rootdir and 
        
        ##############################
        self.gt_path = opt.gt_path
        self.pseudo_mask_warmup_path = opt.pseudo_mask_warmup_path
        self._set_files()
        self.ignore_index = 255
        self.n_classes = 21
        
        ###########################

    def _set_files(self):
#         self.root = osp.join(self.root, "VOC{}".format(self.year))
        self.image_dir_path = osp.join(self.root, 'JPEGImages')
        self.label_dir_path = osp.join(self.root, self.gt_path)
        self.pseudo_mask_warmup_dir_path = osp.join(self.root, self.pseudo_mask_warmup_path)
        
        self.datalist_file = osp.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        #print(self.datalist_file)
        self.image_ids, self.cls_labels = self.read_labeled_image_list(self.root, self.datalist_file)
        
    def _load_data(self, index):
        image_id = self.image_ids[index]
        ###
#         cls_labels = self.cls_labels[index]
#         print("image_id {}  cls_labels {}".format(image_id,cls_labels))
        ###
        
        
        image_path = osp.join(self.image_dir_path, image_id + '.jpg')
        label_path = osp.join(self.label_dir_path, image_id + '.png')
        pseudo_mask_warmup_path = osp.join(self.pseudo_mask_warmup_dir_path, image_id + '.png')
        
        
        label = Image.open(label_path)
        label = label.resize((self.opt.resize,self.opt.resize), Image.NEAREST)   
#         print("self.opt.resize {}".format(self.opt.resize))
        label = np.asarray(label, dtype=np.int32)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
#         if(self.opt.debug_point == "true"):
#             #img_temp = images_val.cpu().numpy().copy()
#                 img_temp = image.copy()
# #                 img_temp = np.squeeze(img_temp)
# #                 img_temp = img_temp.transpose(1,2,0)
#                 print("img_temp shape before resize  {}".format(img_temp.shape))
#                 img_temp = Image.fromarray(np.uint8(img_temp))
#                 temp_save_pth = os.path.join("debug_miou_tarin_before_resize",str(image_id)+"_img.png")
#                 img_temp.save(temp_save_pth)
        
        image = cv2.resize(image, dsize=(self.opt.resize,self.opt.resize))
        
        saliency_root = self.opt.saliency_root
        saliency_path = os.path.join(saliency_root, img_name + '.png')
        saliency = PIL.Image.open(get_saliency_path(img_id, self.saliency_root)).convert("RGB")
        
        
        cls_label = self.cls_labels[index]
        
        if(self.split=="train" or self.split=="train_cls"):
            pseudo_mask_warmup = Image.open(pseudo_mask_warmup_path)
            
#             if(self.opt.debug_point == "true" ):
# #                 img_temp = pseudo_mask_warmup.clone().numpy()
# #                 img_temp = img_temp.copy()
# #                 img_temp = np.squeeze(img_temp)
# #                 img_temp = img_temp.transpose(1,2,0)
# #                 print("pm shape after transform {}".format(img_temp.shape))
# #                 img_temp = Image.fromarray(np.uint8(img_temp))
#                 temp_save_pth = os.path.join("debug_pm/pm_after_open",str(image_id)+"_img.png")
#                 pseudo_mask_warmup.save(temp_save_pth)
            
            pseudo_mask_warmup = pseudo_mask_warmup.resize((self.opt.resize,self.opt.resize), Image.NEAREST)   
#             if(self.opt.debug_point == "true" ):
# #                 img_temp = pseudo_mask_warmup.clone().numpy()
# #                 img_temp = pseudo_mask_warmup.copy()
# #                 img_temp = np.squeeze(img_temp)
# #                 img_temp = img_temp.transpose(1,2,0)
# #                 print("pm shape after resize {}".format(img_temp.shape))
# #                 img_temp = Image.fromarray(np.uint8(img_temp))
#                 temp_save_pth = os.path.join("debug_pm/pm_after_resize",str(image_id)+"_img.png")
#                 pseudo_mask_warmup.save(temp_save_pth)
            
            pseudo_mask_warmup = np.asarray(pseudo_mask_warmup, dtype=np.int32)
            
#             if(self.opt.debug_point == "true" ):
# #                 img_temp = pseudo_mask_warmup.clone().numpy()
#                 img_temp = pseudo_mask_warmup.copy()
# #                 img_temp = np.squeeze(img_temp)
# #                 img_temp = img_temp.transpose(1,2,0)
#                 print("pm shape after np {}".format(img_temp.shape))
#                 img_temp = Image.fromarray(np.uint8(img_temp))
#                 temp_save_pth = os.path.join("debug_pm/pm_after_np",str(image_id)+"_img.png")
#                 img_temp.save(temp_save_pth)
            
            return image_id, image, label, pseudo_mask_warmup,cls_label,saliency
        #img = np.array(img, dtype=np.uint8)
        #lbl = np.array(lbl, dtype=np.uint8)
        
        
        #label = np.asarray(Image.open(label_path), dtype=np.int32) #if osp.exists(label_path) else np.zeros((100, 100))
        
        return image_id, image, label, cls_label
    
    def read_labeled_image_list(self, data_dir, data_list):
        #img_dir = os.path.join(data_dir, "JPEGImages")
        
        with open(data_list, 'r') as f:
            lines = f.readlines()
        img_name_list = []
        img_labels = []
        
        for line in lines:
            fields = line.strip().split()
            
            labels = np.zeros((20,), dtype=np.float32)
#             labels[0] = 1. #background
            
            for i in range(len(fields)-1):
                index = int(fields[i+1])
#                 labels[index+1] = 1.
                labels[index] = 1.
                
            #img_name_list.append(os.path.join(img_dir, image))
            img_name_list.append(fields[0])
            img_labels.append(labels)
            
        return img_name_list, img_labels
    def __getitem__(self, index):
        
        if(self.split=="train" or self.split=="train_cls"):
            image_id, img, lbl, pseudo_mask_warmup,cls_label,saliency = self._load_data(index)    ###
            
        elif(self.split=="val"):
            image_id, img, lbl, cls_label = self._load_data(index)

        img_full = img.copy().astype(np.float64)

        img_full = cv2.resize(img_full, (self.opt.rcrop_full_img, self.opt.rcrop_full_img), interpolation=cv2.INTER_LINEAR)
        img_full -= self.mean
        img_full = img_full.transpose(2, 0, 1)
        
        
        lp, lpsoft, weak_params = None, None, None
        if self.split == 'train_cls' and self.opt.used_save_pseudo:
            if self.opt.proto_rectify:

                lpsoft = np.load(os.path.join(self.opt.path_soft, image_id+'.npy'))
            
                    
        input_dict = {}
        if self.augmentations!=None:
            
            if(self.split=="train" or self.split=="train_cls"):

                img, lbl, pseudo_mask_warmup, lp, lpsoft, weak_params = self.augmentations(self.opt,image_id,img, lbl,pseudo_mask_warmup,lp, lpsoft)
    

                    
            elif(self.split=="val"):
                img, lbl, lp, lpsoft, weak_params = self.augmentations(self.opt,image_id,img, lbl, lp, lpsoft)
            img_strong, params = self.randaug(Image.fromarray(img))     #img_strong 额外多一个aug
            img_strong, _, _,_ = self.transform(img_strong, lbl)

            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        if(self.split=="train" or self.split=="train_cls"):

            
            img, lbl_, lp, pseudo_mask_warmup = self.transform(img, lbl, pseudo_mask_warmup,lp)
            input_dict['pseudo_mask_warmup'] = pseudo_mask_warmup
            

            
        elif(self.split=="val"):
            img, lbl_, lp,_ = self.transform(img, lbl, lp)

        input_dict['img'] = img

        input_dict['img_full'] = torch.from_numpy(img_full).float()
        input_dict['label'] = lbl_
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params  #full2weak
        input_dict['img_path'] = image_id
        
        input_dict['cls_label'] = cls_label
        
        input_dict['sal_label'] = sal_label

        input_dict = {k:v for k, v in input_dict.items() if v is not None}
        return input_dict
        #############################################

    def transform(self, img, lbl, pseudo_mask_warmup=None, lp=None, check=True):
        """transform

        :param img:
        :param lbl:
        """

        img = np.array(img)
        img = img.astype(np.float64)
        img -= self.mean
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        lbl = lbl.astype(int)
    
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if check and not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):   #todo: understanding the meaning 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        ####################################pseudo_mask_warmup
        if pseudo_mask_warmup is not None:
            pseudo_mask_warmup = np.array(pseudo_mask_warmup)
            pseudo_mask_warmup = pseudo_mask_warmup.astype(int)
            pseudo_mask_warmup = torch.from_numpy(pseudo_mask_warmup).float() #Lone
        ###################################
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)

            lp = torch.from_numpy(lp).long()
        return img, lbl, lp, pseudo_mask_warmup
    def __len__(self):
        return len(self.image_ids)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    dataset = VOCAug(
        root="/media/kazuto1011/Extra/VOCdevkit",
        split="train_aug",
        ignore_label=255,
        mean_bgr=(104.008, 116.669, 122.675),
        year=2012,
        augment=True,
        base_size=None,
        crop_size=513,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )
    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, (image_ids, images, labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        if i == 0:
            mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
            images += mean.expand_as(images)
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 21.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break
