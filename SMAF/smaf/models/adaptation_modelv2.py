# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import os, sys
import torch
import numpy as np

from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.deeplabv2 import Deeplab
from models.EPS import EPS_net

from models.discriminator import FCDiscriminator
from .utils import freeze_bn, get_scheduler, cross_entropy2d
from data.randaugment import affine_sample
from PIL import Image

from models.optimizer import get_optimizer

from models.eps_loss import get_eps_loss

from torch.nn import L1Loss, MSELoss
from torch.cuda.amp import autocast, GradScaler

class feat_prototype_distance_module(nn.Module):
    def __init__(self):
        super(feat_prototype_distance_module, self).__init__()

    def forward(self, feat, objective_vectors, class_numbers):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, class_numbers, H, W)).to(feat.device)
        feat_proto_distance = F.relu(torch.cosine_similarity(feat.unsqueeze(1), objective_vectors.view(21, C, 1, 1).unsqueeze(0), dim=2))
        
        return feat_proto_distance

class CustomModel():
    def __init__(self, opt, isTrain=True): #logger
        self.opt = opt
        self.class_numbers = opt.n_class
        self.best_iou = -100
        self.nets = []
        self.nets_DP = []
        self.nets_ema = []
        self.nets_DP_ema = []
        self.default_gpu = 0
        self.objective_vectors = torch.zeros([self.class_numbers, 128])
        self.objective_vectors_num = torch.zeros([self.class_numbers])
        self.device = None

        if opt.bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif opt.bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(opt.bn))

        if self.opt.no_resume:
            restore_from = None
        else:
            restore_from= opt.resume_path
            self.best_iou = 0
        
        self.BaseNet = EPS_net(self.opt)    
        print("EPS backbone succeed")


        self.nets.extend([self.BaseNet])
        
        ##EPS的优化器
        max_step = opt.max_iters
        self.optimizer = get_optimizer(opt, self.BaseNet, max_step)
        
        self.optimizers = []
        self.schedulers = []        

        if self.opt.ema:
            self.BaseNet_ema = EPS_net(self.opt)

            self.BaseNet_ema.load_state_dict(self.BaseNet.state_dict().copy())
            print("emd net succeeed")
            self.nets_ema.extend([self.BaseNet_ema])
        
        #########################################################
        #装入GPU
        self.feat_prototype_distance_DP = self.init_device(feat_prototype_distance_module(), gpu_id=self.default_gpu, whether_DP=False) 

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=False)
        self.nets_DP.append(self.BaseNet_DP)
        if self.opt.ema:
            self.BaseNet_ema_DP = self.init_device(self.BaseNet_ema, gpu_id=self.default_gpu, whether_DP=False)
            self.nets_DP_ema.append(self.BaseNet_ema_DP)
        ##################################################################
        
        self.BaseOpti = self.optimizer


        self.optimizers.extend([self.optimizer])
        self.BaseSchedule = get_scheduler(self.BaseOpti, opt)
        self.schedulers.extend([self.BaseSchedule])

    def get_params(self,model,key):
    # For Dilated FCN
        if key == "1x":
            for m in model.named_modules():
                if "layer" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
        if key == "10x":
            b = []
            b.append(model.aspp.parameters())

            for j in range(len(b)):
                for i in b[j]:
                    yield i

                        
    def calculate_mean_vector(self, feat_cls, outputs, labels=None, thresh=None):

        outputs_softmax = F.softmax(outputs, dim=1)
        if thresh is None:
            thresh = -1
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = conf.ge(thresh)    ##判断是否大于阈值
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels is None:
            outputs_pred = outputs_argmax
        else:

            bg = torch.ones_like(labels[:,-2:,:,:]).to(self.default_gpu)
            

            labels = torch.cat((labels,bg),dim=1)

            outputs_pred = labels * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t] * mask[n]
                
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids


    def step_adv(self, source_x, source_label):
        if self.opt.S_pseudo_src > 0:
            source_output = self.BaseNet_DP(source_imageS)
            source_label_d4 = F.interpolate(source_label.unsqueeze(1).float(), size=source_output['out'].size()[2:])
            source_labelS = self.label_strong_T(source_label_d4.clone().float(), source_params, padding=255, scale=4).to(torch.int64)
            loss_ = cross_entropy2d(input=source_output['out'], target=source_labelS.squeeze(1))
            loss_GTA = loss_ * self.opt.S_pseudo_src
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)
        else:
            source_output = self.BaseNet_DP(source_x)
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)

            loss_GTA = cross_entropy2d(input=source_outputUp, target=source_label, size_average=True, reduction='mean')


        loss_G = loss_GTA
        loss_G.backward()
        self.BaseOpti.step()

        return loss_GTA.item()    
    

    def step(self,iteration,gth,img_id, img, saliency, label,img_full,param,target_lpsoft,device):
        
#         with autocast():
#             pred, cam = self.BaseNet_DP(img)

#             loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label) 

#             loss_sal, fg_map, bg_map, sal_pred = \
#                 get_eps_loss(cam, saliency, self.opt.num_classes, label, self.opt.tau, self.opt.lam, intermediate=True)
#         #loss 汇总
#             loss = loss_cls + loss_sal

        
        #优化器反向传播
#         self.optimizer.zero_grad()
#         self.scaler.scale(loss).backward()
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
        
        
#         return loss_cls.item(),loss_sal.item(),loss.item()
        
        if self.opt.proto_rectify:
            threshold_arg = F.interpolate(target_lpsoft, scale_factor=0.25, mode='bilinear', align_corners=True)
                        
            
        else:
            threshold_arg = F.interpolate(target_lp.unsqueeze(1).float(), scale_factor=0.25).long()

        if self.opt.ema:
            ema_input = img_full   #img   img_full
            with torch.no_grad():
                self.eval_ema()
                pred_ema, cam_ema, feat_ema = self.BaseNet_ema_DP(ema_input)

            cam_ema = F.interpolate(cam_ema, size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
            feat_ema = F.interpolate(feat_ema, size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
        
        
            
            
        pred, cam, feat = self.BaseNet_DP(target_imageS) if self.opt.S_pseudo > 0 else self.BaseNet_DP(img) #img
        # loss_cls = F.multilabel_soft_margin_loss(pred[:, :-1], label) 
        cam = F.interpolate(cam, size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
        feat = F.interpolate(feat, size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
        
        
                
        loss = torch.Tensor([0]).to(self.default_gpu)
        batch, _, w, h = threshold_arg.shape
        
        
        
        if self.opt.proto_rectify:
            # weights = self.get_prototype_weight(device,feat_ema, target_weak_params=param)
            # weights = weights.to(device)
            # b,c,h,w = weights.shape
                        
        
            # saliency = F.interpolate(saliency, size=(h, w),mode='bilinear', align_corners=True)
            # saliency_mask = torch.zeros_like(saliency).to(device)
            # saliency_mask[saliency>0.5] = 1
            
            # rectified = weights*threshold_arg[:,:-1,:,:]#*saliency_mask ### 20 21 通道，saliency前景区域为值，背景为0的cam  weights * *saliency_mask
            rectified = threshold_arg[:,:-1,:,:]  
            
            fg = rectified
            bg = torch.ones_like(fg[:,-1,:,:])*0.2
            bg = bg.unsqueeze(1).to(device)
            ### *label
            B,C,H,W = fg.shape
            label = label.reshape(B,C,1,1).expand(B,C,H,W)
            fg = fg*label
            fg_maxval,fg_maxid = fg.max(dim=1, keepdim=True)

            rectified = torch.cat((fg,bg),dim=1)
            fg_maxval,fg_maxid = rectified.max(dim=1, keepdim=True)
            rectified = rectified/fg_maxval
            
            cam = F.softmax(cam, dim=1)
            
        
        loss_L2 = MSELoss()
        loss_cam = loss_L2(cam, rectified)
        
        loss = loss + 100*loss_cam
        loss.backward()
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()

        if self.opt.moving_prototype: #update prototype
            ema_vectors, ema_ids = self.calculate_mean_vector(feat_ema.detach(), cam_ema.detach())
            # ema_vectors, ema_ids = self.calculate_mean_vector(feat_ema.detach(), cam_ema.detach(),label)
            for t in range(len(ema_ids)):
                self.update_objective_SingleVector(ema_ids[t], ema_vectors[t].detach(), start_mean=False)
        
        if self.opt.ema: #update ema model
            for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
                param_k.data = param_k.data.clone() * 0.999 + param_q.data.clone() * (1. - 0.999)
            for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.BaseNet_ema.buffers()):
                buffer_k.data = buffer_q.data.clone()
        
        return loss.item() #, loss_CTS.item(), loss_consist.item()

    
    ### 
    def full2weak(self, feat, target_weak_params):
        tmp = []
        for i in range(feat.shape[0]):
#             target_size = target_weak_params["randomresize"]
#             print("i {}".format(i))
            target_shape = target_weak_params["randomresize"]

            feat_ = F.interpolate(feat[i:i+1], size=[int(target_shape[1][i]//4), int(target_shape[0][i]//4)], mode='bilinear', align_corners=True)
#             print("feat_.shape {}".format(feat_.shape))
            b,c,h,w = feat_.shape
            randomcrop_dict = target_weak_params["randomcrop"]
#             print("randomcrop_dict {}".format(randomcrop_dict))
            cont_top = randomcrop_dict["cont_top"][i]
#             print("cont_top {}".format(cont_top))
            ch = randomcrop_dict["ch"][i]
            cont_left = randomcrop_dict["cont_left"][i]
            cw = randomcrop_dict["cw"][i]
            img_top = randomcrop_dict["img_top"][i]
            img_left = randomcrop_dict["img_left"][i]
            
            feat_temp = torch.zeros([b,c,int(self.opt.crop_size)//4,int(self.opt.crop_size)//4])

            feat_temp[:,:,cont_top//4:cont_top//4+ch//4, cont_left//4:cont_left//4+cw//4] = feat_[:,:,img_top//4:img_top//4+ch//4, img_left//4:img_left//4+cw//4]
            tmp.append(feat_temp)
        feat = torch.cat(tmp, 0)
        return feat

    def feat_prototype_distance(self, feat):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, self.class_numbers, H, W)).to(feat.device)
#         for i in range(self.class_numbers):
#             feat_proto_distance[:, i, :, :] = torch.norm(self.objective_vectors[i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)

        feat_proto_distance = F.relu(torch.cosine_similarity(feat.unsqueeze(1), self.objective_vectors.view(21, C, 1, 1).unsqueeze(0), dim=2))
        return feat_proto_distance

    def get_prototype_weight(self, device,feat, label=None, target_weak_params=None):
        feat = self.full2weak(feat, target_weak_params)
        feat = feat.to(device)
        feat_proto_distance = self.feat_prototype_distance(feat)
        feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)

        feat_proto_distance = feat_proto_distance# - feat_nearest_proto_distance

        return feat_proto_distance[:,:-1,:,:]

    
    def process_label(self, label):
        
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers+1, w, h).to(self.default_gpu)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(self.default_gpu))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()
    
    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        
        if whether_DP:
            net = nn.DataParallel(net)
        net.to(device)

        return net
    
    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
        else:
            net.eval()
        return
    def eval_ema(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        if net == None:
            for net in self.nets_ema:
                net.eval()
            for net in self.nets_DP_ema:
                net.eval()
        else:
            net.eval()
        return
    

    def train(self, net=None, logger=None):
        if net==None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
        else:
            net.train()
        self.scaler = GradScaler()
        return

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':

            self.objective_vectors[id] = self.objective_vectors[id] * (1 - self.opt.proto_momentum) + self.opt.proto_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

