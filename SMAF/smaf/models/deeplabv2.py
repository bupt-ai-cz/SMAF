#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, 256, 3, 1, padding=rate, dilation=rate, bias=True),
            )
        self.add_module(
                "head0",
                nn.Conv2d(256, out_ch, 1, 1, padding=0, dilation=1, bias=True),
            )
        for m in self.children():
#             print("**********************")
#             print(m)
#             print("######################")
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
#         print("++++++++++++++++++++++++++++++++++++++++++++")
#         for m in self.named_modules():
#             print("**********************")
#             print(m)
#             print("######################")
#             if "aspp" in m[0]
            

    def forward(self, x):
        out_dict = {}
        x_list = []
        x_list.append(self.c0(x))
        x_list.append(self.c1(x))
        x_list.append(self.c2(x))
        x_list.append(self.c3(x))
        
        x = sum(x_list)
#         print("sun shape {}".format(x.shape))
#         return sum([stage(x) for stage in self.children()])
#         x = sum([stage(x) for stage in self.children()])
        out_dict['feat'] = x
        x = self.head0(x)
#         print("head0 shape {}".format(x.shape))
        out_dict['out'] = x
        
        return out_dict
###################################################################################

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=True):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out
#######################################################################################################


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        #self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))
        self.add_module("aspp", Classifier_Module2(ch[5], atrous_rates, atrous_rates,n_classes))

    def freeze_bn_(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

################################################################################

#warmup
# def Deeplab(BatchNorm, num_classes=21, freeze_bn=False, restore_from=None, initialization=None, bn_clr=False):
#     model = DeepLabV2(n_classes=num_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
#     if freeze_bn:
#         model.freeze_bn_()
# #         model.apply(freeze_bn)
#     if restore_from is not None: 
#         checkpoint = torch.load(restore_from)
#         print("restore_from***************** {}".format(restore_from))
# #         model.load_state_dict(checkpoint['ResNet101']["model_state"])
#         #model.load_state_dict(checkpoint['ema'])
#         model.load_state_dict(checkpoint, strict=False)
#     return model
########################################################################################
#proda
def Deeplab(BatchNorm, num_classes=21, freeze_bn=False, restore_from=None, initialization=None, bn_clr=False):
    model = DeepLabV2(n_classes=num_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
    if freeze_bn:
        model.freeze_bn_()
#         model.apply(freeze_bn)


    if restore_from is not None: 
        checkpoint = torch.load(restore_from)
        print("restore_from***************** {}".format(restore_from))
        model.load_state_dict(checkpoint['DeepLabV2']["model_state"])
        #model.load_state_dict(checkpoint['ema'])
#         model.load_state_dict(checkpoint, strict=False)
    
    return model


