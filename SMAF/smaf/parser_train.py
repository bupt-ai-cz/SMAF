# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json

def parser_(parser):
    parser.add_argument('--root', type=str, default='../dataset/VOC2012', help='root path')
    parser.add_argument('--model_name', type=str, default='deeplabv2', help='deeplabv2')
    parser.add_argument('--name', type=str, default='voc2012', help='pretrain source model')
    parser.add_argument('--lr', type=float, default=0.01)  #0.0001     #  eps:0.01
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--epochs', type=int, default=84)
    parser.add_argument('--train_iters', type=int, default=90000)
    parser.add_argument('--moving_prototype', action='store_true')
    parser.add_argument('--bn', type=str, default='bn', help='sync_bn|bn|gn|adabn')
    ##################################################################################
    parser.add_argument('--gt_path', type=str, default='SegmentationClassAug', help='root path')
    parser.add_argument('--debug_point', type=str, default='true', help='root path')
    parser.add_argument('--pseudo_mask_warmup_path', type=str, default='EPS_pseudo_mask', help='root path')
    ####################################################
    parser.add_argument('--max_iters', type=int, default=20000)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--weights', type=str, default='weight/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', help='root path')
    parser.add_argument('--log_folder', type=str, default='logdir/eps_cls_mode', help='root path')
    
#     parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wt_dec', type=float, default=5e-4)
    parser.add_argument('--network_type', type=str, default='eps', help='root path')
    parser.add_argument('--dataset', type=str, default='voc12', help='root path')
    parser.add_argument("--train_list", default="metadata/voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="metadata/voc12/train.txt", type=str)
    parser.add_argument("--data_root", default="../dataset/VOC2012/JPEGImages", type=str)
    parser.add_argument("--saliency_root", default="../dataset/VOC2012/SALImages", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--resize_size", default=(256, 512), type=int, nargs='*')
    parser.add_argument("--batch_size", default=8, type=int)
    
    
    ### gen_cam or
#     parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--n_processes_per_gpu", nargs='*', type=int)
    parser.add_argument("--n_total_processes", default=1, type=int)
    parser.add_argument("--img_root", default='VOC2012', type=str)
    parser.add_argument("--crf", default=None, type=str)
    # parser.add_argument("--crf_alpha", nargs='*', type=int)
    parser.add_argument("--crf_alpha", nargs='*',default=[4], type=int)
    # parser.add_argument("--crf_t", nargs='*', type=int)
    parser.add_argument("--crf_t", nargs='*',default=[10], type=int)
    parser.add_argument("--cam_npy", default=None, type=str)
    parser.add_argument("--cam_png", default=None, type=str)
    parser.add_argument("--crf_png", default=None, type=str)
    parser.add_argument("--thr", default=0.20, type=float)
    ### gen cam or end
    parser.add_argument("--split", default="train", type=str)
    #######################################################################################
    #training
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--stage', type=str, default='stage1', help='stage1|stage1|stage2|stage3')
    parser.add_argument('--finetune', action='store_true')
    #model
    parser.add_argument('--resume_path', type=str, default='pretrained/EPS_warmup_proda_deeplab/from_EPS_to_cityscapes_on_deeplabv2_best_model.pkl', help='resume model path')
    parser.add_argument('--ema', action='store_true', help='use ema model')
    parser.add_argument('--ema_bn', action='store_true', help='add extra bn for ema model')
    parser.add_argument("--student_init", default='stage1', type=str, help="stage1|imagenet|simclr")
    parser.add_argument("--proto_momentum", default=0.0001, type=float)
    parser.add_argument("--bn_clr", action='store_true', help="if true, add a bn layer for the output of simclr model")
    #data
    parser.add_argument('--src_dataset', type=str, default='EPS', help='gta5|synthia')
    parser.add_argument('--tgt_dataset', type=str, default='cityscapes', help='cityscapes')
    parser.add_argument('--src_rootpath', type=str, default='Dataset/GTA5')
    parser.add_argument('--tgt_rootpath', type=str, default='Dataset/cityscapes')
    parser.add_argument('--path_LP', type=str, default='Pseudo/pretraisn_warmup/LP0.95', help='path of probability-based PLA')
    parser.add_argument('--path_soft', type=str, default='Pseudo/pretrain_warmup_soft/LP0.0', help='soft pseudo label for rectification')
    parser.add_argument("--train_thred", default=0, type=float)
    parser.add_argument('--used_save_pseudo', action='store_true', help='if True used saved pseudo label')
    parser.add_argument('--no_droplast', action='store_true')

#     parser.add_argument('--resize', type=int, default=2200, help='resize long size')
    parser.add_argument('--resize', type=int, default=512, help='resize long size')
    #parser.add_argument('--rcrop', type=str, default='896,512', help='rondom crop size')
    parser.add_argument('--rcrop', type=str, default='224,224', help='rondom crop size')
    parser.add_argument('--rcrop_full_img', type=int, default=224, help='rondom crop size full -img')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')

    #parser.add_argument('--n_class', type=int, default=19, help='19|16|13')
    parser.add_argument('--n_class', type=int, default=21, help='19|16|13')
    parser.add_argument('--num_workers', type=int, default=6)
    #loss
    parser.add_argument('--gan', type=str, default='LS', help='Vanilla|LS')
    parser.add_argument('--adv', type=float, default=0.01, help='loss weight of adv loss, only use when stage=warm_up')
    parser.add_argument('--S_pseudo_src', type=float, default=0.0, help='loss weight of pseudo label for strong augmentation of source')
    parser.add_argument("--rce", action='store_true', help="if true, use symmetry cross entropy loss")
    parser.add_argument("--rce_alpha", default=0.1, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--rce_beta", default=1.0, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--regular_w", default=0, type=float, help='loss weight for regular term')
    parser.add_argument("--regular_type", default='MRKLD', type=str, help='MRENT|MRKLD')
#     parser.add_argument('--proto_consistW', type=float, default=1.0, help='loss weight for proto_consist')
    parser.add_argument('--proto_consistW', type=float, default=0, help='loss weight for proto_consist')
    parser.add_argument("--distillation", default=0, type=float, help="kl loss weight")

    parser.add_argument('--S_pseudo', type=float, default=0.0, help='loss weight of pseudo label for strong augmentation')

    #print
    parser.add_argument('--print_interval', type=int, default=50, help='print loss')
#     parser.add_argument('--val_interval', type=int, default=1000, help='validate model iter')
    parser.add_argument('--val_interval', type=int, default=500, help='validate model iter')

    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--noaug', action='store_true', help='do not use data augmentation')

    parser.add_argument('--proto_rectify', action='store_true')
    parser.add_argument('--proto_temperature', type=float, default=1.0)
    #stage2
    parser.add_argument("--threshold", default=-1, type=float)
    return parser

def relative_path_to_absolute_path(opt):
    opt.rcrop = [int(opt.rcrop.split(',')[0]), int(opt.rcrop.split(',')[1])]
    opt.src_rootpath = os.path.join(opt.root, opt.src_rootpath)
    opt.tgt_rootpath = os.path.join(opt.root, opt.tgt_rootpath)

    opt.logdir = os.path.join(opt.log_folder,'logs')
    return opt