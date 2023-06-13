# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path


from utils import get_logger
from models import adaptation_modelv2
from metrics import runningScore, averageMeter
from tensorboardX import SummaryWriter

from PIL import Image
from data.eps_dataloader import get_dataloader

from util import pyutils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



def train(opt, logger):
    
    avg_meter = pyutils.AverageMeter('loss_cam')
    timer = pyutils.Timer("Session started: ")
    max_step = opt.max_iters
    ######
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    #########
    train_loader,val_loader = get_dataloader(opt)
    
    model = adaptation_modelv2.CustomModel(opt, logger)
    
    if opt.stage == 'stage1':
        objective_vectors = torch.load("logdir/cam_o_vec/prototypes_eps")
        model.objective_vectors = torch.Tensor(objective_vectors).to(device)
    
    
    # Setup Metrics
    running_metrics_val = runningScore(opt.n_class)

    loader_iter = iter(train_loader)
    model.iter = 0
    start_epoch = 0
    # miou = validate(model, val_loader,0,device)
    for iteration in range(opt.max_iters):   #此参数缺
            try:
                img_id, img, saliency, label,img_full,param,lpsoft,gth = next(loader_iter)
            except:
                loader_iter = iter(train_loader)
                img_id, img, saliency, label,img_full,param,lpsoft,gth = next(loader_iter)
            
            

            img = img.to(device)
            saliency = saliency.to(device)
            label = label.to(device)
            img_full = img_full.to(device)
            gth = gth.to(device)
            lpsoft = lpsoft.to(device)


            
            model.iter += 1
            i = model.iter
            
            start_ts = time.time()

            model.train(logger=logger)
            if opt.freeze_bn:
                model.freeze_bn_apply()
            model.optimizer_zerograd()
            
            if opt.stage == 'warm_up':
                loss_GTA = model.step_adv(target_image, pseudo_mask_warmup)
    
            elif opt.stage == 'stage1':

                loss_cam = model.step(i,gth,img_id, img, saliency, label,img_full,param,lpsoft,device)

                avg_meter.add({'loss_cam': loss_cam})
            else:
                loss_GTA, loss = model.step_distillation(images, labels, target_image, target_imageS, target_params, target_lp)


            if (i + 1) % 50 == 0:
                timer.update_progress(i / max_step)

                loss_cam_average = avg_meter.pop('loss_cam')


                print('Iter:%5d/%5d' % (iteration, opt.max_iters),
                      'loss_cam:%.4f' % (loss_cam_average),
                      'Rem:%s' % (timer.get_est_remain()),flush=True)

                writer.add_scalar('loss_cam_average', loss_cam_average, iteration)
            timer.reset_stage()  #model.module.state_dict()
            if (i + 1) % 1000 == 0:
                # state = {}
                # for net in model.nets:
                #     new_state = {
                #         "model_state": net.state_dict(),
                #         "objective_vectors": model.objective_vectors,                
                #         }
                # state[net.__class__.__name__] = new_state
                # state['iter'] = iteration

                # torch.save(state, os.path.join(opt.log_folder,str(i-1)+ 'checkpoint_cls.pth'))
                
                miou = validate(model, val_loader,iteration,device)
    state = {}
    for net in model.nets:
        new_state = {
            "model_state": net.state_dict(),
            "objective_vectors": model.objective_vectors,
        }
    state[net.__class__.__name__] = new_state
    state['iter'] = iteration
    torch.save(state, os.path.join(opt.log_folder,'checkpoint_cls_finally.pth'))
    miou = validate(model, val_loader,iteration,device)


                    
def validate(model, data_loader,iteration,device):

    print('validating ... ')

    model.eval()
    running_metrics_val = runningScore(opt.n_class)
    with torch.no_grad():
        preds = []
        for iter, data in enumerate(data_loader):     
            img_id, img,label,gth = data
            original_image_size = np.asarray(img).shape[2:]
            img = img.to(device)
            label = label.to(device)
            gth = gth.numpy()#.to(device)
            cam = model.BaseNet_DP.forward_cam(img)
            cam = F.softmax(cam, dim=1)
            cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)#[0]

            cam_fg = cam[:,:-1,:,:]
            cam_bg = cam[:,-1,:,:]
            cam_bg = torch.ones_like(cam_fg[:,-1,:,:])*0.2
            cam_bg = cam_bg.unsqueeze(1).to(device)

            cam_fg = cam_fg * label.reshape(opt.num_classes, 1, 1)

            cam = torch.cat((cam_bg,cam_fg),dim=1).cpu().numpy()
            pred = np.argmax(cam, axis=1).astype(np.uint8)
            running_metrics_val.update(gth, pred)
            
        score, class_iou = running_metrics_val.get_scores()
        running_metrics_val.reset()
        writer.add_scalar('miou', score["Mean IoU : \t"], iteration)
        print("score: {}".format(score))
        print("Mean IoU: {}".format(score["Mean IoU : \t"]))
        

    model.train()

    return score["Mean IoU : \t"]
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    print("opt : {}".format(opt))

    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)
    if not os.path.exists(opt.log_folder):
        os.makedirs(opt.log_folder)

    logger = get_logger(opt.logdir)
    writer = SummaryWriter(log_dir='tensorboard_log')
    
    
    train(opt, logger)
