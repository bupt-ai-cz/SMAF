import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path
from tqdm import tqdm

from data import create_dataset
from data.VOC_dataset import VOCAug_stage1
from models import adaptation_modelv2
from metrics import runningScore

from crf_lib.utils import DenseCRF, PolynomialLR, scores
from crf_lib.utils.decode import decode_segmap
import multiprocessing
import joblib
import cv2
import json


# @main.command()

# @click.option(
#     "-j",
#     "--n-jobs",
#     type=int,
#     default=multiprocessing.cpu_count(),
#     show_default=True,
#     help="Number of parallel jobs",
# )
# @click.option(
#     "--log_dir", required=True, help="tensorboard log dir"
# )
def crf(n_jobs, opt):
    
    """
    CRF post-processing on pre-computed logits
    """
    
    # Configuration
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Dataset
#     datasets = create_dataset(opt, logger) 
    datasets = VOCAug_stage1(opt, logger, augmentations=None, split='val')
    
#     data_i = datasets.target_valid_loader.__getitem__(i)
#     print("********************")
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )
#     postprocessor = DenseCRF(
#         iter_max=CONFIG.CRF.ITER_MAX,
#         pos_xy_std=CONFIG.CRF.POS_XY_STD,
#         pos_w=CONFIG.CRF.POS_W,
#         bi_xy_std=CONFIG.CRF.BI_XY_STD,
#         bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
#         bi_w=CONFIG.CRF.BI_W,
#     )

    # Path to logit files
    root_crf_dir = os.path.join(opt.logdir,"proda_temp")
    logit_dir = os.path.join(root_crf_dir,"np_for_crf")
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(root_crf_dir,"result_crf")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir_img = os.path.join(save_dir,"result")
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
        
#     makedirs(save_dir)
#     makedirs("data/result")
    #save_path = os.path.join(save_dir, "scores_crf.json")
    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)

    pixel_mean = np.array((104.008, 116.669, 122.675))
    
    ############################################################################3
#     for data_i in tqdm(valid_loader):
#         image_id, image, gt_label, cls_labels = datasets.target_valid_loader.__getitem__(i)
#         data_i = datasets.target_valid_loader.__getitem__(i)
#         images_val = data_i['img']
#         gt_label = data_i['label']
#         image_id = data_i['img_path']
    
#         filename = os.path.join(logit_dir, image_id + ".npy")
#         logit = np.load(filename)

#         _, H, W = image.shape
#         logit = torch.FloatTensor(logit)[None, ...]
#         logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
#         prob = F.softmax(logit, dim=1)[0].numpy()

#         image += pixel_mean[:, None, None]
#         image = image.astype(np.uint8).transpose(1, 2, 0)
#         prob = postprocessor(image, prob)
#         label = np.argmax(prob, axis=0)
        
#         gt_map = decode_segmap(gt_map) * 255
#         pred_map = decode_segmap(pred_map) * 255
            
#         gt_map = gt_map[:, :, ::-1].astype(np.uint8)
#         pred_map = pred_map[:, :, ::-1].astype(np.uint8)

#         result_img = cv2.hconcat([image, gt_map, pred_map])
#         cv2.imwrite(save_dir_img + "/{}.png".format(image_id), result_img)
# #             cv2.imwrite("data/result/%s.png" % (image_id), result_img)
    
    
#     # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
#     score = scores(gts, preds, n_class=21)
#     print(score)
    
#     with open(save_path, "w") as f:
#         json.dump(score, f, indent=4, sort_keys=True)
    ############################################################################
    
    # Process per sample
    def process(i):
#         image_id, image, gt_label, cls_labels = datasets.target_valid_loader.__getitem__(i)
        data_i = datasets.__getitem__(i)
        images_val = data_i['img']
        gt_label = data_i['label']
        image_id = data_i['img_path']
    
        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)
        logit = np.squeeze(logit)
        _, H, W = images_val.shape
        logit = torch.FloatTensor(logit)[None, ...]
        print("logit shape {}".format(logit.shape))
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        images_val += pixel_mean[:, None, None]
        images_val = images_val.numpy().astype(np.uint8).transpose(1, 2, 0)
#         images_val = images_val.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(images_val, prob)
        label = np.argmax(prob, axis=0)
        
        return image_id, images_val, label, gt_label.numpy()

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(datasets))]
    )

    image_ids, images, preds, gts = zip(*results)

    if True: # save img
        for image_id, image, pred_map, gt_map in zip(image_ids, images, preds, gts):
            gt_map = decode_segmap(gt_map) * 255
            pred_map = decode_segmap(pred_map) * 255
            
            gt_map = gt_map[:, :, ::-1].astype(np.uint8)
            pred_map = pred_map[:, :, ::-1].astype(np.uint8)

            result_img = cv2.hconcat([image, gt_map, pred_map])
            cv2.imwrite(save_dir_img + "/{}.png".format(image_id), result_img)
#             cv2.imwrite("data/result/%s.png" % (image_id), result_img)
    
    
    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=21)
    print(score)
    
    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)
        
def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)
    
    n_jobs = multiprocessing.cpu_count()
    crf(n_jobs, opt)
#     test(opt, logger)