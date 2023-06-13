python3 gen_fused_cam.py \
    --dataset voc12 \
    --infer_list metadata/voc12/train_aug.txt \
    --img_root ../dataset/VOC2012/JPEGImages \
    --network network.resnet38_eps \
    --weights logdir/eps_cls_mode/checkpoint_cls_finally.pth \
    --thr 0.20 \
    --n_gpus 1 \
    --n_processes_per_gpu 1 1 \
    --cam_png logdir/cam_o_vec/cam_png \
    --cam_npy logdir/cam_o_vec/cam_npy \
python3 evaluate.py \
    --dataset voc12 \
    --datalist metadata/voc12/train_aug.txt \
    --gt_dir ../dataset/VOC2012/SegmentationClassAug/ \
    --save_path logdir/cam_o_vec/train_aug.txt \
    --pred_dir logdir/cam_o_vec/cam_png

python train.py \
    --name eps_proda_stage1Denoise \
    --proto_rectify \
    --used_save_pseudo \
    --ema \
    --path_soft ./logdir/cam_o_vec/cam_npy \
    --weights logdir/eps_cls_mode/checkpoint_cls_finally.pth \
    --log_folder logdir/proda_model \
    --moving_prototype 




#crf infer 验证优化后的模型infer 验证优化后的模型
python3 gen_pseudo_mask.py \
    --dataset voc12 \
    --infer_list metadata/voc12/train.txt \
    --img_root ../dataset/VOC2012/JPEGImages \
    --network network.resnet38_eps \
    --weights logdir/proda_model/checkpoint_cls_finally.pth \
    --thr 0.20 \
    --n_gpus 1 \
    --n_processes_per_gpu 1 1 \
    --cam_png logdir/proda_model/result_crf/cam_png \
    --crf crf \
    --crf_png logdir/proda_model/result_crf/crf_png
    --cam_npy logdir/proda_model/result/cam_npy \
    
python3 evalue.py \
    --dataset voc12 \
    --datalist metadata/voc12/train.txt \
    --gt_dir ../dataset/VOC2012/SegmentationClassAug/ \
    --save_path logdir/proda_model/result_crf/train.txt \
    --pred_dir logdir/proda_model/result_crf/crf_png


