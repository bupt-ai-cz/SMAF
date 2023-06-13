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