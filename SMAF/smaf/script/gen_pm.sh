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
