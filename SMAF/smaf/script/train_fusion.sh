python train.py \
    --name eps_proda_stage1Denoise \
    --proto_rectify \
    --used_save_pseudo \
    --ema \
    --path_soft ./logdir/cam_o_vec/cam_npy \
    --weights logdir/eps_cls_mode/checkpoint_cls_finally.pth \
    --log_folder logdir/proda_model \
    --moving_prototype 