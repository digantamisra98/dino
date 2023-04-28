torchrun --nproc_per_node=8 main_dino.py \
    --arch vit_base --epochs 30 --batch_size_per_gpu 64 \
    --accum_iter 2 --lr 5e-4 --warmup_epochs 3 \
    --data_path /home/storage/diganta/imagenet/train \
    --output_dir /home/storage/diganta/resoultion_exps/dino_base_384_96_lr5e-4 \
    --pretrained_weights /home/storage/diganta/pretrain/dino_vitbase16_pretrain.pth
