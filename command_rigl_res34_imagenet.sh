python train_imagenet_rigl.py -a resnet34 \
    --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' \
    --multiprocessing-distributed 1 --world-size 1 --rank 0 \
    --data $IMAGENET_PATH --workers 36 \
    --batch-size 1024 --lr 0.4 \
    --train-step-multiplier 5.0 \
    --seed 42 --print-freq 100
