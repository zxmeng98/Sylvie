python main.py \
  --dataset ogbn-products \
  --dropout 0.5 \
  --lr 0.01 \
  --weight_decay 0.05 \
  --n-partitions 4 \
  --n-epochs 500 \
  --model gat \
  --sampling-rate 1 \
  --n-layers 3 \
  --n-hidden 256 \
  --log-every 10 \
  --datatype fp32 \
  --use-pp \
  --fix-seed \
  --n-class 47 \
  --n-feat 100 \
  --n-train 196615 

