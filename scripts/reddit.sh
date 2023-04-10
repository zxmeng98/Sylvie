python main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.01 \
  --weight_decay 0 \
  --n-partitions 4 \
  --n-epochs 800 \
  --model jknet \
  --n-layers 8 \
  --n-hidden 128 \
  --log-every 10 \
  --inductive \
  --datatype adap \
  --use-pp \
  --fix-seed \
  --skip-partition \
  --n-class 41 \
  --n-feat 602 \
  --n-train 153431 \
  --no-eval
  # --save_csv \
  # --save_testacc
 


# for FIX in 10 20 100
# do
# python main.py \
#   --dataset reddit \
#   --dropout 0.5 \
#   --lr 0.01 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --sampling-rate 1 \
#   --n-layers 4 \
#   --n-hidden 256 \
#   --log-every 10 \
#   --inductive \
#   --datatype int1 \
#   --use-pp \
#   --enable_pipeline \
#   --fixed-synchro ${FIX} \
#   --save_testacc
# done

# for FIX in 5 10 20 100
# do
# python main.py \
#   --dataset yelp \
#   --dropout 0.1 \
#   --lr 0.01 \
#   --n-partitions 4 \
#   --n-epochs 2000 \
#   --model graphsage \
#   --sampling-rate 1 \
#   --n-layers 4 \
#   --n-linear 2 \
#   --n-hidden 512 \
#   --log-every 10 \
#   --inductive \
#   --datatype int1 \
#   --use-pp \
#   --enable_pipeline \
#   --fixed-synchro ${FIX} \
#   --save_csv \
#   --save_testacc
# done