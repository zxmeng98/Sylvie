mkdir results
for N_PARTITIONS in 4 
do
  for MODEL in gcn
  do
    for DATATYPE in int1
    do
      if [ ${MODEL} == gat ]
      then
        EPOCH=1000
      else
        EPOCH=1000
      fi
      # echo -e "\033[1mclean python processes\033[0m"
      # sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      # echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
      # python main.py \
      #   --dataset ogbn-products \
      #   --dropout 0.3 \
      #   --lr 0.01 \
      #   --n-partitions ${N_PARTITIONS} \
      #   --n-epochs ${EPOCH} \
      #   --model ${MODEL} \
      #   --n-layers 3 \
      #   --n-hidden 128 \
      #   --log-every 10 \
      #   --use-pp \
      #   --datatype ${DATATYPE} \
      #   --fix-seed \
      #   --save_testacc

      # echo -e "\033[1mclean python processes\033[0m"
      # sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      # echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, pipeline\033[0m"
      # python main.py \
      #   --dataset ogbn-products \
      #   --dropout 0.3 \
      #   --lr 0.01 \
      #   --n-partitions ${N_PARTITIONS} \
      #   --n-epochs ${EPOCH} \
      #   --model ${MODEL} \
      #   --n-layers 3 \
      #   --n-hidden 128 \
      #   --log-every 10 \
      #   --use-pp \
      #   --enable_pipeline \
      #   --datatype ${DATATYPE} \
      #   --save_testacc 

      echo -e "\033[1mclean python processes\033[0m"
      sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, fix synchro\033[0m"
      for FIX in 2 5 
      do
      python main.py \
        --dataset ogbn-products \
        --dropout 0.3 \
        --lr 0.01 \
        --n-partitions ${N_PARTITIONS} \
        --n-epochs ${EPOCH} \
        --model ${MODEL} \
        --n-layers 3 \
        --n-hidden 128 \
        --log-every 10 \
        --use-pp \
        --enable_pipeline \
        --datatype ${DATATYPE} \
        --fixed-synchro ${FIX} \
        --save_testacc
      done
    done
  done
done