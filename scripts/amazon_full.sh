mkdir results
for N_PARTITIONS in 4 8
do 
  for MODEL in graphsage gcn gat
  do
    for DATATYPE in fp32 int1
    do
      if [ ${MODEL} == gat ]
      then
        EPOCH=150
        LINEAR=0
        LAYER=3
        HIDDEN=128
      else
        EPOCH=150
        LINEAR=2
        LAYER=4
        HIDDEN=512
      fi
      echo -e "\033[1mclean python processes\033[0m"
      sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
      python main.py \
        --dataset amazon \
        --dropout 0.1 \
        --lr 0.01 \
        --n-partitions ${N_PARTITIONS} \
        --n-epochs ${EPOCH} \
        --model ${MODEL} \
        --n-layers ${LAYER} \
        --n-linear ${LINEAR} \
        --n-hidden ${HIDDEN} \
        --log-every 10 \
        --datatype ${DATATYPE} \
        --inductive \
        --use-pp \
        --save_csv \
        --no-eval 

      echo -e "\033[1mclean python processes\033[0m"
      sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, pipeline\033[0m"
      python main.py \
        --dataset amazon \
        --dropout 0.1 \
        --lr 0.01 \
        --n-partitions ${N_PARTITIONS} \
        --n-epochs ${EPOCH} \
        --model ${MODEL} \
        --n-layers ${LAYER} \
        --n-linear ${LINEAR} \
        --n-hidden ${HIDDEN} \
        --log-every 10 \
        --enable_pipeline \
        --datatype ${DATATYPE} \
        --inductive \
        --use-pp \
        --save_csv \
        --no-eval
      
      # echo -e "\033[1mclean python processes\033[0m"
      # sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      # echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, fix synchro\033[0m"
      # for FIX in 2 5 10 20 100
      # do
      # python main.py \
      #   --dataset amazon \
      #   --dropout 0.1 \
      #   --lr 0.01 \
      #   --n-partitions ${N_PARTITIONS} \
      #   --n-epochs ${EPOCH} \
      #   --model ${MODEL} \
      #   --n-layers ${LAYER} \
      #   --n-linear ${LINEAR} \
      #   --n-hidden ${HIDDEN} \
      #   --log-every 10 \
      #   --datatype ${DATATYPE} \
      #   --inductive \
      #   --use-pp \
      #   --fixed-synchro ${FIX} \
      #   --save_csv \
      #   --save_testacc 
      # done
    done
  done
done