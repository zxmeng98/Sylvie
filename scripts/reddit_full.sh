mkdir results
for N_PARTITIONS in 4 8
do 
  for MODEL in graphsage gcn gat
  do
    for DATATYPE in fp32 int1
    do
      if [ ${MODEL} == gat ]
      then
        EPOCH=100
        LAYER=2
      else
        EPOCH=100
        LAYER=4
      fi
      echo -e "\033[1mclean python processes\033[0m"
      sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
      python main.py \
        --dataset reddit \
        --dropout 0.5 \
        --lr 0.01 \
        --n-partitions ${N_PARTITIONS} \
        --n-epochs ${EPOCH} \
        --model ${MODEL} \
        --sampling-rate 1 \
        --n-layers ${LAYER} \
        --n-hidden 256 \
        --log-every 10 \
        --inductive \
        --use-pp \
        --datatype ${DATATYPE} \
        --fix-seed \
        --save_csv \
        --no-eval
      
      echo -e "\033[1mclean python processes\033[0m"
      sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, pipeline\033[0m"
      python main.py \
        --dataset reddit \
        --dropout 0.5 \
        --lr 0.01 \
        --n-partitions ${N_PARTITIONS} \
        --n-epochs ${EPOCH} \
        --model ${MODEL} \
        --sampling-rate 1 \
        --n-layers ${LAYER} \
        --n-hidden 256 \
        --log-every 10 \
        --inductive \
        --use-pp \
        --enable_pipeline \
        --datatype ${DATATYPE} \
        --fix-seed \
        --save_csv \
        --no-eval

      # echo -e "\033[1mclean python processes\033[0m"
      # sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      # echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}, fix synchro\033[0m"
      # for FIX in 2 5 10 20 100
      # do 
      #   python main.py \
      #     --dataset reddit \
      #     --dropout 0.5 \
      #     --lr 0.01 \
      #     --n-partitions ${N_PARTITIONS} \
      #     --n-epochs ${EPOCH} \
      #     --model ${MODEL} \
      #     --sampling-rate 1 \
      #     --n-layers ${LAYER} \
      #     --n-hidden 256 \
      #     --log-every 10 \
      #     --inductive \
      #     --use-pp \
      #     --enable_pipeline \
      #     --datatype ${DATATYPE} \
      #     --fix-seed \
      #     --fixed-synchro ${FIX} \
      #     --save_testacc \
      #     --save_csv
      # done
    done
  done
done
