mkdir results
for N_PARTITIONS in 4 7
do 
  for MODEL in graphsage gcn gat
  do
    for DATATYPE in fp32 int4 int1
    do
      if [ ${MODEL} == gat ]
      then
        EPOCH=500
      else
        EPOCH=2000
      fi
      echo -e "\033[1mclean python processes\033[0m"
      sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
      echo -e "\033[1mmodel ${MODEL}, datatype ${DATATYPE}, epoch ${EPOCH}\033[0m"
      python main.py \
        --dataset ogbn-arxiv \
        --dropout 0.3 \
        --lr 0.01 \
        --n-partitions ${N_PARTITIONS} \
        --n-epochs ${EPOCH} \
        --model ${MODEL} \
        --sampling-rate 1 \
        --n-layers 4 \
        --n-hidden 256 \
        --log-every 10 \
        --inductive \
        --use-pp \
        --datatype ${DATATYPE} \
        --save_csv

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
        --n-layers 4 \
        --n-hidden 256 \
        --log-every 10 \
        --inductive \
        --use-pp \
        --enable_pipeline \
        --datatype ${DATATYPE} \
        --save_csv
    done
  done
done
