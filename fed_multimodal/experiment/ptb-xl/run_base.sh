for fed_alg in fed_avg fed_prox fed_opt; do
    taskset -c 1-30 python3 train.py --hid_size 128 --sample_rate 0.5 --learning_rate 0.05 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg  --global_learning_rate 0.01 --mu 0.01
done