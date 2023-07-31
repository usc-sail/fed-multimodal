for alpha in 0.1 5.0; do
    for fed_alg in fed_avg fed_prox fed_opt; do
        taskset 100 python3 train.py --alpha $alpha --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.01 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --hid_size 128 --mu 0.01
        # taskset 100 python3 train.py --alpha $alpha --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 200 --fed_alg $fed_alg --hid_size 128 --mu 0.1
    done
done