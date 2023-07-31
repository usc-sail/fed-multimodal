for alpha in 0.1 5.0; do 
# for alpha in 0.1; do
    # for fed_alg in fed_avg scaffold; do
    # for fed_alg in scaffold; do
    # for fed_alg in fed_prox; do
    # for fed_alg in fed_avg; do
    # for fed_alg in fed_opt fed_avg; do
    # for fed_alg in fed_avg; do
    # for fed_alg in fed_avg fed_prox fed_rs fed_opt; do
    # for fed_alg in fed_prox fed_opt; do
    for fed_alg in fed_opt fed_prox fed_rs; do
        taskset 100 python3 train.py --alpha $alpha --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --global_learning_rate 0.005 --mu 0.1
        # taskset 100 python3 train.py --alpha $alpha --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --num_epochs 200 --fed_alg $fed_alg --global_learning_rate 0.001 --mu 1.0
    done
done
