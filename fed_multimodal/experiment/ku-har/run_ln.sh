for label_noisy in 0.2 0.4; do
    for fed_alg in fed_opt; do
        taskset 100 python3 train.py --hid_size 128 --sample_rate 0.1 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --en_label_nosiy --label_nosiy_level $label_noisy
    done
done