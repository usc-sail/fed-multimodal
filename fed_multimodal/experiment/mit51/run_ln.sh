# for missing_rate in 0.9 0.7 0.5 0.3 0.1; do
# for alpha in 0.1 0.5 5.0; do
# for label_noisy in 0.25 0.5; do
for label_noisy in 0.2 0.4; do
    for fed_alg in fed_opt; do
        taskset 100 python3 train.py --alpha 0.1 --hid_size 128 --sample_rate 0.05 --learning_rate 0.05 --global_learning_rate 0.025 --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg --en_label_nosiy --label_nosiy_level $label_noisy
    done
done