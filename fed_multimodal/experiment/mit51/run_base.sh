# python3 ucf101/train.py --alpha 1.0 --en_missing_modality --missing_label_rate 0.5
for alpha in 0.1 5.0; do 
    for mu in 0.01; do
        for fed_alg in fed_avg fed_prox fed_opt; do
            taskset -c 1-30 python3 train.py --alpha $alpha --hid_size 128 --sample_rate 0.05 --learning_rate 0.05 --global_learning_rate 0.025 --mu $mu --num_epochs 300 --en_att --att_name fuse_base --fed_alg $fed_alg
            # taskset 500 python3 train.py --alpha $alpha --hid_size 128 --sample_rate 0.05 --learning_rate 0.05 --global_learning_rate 0.025 --mu $mu --num_epochs 300 --fed_alg $fed_alg
        done
    done
done