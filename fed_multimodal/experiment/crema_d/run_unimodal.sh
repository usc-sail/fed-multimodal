for fed_alg in fed_opt; do
    for modality in audio video; do
        taskset 50 python3 train_unimodal.py --hid_size 128 --sample_rate 0.1 --modality $modality --learning_rate 0.05 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --global_learning_rate 0.025
    done
done
