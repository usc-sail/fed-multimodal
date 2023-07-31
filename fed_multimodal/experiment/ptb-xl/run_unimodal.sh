for fed_alg in fed_opt; do
    for modality in i_to_avf v1_to_v6; do
        taskset -c 1-30 python3 train_unimodal.py --modality $modality --hid_size 128 --sample_rate 0.25 --learning_rate 0.05 --num_epochs 200 --en_att --att_name base --fed_alg fed_opt --global_learning_rate 0.025
    done
done
