for fed_alg in fed_opt; do
    for modality in audio video; do
        taskset 100 python3 train_unimodal.py --alpha 0.1 --modality $modality --hid_size 128 --sample_rate 0.05 --learning_rate 0.05 --num_epochs 300 --en_att --att_name base --fed_alg fed_opt --global_learning_rate 0.05
    done
done
