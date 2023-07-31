for fed_alg in fed_opt; do
    for modality in text; do
        # taskset 50 python3 train_unimodal.py --hid_size 128 --sample_rate 0.1 --modality $modality --learning_rate 0.05 --num_epochs 200 --en_att --att_name fuse_base --fed_alg $fed_alg --global_learning_rate 0.005
        # taskset 100 python3 train_unimodal.py --modality $modality --alpha 0.1 --hid_size 128 --sample_rate 0.05 --learning_rate 0.05 --num_epochs 300 --en_att --att_name base --fed_alg fed_opt --global_learning_rate 0.0025
        taskset 100 python3 train.py --alpha $alpha --modality $modality --hid_size 128 --sample_rate 0.1 --learning_rate 0.05  --global_learning_rate 0.001 --num_epochs 200 --en_att --att_name base --fed_alg $fed_alg --mu 0.01
    done
done
