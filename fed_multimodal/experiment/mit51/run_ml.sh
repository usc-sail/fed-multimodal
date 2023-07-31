
for ml_rate in 0.1 0.2 0.3 0.4 0.5; do
   taskset 100 python3 train.py --alpha 0.1 --en_missing_label --missing_label_rate $ml_rate --sample_rate 0.05 --learning_rate 0.05 --global_learning_rate 0.05 --num_epochs 300 --fed_alg fed_opt --mu 0.01 --en_att --att_name fuse_base --hid_size 128
done
