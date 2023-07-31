# python3 ucf101/train.py --alpha 1.0 --en_missing_modality --missing_label_rate 0.5


# taskset 100 python3 train.py --alpha 1.0 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300  --en_att
# taskset 100 python3 train.py --alpha 0.25 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att
# taskset 100 python3 train.py --alpha 0.1 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att

taskset 100 python3 train.py --alpha 1.0 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300  --en_att --en_missing_modality --missing_modailty_rate 0.3
taskset 100 python3 train.py --alpha 0.25 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att --en_missing_modality --missing_modailty_rate 0.3
taskset 100 python3 train.py --alpha 0.1 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att --en_missing_modality --missing_modailty_rate 0.3

taskset 100 python3 train.py --alpha 1.0 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300  --en_att --en_missing_modality --missing_modailty_rate 0.1
taskset 100 python3 train.py --alpha 0.25 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att --en_missing_modality --missing_modailty_rate 0.1
taskset 100 python3 train.py --alpha 0.1 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att --en_missing_modality --missing_modailty_rate 0.1

# taskset 100 python3 train.py --alpha 1.0 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300  --en_att --en_missing_modality --missing_modailty_rate 0.5
# taskset 100 python3 train.py --alpha 0.25 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att --en_missing_modality --missing_modailty_rate 0.5
# taskset 100 python3 train.py --alpha 0.1 --sample_rate 0.1 --learning_rate 0.1 --num_epochs 300 --en_att --en_missing_modality --missing_modailty_rate 0.5


