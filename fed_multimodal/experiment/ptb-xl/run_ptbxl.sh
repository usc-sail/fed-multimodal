# 0. download data
cd data
bash download_ptbxl.sh
cd ..

# 1. data partition, we use site as partition
python3 features/data_partitioning/ptb-xl/data_partition.py
python3 features/data_partitioning/ptb-xl/data_partition.py
# [client: [key, filepath, label]]
# /data/tiantiaf/fed-mm/output/partition/ucihar/{client_id}.json

# 2. feature extraction
python3 features/feature_processing/ptb-xl/extract_feature.py
python3 features/feature_processing/ptb-xl/extract_feature.py

# client: [[key, filepath, label, data]]
# output/feature/acc/ucihar/{client_id}.pkl
# output/feature/gyro/ucihar/{client_id}.pkl

# 3. simulate missing modality conditions
cd features/simulation_features/ptb-xl
# output/mm/ucihar/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../

# label noise
# bash run_ln.sh

# missing labels
cd experiment/ptb-xl
bash run_base.sh

