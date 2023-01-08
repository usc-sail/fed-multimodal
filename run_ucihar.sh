# 1. data partition
python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1
python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0
# [client: [key, filepath, label]]
# /data/tiantiaf/fed-mm/output/partition/ucihar/{client_id}.json

# 2. feature extraction
python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0

# client: [[key, filepath, label, data]]
# /data/tiantiaf/fed-mm/output/feature/acc/ucihar/{client_id}.pkl
# /data/tiantiaf/fed-mm/output/feature/gyro/ucihar/{client_id}.pkl

# 3. simulate conditions
# cd simulation_features/uci-har
# /data/tiantiaf/fed-mm/output/mm/ucihar/{client_id}_{mm_rate}.json

# missing modalities
# bash run_mm.sh

# label noise
# bash run_ln.sh

# missing labels
cd experiment/uci-har
bash run_base.sh

