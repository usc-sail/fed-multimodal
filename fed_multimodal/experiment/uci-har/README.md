### Quick Start -- UCI-HAR (Acc. and Gyro)
Here we provide an example to quickly start with the experiments, and reproduce the UCI-HAR results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.


#### 0. Download data: The data will be under data/uci-har by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_uci_har.sh
cd ..
```

#### 1. Partition the data

alpha specifies the non-iidness of the partition, the lower, the higher data heterogeneity. As each subject performs the same amount activities, we partition each subject data into 5 sub-clients.

```
python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 5
python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0 --num_clients 5
```

The return data is a list, each item containing [key, file_name, label]

#### 2. Feature extraction

For UCI-HAR dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/uci-har
# output/mm/ucihar/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/uci-har
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality
```
cd experiment/uci-har
bash run_mm.sh
```

#### Baseline results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
UCI-HAR | Acc+Gyro | [UCI-Data](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 | 105 | Natural+Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 77.74% <br> 85.17% <br> 76.66% <br> 79.80% | 0.05 | 200 |

