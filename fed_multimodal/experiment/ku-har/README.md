
### How to run KU-HAR Dataset in FedMultimodal (Acc. and Gyro)
Here we provide an example to quickly start with the experiments, and reproduce the KU-HAR results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.


#### 0. Download data: The data will be under data/ku-har by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_ku_har.sh
cd ..
```

Data will be under data/ku-har

#### 1. Partition the data

This data has the natural partition, so we do not use any simulating. We partition the data in 5-fold, so we can get averaged performance.

```
python3 features/data_partitioning/ku-har/data_partition.py
```

The return data is a list, each item containing [key, file_name, label]

#### 2. Feature extraction

For KU-HAR dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/ku-har/extract_feature.py
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/ku-har
# output/mm/kuhar/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/ku-har
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality (the same for missing labels and label noises)
```
cd experiment/ku-har
bash run_mm.sh
```

#### Baseline results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
KU-HAR | Acc+Gyro | [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5) | 18 | ~65 | Manual | - |  FedAvg <br> FedOpt | 61.78% <br> 71.41% | 0.05 | 200 |

