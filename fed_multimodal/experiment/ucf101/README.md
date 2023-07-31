### Quick Start -- UCF-101 (Video and Audio)
Here we provide an example to quickly start with the experiments, and reproduce the UCF-101 results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.


#### 0. Download data

If you want start from scratch, please follow [MMAction2](https://github.com/open-mmlab/mmaction2) to download the data, and preprocess the data using their tutorial. To skip all these steps, we provide a Google Drive download script, so you can download all the features, partition, and ablation files. We want to highlight that only 51 classes of data are with both audio and video data.

```
cd data
python download_ucf101.py
```


#### 1. (Optional) Partition the data

alpha specifies the non-iidness of the partition, the lower, the higher data heterogeneity. As each subject performs the same amount activities, we partition each subject data into 5 sub-clients.

```
python3 features/data_partitioning/ucf101/data_partition.py --alpha 0.1
python3 features/data_partitioning/ucf101/data_partition.py --alpha 5.0
```

The return data is a list, each item containing [key, file_name, label]

#### 2. (Optional) Feature extraction

For UCF-101 dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/ucf101/extract_feature.py --alpha 0.1
python3 features/feature_processing/ucf101/extract_feature.py --alpha 5.0
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/ucf101
# output/mm/ucf101/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/ucf101
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality
```
cd experiment/ucf101
bash run_mm.sh
```

#### Baseline results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
UCF-101 | Video+Audio | [UCF101-Data](https://www.crcv.ucf.edu/data/UCF101.php) | 51 | 100 | Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 75.13% <br> 75.89% <br> 74.53% <br> 75.05% | 0.05 | 200 |

