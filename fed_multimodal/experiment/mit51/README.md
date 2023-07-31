### Quick Start -- Moments in Time - Top 51 labels (Video and Audio)
Here we provide an example to quickly start with the experiments, and reproduce the MiT-51 results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.


#### 0. Download data

If you want start from scratch, please follow [MMAction2](https://github.com/open-mmlab/mmaction2) to download the data, and preprocess the data using their tutorial. We want to highlight that we keep top 51 classes of data with both audio and video data. You will need to go through the whole step setting up the data, and we cannot distributed the features as stated in data share agreement.


#### 1. Partition the data

alpha specifies the non-iidness of the partition, the lower, the higher data heterogeneity. As each subject performs the same amount activities, we partition each subject data into 5 sub-clients.

```
python3 features/data_partitioning/mit51/data_partition.py --alpha 0.1
python3 features/data_partitioning/mit51/data_partition.py --alpha 5.0
```

The return data is a list, each item containing [key, file_name, label]

#### 2. Feature extraction

For UCF-101 dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/mit51/extract_feature.py --alpha 0.1
python3 features/feature_processing/mit51/extract_feature.py --alpha 5.0
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/mit51
# output/mm/mit51/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/mit51
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality
```
cd experiment/mit51
bash run_mm.sh
```

#### Baseline results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | Acc (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
Moments in Time | Video+Audio | [MiT51-Data](http://moments.csail.mit.edu/) | 51 | 1000 | Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 33.90% <br> 35.62% <br> 32.41% <br> 35.35% | 0.05 | 300 |

