
### How to run PTB-XL Dataset in FedMultimodal (I-AVF and V1-V6)
Here we provide an example to quickly start with the experiments, and reproduce the PTB-XL results. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.

#### Disclaimer: We used the v2 data in the paper, so the performance number might vary for a little, but any conclusion would stay the same. While PTB-XL release a v3 makes some data corrections, we highly recommend the reader to use v3 data.


#### 0. Download data: The data will be under data/ptb-xl by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_ptbxl.sh
cd ..
```

Data will be under data/ptb-xl

#### 1. Partition the data

This data has the natural partition (by site), so we do not use any simulating.

```
python3 features/data_partitioning/ptb-xl/data_partition.py
```

The return data is a list, each item containing [key, file_name, label], e.g.:

```
[
    "dev/records100/00000/00008_lr",
    "records100/00000/00008_lr",
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]
]
```

#### 2. Feature extraction

For PTB-XL dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/ptb-xl/extract_feature.py
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/ptb-xl
# output/mm/ptb-xl/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/ptb-xl
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality (the same for missing labels and label noises)
```
cd experiment/ptb-xl
bash run_mm.sh
```

#### Baseline results for executing the above - We are aware that previous studies have also reported AUC, feel free to add evaluation metric using AUC. This result is different than what we reported in the paper (ptb-xl v2), as we used ptb-xl v3 here.
Dataset | Modality | Link | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | Macro-F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
PTB-XL | I-AVF/V1-V6 | [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) | 5 | 33 | Natural | - |  FedAvg <br> FedOpt | 61.88% <br> 63.16% | 0.05 | 200 |

