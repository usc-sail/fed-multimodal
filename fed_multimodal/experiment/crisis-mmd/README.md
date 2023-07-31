
### How to run CRISIS-MMD Dataset in FedMultimodal (Image and Text)
Here we provide an example to quickly start with the experiments, and reproduce the CRISIS-MMD results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.

#### This is a very challenging dataset, and we aim to improve the current results with better pre-trained architectures.


#### 0. Download data: The data will be under data/crisis-mmd by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_crisismmd.sh
cd ..
```

Data will be under data/crisis-mmd

#### 1. Partition the data

We partition the data using direchlet distribution.

```
# Low data heterogeneity
python3 features/data_partitioning/crisis-mmd/data_partition.py --alpha 5.0

# High data heterogeneity
python3 features/data_partitioning/crisis-mmd/data_partition.py --alpha 0.1
```

The return data is a list, each item containing [key, img_file, label, text_data], e.g.:

```
[
    "920300145253613569_0",
    "/home/tiantiaf/fed-multimodal/fed_multimodal/data/crisis-mmd/CrisisMMD_v2.0/data_image/california_wildfires/17_10_2017/920300145253613569_0.jpg",
    0,
    "Teams, players send young fan new memorabilia to replace collection lost in California fire"
],
```

#### 2. Feature extraction

For Crisis-MMD dataset, the feature extraction includes text/visual feature extraction.

```
# extract mobilenet_v2 feature
python3 features/feature_processing/crisis-mmd/extract_img_feature.py --feature_type mobilenet_v2 --alpha 5.0
python3 features/feature_processing/crisis-mmd/extract_img_feature.py --feature_type mobilenet_v2 --alpha 0.1

# extract mobile-bert feature
python3 features/feature_processing/crisis-mmd/extract_text_feature.py --feature_type mobilebert --alpha 5.0
python3 features/feature_processing/crisis-mmd/extract_text_feature.py --feature_type mobilebert --alpha 0.1
```

#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/crisis-mmd
# output/mm/crisis-mmd/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/crisis-mmd
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality (the same for missing labels and label noises)
```
cd experiment/crisis-mmd
bash run_mm.sh
```

#### Baseline results for executing the above - We are aware that previous studies have also reported AUC, feel free to add evaluation metric using AUC
Dataset | Modality | Link | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | Macro-F1 (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
Crisis-MMD | Image/Text | [Crisis-Data](https://crisisnlp.qcri.org/crisismmd) | 8 | 100 | Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 39.89% <br> 38.74% <br> 8.93% <br> 27.59% | 0.05 | 200 |

