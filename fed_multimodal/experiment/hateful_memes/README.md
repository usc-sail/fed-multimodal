
### How to run Hateful-Memes Dataset in FedMultimodal (Image and Text)
Here we provide an example to quickly start with the experiments, and reproduce the Hateful-Memes results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results (see Table 4, attention-based column) as reported from our paper.

#### This is a very challenging dataset, and we aim to improve the current results with better pre-trained architectures.


#### 0. Download data: The data will be under data/hateful_memes by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_hateful_memes.sh
cd ..
```

Data will be under data/hateful_memes

#### 1. Partition the data

We partition the data using direchlet distribution.

```
# Low data heterogeneity
python3 features/data_partitioning/hateful_memes/data_partition.py --alpha 5.0

# High data heterogeneity
python3 features/data_partitioning/hateful_memes/data_partition.py --alpha 0.1
```

The return data is a list, each item containing [key, img_file, label, text_data].


#### 2. Feature extraction

For Hateful-Memes dataset, the feature extraction includes text/visual feature extraction.

```
# extract mobilenet_v2 feature
python3 features/feature_processing/hateful_memes/extract_img_feature.py --feature_type mobilenet_v2 --alpha 5.0
python3 features/feature_processing/hateful_memes/extract_img_feature.py --feature_type mobilenet_v2 --alpha 0.1

# extract mobile-bert feature
python3 features/feature_processing/hateful_memes/extract_text_feature.py --feature_type mobilebert --alpha 5.0
python3 features/feature_processing/hateful_memes/extract_text_feature.py --feature_type mobilebert --alpha 0.1
```

#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/hateful_memes
# output/mm/hateful_memes/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```
The return data is a list, each item containing:
[missing_modalityA, missing_modalityB, new_label, missing_label]

missing_modalityA and missing_modalityB indicates the flag of missing modality, new_label indicates erroneous label, and missing label indicates if the label is missing for a data.

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/hateful_memes
bash run_base.sh
```

#### 5. Run ablation experiments, e.g Missing Modality (the same for missing labels and label noises)
```
cd experiment/hateful_memes
bash run_mm.sh
```
